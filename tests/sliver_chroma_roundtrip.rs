//! Roundtrip gate for 4:1 sliver partitions (PARTITION_HORZ_4/VERT_4) under
//! chroma subsampling: a 4:2:0 encode that picks BLOCK_16X4/BLOCK_4X16 must
//! still be valid, decodable AV1 (zenrav1e#35 / zenavif#29 regression: the
//! chroma TU grid for those sizes truncated to zero iterations, so no chroma
//! TUs were written while conforming decoders parse a TX_8X4/TX_4X8 TU —
//! every 4:2:0 encode reaching HORZ_4/VERT_4 desynced). Proven against an
//! independent decoder (rav1d-safe). Self-contained synthetic frame; runs in
//! the normal test suite.
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Synthetic photo-like content with strong thin horizontal and vertical
/// structure: gradient bands and 4px-period stripes make 4:1 slivers
/// (BLOCK_16X4 from HORZ_4, BLOCK_4X16 from VERT_4) RD-attractive on
/// 16x16 parents, with enough chroma activity that the sliver blocks
/// carry non-skip chroma coefficients (the desync needs a coded chroma TU).
fn synth_striped(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (cw, ch) = (w / 2, h / 2);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      // Diagonal gradient base.
      let mut v = (32 + (i * 3 / 2 + j) / 4 % 160) as u8;
      // Thin horizontal bands (4px period) over the top half.
      if j < h / 2 && j % 8 < 4 {
        v = v.saturating_add(40);
      }
      // Thin vertical bands over the bottom half.
      if j >= h / 2 && i % 8 < 4 {
        v = v.saturating_sub(40);
      }
      y[j * w + i] = v;
    }
  }
  // Chroma follows the band structure so sliver blocks have real chroma
  // residual (a flat-chroma frame would let skip hide the missing TUs).
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      u[j * cw + i] = (96 + (i + j * 2) / 3 % 64) as u8;
      v[j * cw + i] = if (j * 2) < ch {
        110 + (j % 4 * 8) as u8
      } else {
        150 - (i % 4 * 8) as u8
      };
    }
  }
  (y, u, v)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
  let n = a.len().min(b.len());
  let mut s = 0u64;
  for i in 0..n {
    let d = a[i] as i64 - b[i] as i64;
    s += (d * d) as u64;
  }
  if s == 0 {
    100.0
  } else {
    10.0 * (255.0 * 255.0 / (s as f64 / n as f64)).log10()
  }
}

/// Encode one 4:2:0 still frame through the topdown partition path with the
/// given non-square partition threshold (`BLOCK_64X64` arms HORZ_4/VERT_4 on
/// 16x16..64x64 parents — the ravif/zenavif production shape; `BLOCK_8X8`
/// keeps them unreachable — the control).
fn encode(
  sy: &[u8], su: &[u8], sv: &[u8], w: usize, h: usize, q: usize,
  threshold: BlockSize,
) -> Vec<u8> {
  // Mirror the ravif/cavif `-s2` mid-quality production shape that first hit
  // the desync (zenavif#29): topdown partition path (the only one offering
  // PARTITION_HORZ_4/VERT_4), a 16x16 partition-range cap (so every
  // superblock must split down to 16x16 parents, where the 4-way candidates
  // are offered against NONE/HORZ/VERT), and simple prediction modes.
  let mut ss = SpeedSettings::from_preset(2);
  ss.partition.encode_bottomup = false;
  ss.partition.partition_range =
    PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_16X16);
  ss.prediction.prediction_modes = PredictionModesSetting::Simple;
  ss.partition.non_square_partition_max_threshold = threshold;
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: q,
    min_quantizer: q as u8,
    still_picture: true,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  f.planes[0].copy_from_raw_u8(sy, w, 1);
  f.planes[1].copy_from_raw_u8(su, w / 2, 1);
  f.planes[2].copy_from_raw_u8(sv, w / 2, 1);
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut out = Vec::new();
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => out.extend_from_slice(&pkt.data),
      Err(EncoderStatus::LimitReached) => break,
      Err(EncoderStatus::Encoded) => {}
      Err(e) => panic!("encode error: {e:?}"),
    }
  }
  assert!(!out.is_empty());
  out
}

/// Decodes a raw AV1 frame OBU stream with rav1d-safe, returning the three
/// planes with strides stripped.
fn decode(data: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut fr = dec.decode(data).expect("decode error");
  if fr.is_none() {
    fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
  }
  let frame = fr.expect("no decoded frame");
  assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
  let (cw, ch) = (w / 2, h / 2);
  let (mut dy, mut du, mut dv) =
    (vec![0u8; w * h], vec![0u8; cw * ch], vec![0u8; cw * ch]);
  match frame.planes() {
    rav1d_safe::Planes::Depth8(p) => {
      for (j, row) in p.y().rows().enumerate().take(h) {
        dy[j * w..(j + 1) * w].copy_from_slice(&row[..w]);
      }
      for (j, row) in p.u().expect("chroma plane").rows().enumerate().take(ch)
      {
        du[j * cw..(j + 1) * cw].copy_from_slice(&row[..cw]);
      }
      for (j, row) in p.v().expect("chroma plane").rows().enumerate().take(ch)
      {
        dv[j * cw..(j + 1) * cw].copy_from_slice(&row[..cw]);
      }
    }
    rav1d_safe::Planes::Depth16(p) => {
      for (j, row) in p.y().rows().enumerate().take(h) {
        for (i, &px) in row[..w].iter().enumerate() {
          dy[j * w + i] = px as u8;
        }
      }
      for (j, row) in p.u().expect("chroma plane").rows().enumerate().take(ch)
      {
        for (i, &px) in row[..cw].iter().enumerate() {
          du[j * cw + i] = px as u8;
        }
      }
      for (j, row) in p.v().expect("chroma plane").rows().enumerate().take(ch)
      {
        for (i, &px) in row[..cw].iter().enumerate() {
          dv[j * cw + i] = px as u8;
        }
      }
    }
  }
  (dy, du, dv)
}

#[test]
fn yuv420_with_4to1_slivers_is_decodable() {
  let (w, h) = (256usize, 256usize);
  let (sy, su, sv) = synth_striped(w, h);

  let mut any_diff = false;
  for q in [60usize, 100, 140] {
    let narrow = encode(&sy, &su, &sv, w, h, q, BlockSize::BLOCK_8X8);
    let wide = encode(&sy, &su, &sv, w, h, q, BlockSize::BLOCK_64X64);
    // Liveness: the wide threshold must actually change the stream (i.e.
    // rect/sliver partitions were chosen somewhere). Byte-equal streams
    // would mean HORZ_4/VERT_4 never fired and this test proves nothing.
    if wide != narrow {
      any_diff = true;
    }

    // Additive dev hook: dump the streams as IVF for external decoders
    // (aomdec conformance harnesses). Never gates or skips any assertion.
    if let Ok(dir) = std::env::var("SLIVER_TEST_DUMP_IVF") {
      for (name, data) in [("narrow", &narrow), ("wide", &wide)] {
        let mut ivf = Vec::new();
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes());
        ivf.extend_from_slice(&32u16.to_le_bytes());
        ivf.extend_from_slice(b"AV01");
        ivf.extend_from_slice(&(w as u16).to_le_bytes());
        ivf.extend_from_slice(&(h as u16).to_le_bytes());
        ivf.extend_from_slice(&25u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&1u32.to_le_bytes());
        ivf.extend_from_slice(&0u32.to_le_bytes());
        ivf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u64.to_le_bytes());
        ivf.extend_from_slice(data);
        std::fs::write(format!("{dir}/sliver_{name}_q{q}.ivf"), ivf).unwrap();
      }
    }

    let (ny, nu, nv) = decode(&narrow, w, h);
    let (wy, wu, wv) = decode(&wide, w, h);
    let (npy, npu, npv) = (psnr(&sy, &ny), psnr(&su, &nu), psnr(&sv, &nv));
    let (wpy, wpu, wpv) = (psnr(&sy, &wy), psnr(&su, &wu), psnr(&sv, &wv));

    println!(
      "q{q}: narrow={} bytes (y={npy:.2} u={npu:.2} v={npv:.2}), wide={} \
       bytes (y={wpy:.2} u={wpu:.2} v={wpv:.2})",
      narrow.len(),
      wide.len()
    );

    // A chroma-TU desync shows up as a catastrophic PSNR collapse on the
    // sliver-partitioned stream (the arithmetic decoder goes off the rails
    // at the first missing TU, so luma collapses too). The sliver stream
    // must decode in the same quality class as the sliver-free control.
    assert!(
      wpy > 25.0 && wpu > 25.0 && wpv > 25.0,
      "implausible sliver-stream quality at q{q}: y={wpy:.2} u={wpu:.2} \
       v={wpv:.2} — 4:2:0 sliver chroma likely desynced"
    );
    assert!(
      wpy >= npy - 3.0 && wpu >= npu - 3.0 && wpv >= npv - 3.0,
      "sliver stream decodes far worse than control at q{q} \
       (y {wpy:.2} vs {npy:.2}, u {wpu:.2} vs {npu:.2}, v {wpv:.2} vs \
       {npv:.2}) — 4:2:0 sliver chroma desynced"
    );
    assert!(npy > 10.0, "control baseline is garbage at q{q}: {npy:.2}");
  }
  assert!(
    any_diff,
    "wide and narrow thresholds produced identical streams at every q — \
     HORZ_4/VERT_4 never fired; the regression gate is vacuous"
  );
}
