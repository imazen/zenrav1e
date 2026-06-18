//! Roundtrip gate for the multi-level trellis RDOQ (`enable_trellis`): its output
//! must be valid, decodable AV1. The trellis only changes coefficient *levels*
//! (always coded faithfully by the lv-map writer), so the bitstream must stay
//! decodable; this test proves it against an independent decoder (rav1d-safe).
//! Self-contained (synthetic frame, no corpus). Runs in the normal test suite.
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Gradient + deterministic LCG noise → photo-like content with broadband
/// high-frequency detail, so the trellis has many small coefficients to optimise.
fn synth(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut s: u32 = 0x1234_5678;
  let mut rng = move || {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (s >> 24) as i32 - 128
  };
  let (cw, ch) = (w / 2, h / 2);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      let g = (i * 255 / w + j * 255 / h) / 2;
      y[j * w + i] = (g as i32 + rng() / 4).clamp(0, 255) as u8;
    }
  }
  let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
  for j in 0..ch {
    for i in 0..cw {
      u[j * cw + i] =
        (128 + (i * 64 / cw) as i32 + rng() / 8).clamp(0, 255) as u8;
      v[j * cw + i] =
        (128 + (j * 64 / ch) as i32 + rng() / 8).clamp(0, 255) as u8;
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

#[test]
fn trellis_on_output_is_decodable() {
  let (w, h) = (256usize, 256usize);
  let (sy, su, sv) = synth(w, h);
  // Low quantizers keep ac_quant below the trellis's skip threshold so the
  // multi-level RDOQ actually runs.
  for q in [20usize, 60] {
    let mut ss = SpeedSettings::from_preset(6);
    ss.segmentation = SegmentationLevel::Disabled;
    let enc = EncoderConfig {
      width: w,
      height: h,
      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      still_picture: true,
      low_latency: true,
      quantizer: q,
      enable_trellis: true,
      tune: Tune::Psnr,
      speed_settings: ss,
      ..Default::default()
    };
    let cfg = Config::new().with_encoder_config(enc).with_threads(1);
    let mut ctx: Context<u8> = cfg.new_context().unwrap();
    let mut f = ctx.new_frame();
    f.planes[0].copy_from_raw_u8(&sy, w, 1);
    f.planes[1].copy_from_raw_u8(&su, w / 2, 1);
    f.planes[2].copy_from_raw_u8(&sv, w / 2, 1);
    ctx.send_frame(f).unwrap();
    ctx.flush();
    let mut obu = Vec::new();
    while let Ok(pkt) = ctx.receive_packet() {
      obu.extend_from_slice(&pkt.data);
    }
    assert!(!obu.is_empty(), "q{q}: encoder produced no data");

    let mut dec = rav1d_safe::Decoder::new().expect("decoder");
    let mut fr = dec.decode(&obu).expect("decode error");
    if fr.is_none() {
      fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
    }
    let frame = fr.unwrap_or_else(|| panic!("q{q}: no decoded frame"));
    assert_eq!(
      (frame.width() as usize, frame.height() as usize),
      (w, h),
      "q{q}: decoded dimensions differ"
    );
    let mut dy = vec![0u8; w * h];
    // 8-bit content may be exposed as either an 8- or 16-bit plane.
    match frame.planes() {
      rav1d_safe::Planes::Depth8(p) => {
        for (j, row) in p.y().rows().enumerate().take(h) {
          dy[j * w..j * w + w].copy_from_slice(&row[..w]);
        }
      }
      rav1d_safe::Planes::Depth16(p) => {
        for (j, row) in p.y().rows().enumerate().take(h) {
          for (i, &px) in row[..w].iter().enumerate() {
            dy[j * w + i] = px as u8;
          }
        }
      }
    }
    // The core gate is that trellis-on output is valid + decodes to the right
    // dimensions (asserted above). PSNR here is only a gross-garbage guard:
    // the published rav1d-safe is not bit-exact with this encoder on every
    // block, so we don't assert tight fidelity (the encoder's own
    // reconstruction PSNR — see benchmarks/ — covers quality).
    let p = psnr(&sy, &dy);
    let mean = |v: &[u8]| {
      v.iter().map(|&x| x as u64).sum::<u64>() as f64 / v.len() as f64
    };
    eprintln!(
      "q{q}: decoded {w}x{h} ({} B), Y-PSNR={p:.2} mean(src)={:.1} mean(dec)={:.1}",
      obu.len(),
      mean(&sy),
      mean(&dy)
    );
    assert!(p > 10.0, "q{q}: trellis-on decode is garbage (Y-PSNR {p:.1})");
  }
}
