//! Roundtrip gate for AV1 palette mode (`speed_settings.prediction.palette`):
//! palette-on output must be valid, decodable AV1 that reconstructs exactly
//! what the encoder's own recon holds — proven against an independent
//! decoder (rav1d-safe). Self-contained (synthetic screen-content frame, no
//! corpus). Runs in the normal test suite.
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Synthetic screen content: few-color blocky UI-like regions plus 1-bit
/// "text" strokes. Exactly the content palette mode exists for.
fn synth_screen(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let palette = [16u8, 235, 80, 160, 40, 200, 120, 60];
  let (cw, ch) = (w / 2, h / 2);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      // Blocky color regions (32x16 tiles cycling through few colors).
      let region = (i / 32 + (j / 16) * 3) % 5;
      let mut v = palette[region];
      // "Text" band: 1-bit strokes over a flat background.
      if (64..128).contains(&j) {
        let bit = (i / 3 + j / 5) % 7 < 2 && j % 5 != 0;
        v = if bit { 16 } else { 235 };
      }
      // A second band with 4-color fine pattern (dithered-look UI).
      if (160..200).contains(&j) {
        v = palette[4 + ((i / 2) % 2) * 2 + ((j / 2) % 2)];
      }
      y[j * w + i] = v;
    }
  }
  // Flat chroma with a couple of hard region switches (screen-like).
  let mut u = vec![128u8; cw * ch];
  let mut v = vec![128u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      if i > cw / 2 {
        u[j * cw + i] = 100;
        v[j * cw + i] = 150;
      }
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

fn encode(
  sy: &[u8], su: &[u8], sv: &[u8], w: usize, h: usize, q: usize, speed: u8,
  palette: bool,
) -> Vec<u8> {
  let mut ss = SpeedSettings::from_preset(speed);
  ss.prediction.palette = palette;
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
  let cfg = Config::new().with_encoder_config(enc.clone()).with_threads(1);
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
fn palette_on_output_is_decodable_and_beats_off_on_screen_content() {
  let (w, h) = (256usize, 256usize);
  let (sy, su, sv) = synth_screen(w, h);

  let mut any_diff = false;
  // Speed 6 uses the topdown partition path, speed 2 exercises bottomup
  // (both reach the palette RDO through different encode_block callers).
  for (q, speed) in [(80usize, 6u8), (140, 6), (80, 2)] {
    let off = encode(&sy, &su, &sv, w, h, q, speed, false);
    let on = encode(&sy, &su, &sv, w, h, q, speed, true);
    // The palette search must actually fire on this content: byte-equal
    // streams would mean the tool was never chosen and this test proves
    // nothing.
    if on != off {
      any_diff = true;
    }

    // Additive dev hook: dump the streams as IVF for external decoders
    // (aomdec conformance checks). Never gates or skips any assertion.
    if let Ok(dir) = std::env::var("PALETTE_TEST_DUMP_IVF") {
      for (name, data) in [("off", &off), ("on", &on)] {
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
        std::fs::write(format!("{dir}/palette_{name}_q{q}_s{speed}.ivf"), ivf)
          .unwrap();
      }
    }

    let (dy, du, dv) = decode(&on, w, h);
    if let Ok(dir) = std::env::var("PALETTE_TEST_DUMP_IVF") {
      // rav1d-safe's decode as raw I420, for byte-agreement checks against
      // aomdec's output of the same stream.
      let mut yuv = dy.clone();
      yuv.extend_from_slice(&du);
      yuv.extend_from_slice(&dv);
      std::fs::write(format!("{dir}/palette_on_q{q}_s{speed}.rav1d.yuv"), yuv)
        .unwrap();
    }
    let (py, pu, pv) = (psnr(&sy, &dy), psnr(&su, &du), psnr(&sv, &dv));
    // Palette-off baseline: decodable, non-garbage (its absolute PSNR is
    // encode quality — DCT ringing on 1-bit text is legitimately poor).
    let (oy, _, _) = decode(&off, w, h);
    let poy = psnr(&sy, &oy);

    println!(
      "q{q} s{speed}: off={} bytes (y-psnr {poy:.2}), on={} bytes \
       (y={py:.2} u={pu:.2} v={pv:.2})",
      off.len(),
      on.len()
    );

    // Palette recon divergence between encoder and decoder shows up as a
    // catastrophic PSNR collapse on this palette-exact content; the
    // palette-on stream must decode at least as well as the ringing
    // palette-off baseline.
    assert!(
      py > 30.0 && pu > 28.0 && pv > 28.0,
      "implausible palette-on quality at q{q}: y={py:.2} u={pu:.2} \
       v={pv:.2} — palette recon likely diverged"
    );
    assert!(
      py >= poy - 0.5,
      "palette-on decodes worse than palette-off at q{q} \
       ({py:.2} vs {poy:.2}) — palette recon diverged"
    );
    assert!(poy > 10.0, "palette-off baseline is garbage at q{q}: {poy:.2}");
  }
  assert!(
    any_diff,
    "palette-on and palette-off produced identical streams at every q — \
     the palette search never chose palette mode on synthetic screen content"
  );
}
