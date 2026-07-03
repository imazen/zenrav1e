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
  palette: PaletteMode,
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
  decode_ss(data, w, h, 1, 1)
}

/// [`decode`] generalized over chroma subsampling (`xdec`/`ydec` are the
/// chroma decimation shifts; chroma planes are `ceil`-sized like AV1).
fn decode_ss(
  data: &[u8], w: usize, h: usize, xdec: usize, ydec: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut fr = dec.decode(data).expect("decode error");
  if fr.is_none() {
    fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
  }
  let frame = fr.expect("no decoded frame");
  assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
  let (cw, ch) = ((w + xdec) >> xdec, (h + ydec) >> ydec);
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
    let off = encode(&sy, &su, &sv, w, h, q, speed, PaletteMode::Off);
    let on = encode(&sy, &su, &sv, w, h, q, speed, PaletteMode::Always);
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

#[test]
fn palette_auto_mode_follows_screen_content_detection() {
  let (w, h) = (256usize, 256usize);

  // Screen content: Auto must behave like Always (detection fires).
  let (sy, su, sv) = synth_screen(w, h);
  let always = encode(&sy, &su, &sv, w, h, 100, 6, PaletteMode::Always);
  let auto = encode(&sy, &su, &sv, w, h, 100, 6, PaletteMode::Auto);
  assert_eq!(
    auto, always,
    "Auto must equal Always on screen content (detection should fire)"
  );

  // Photo-like noise: Auto must disable screen content tools entirely; the
  // stream then differs from Off (which still signals scc for still
  // pictures and writes per-block palette=false flags) but must stay
  // decodable and quality-equivalent.
  let mut s: u32 = 0x1234_5678;
  let mut rng = move || {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (s >> 24) as u8
  };
  let ny: Vec<u8> = (0..w * h).map(|_| rng()).collect();
  let nu: Vec<u8> = (0..w * h / 4).map(|_| rng()).collect();
  let nv: Vec<u8> = (0..w * h / 4).map(|_| rng()).collect();
  let off = encode(&ny, &nu, &nv, w, h, 100, 6, PaletteMode::Off);
  let auto = encode(&ny, &nu, &nv, w, h, 100, 6, PaletteMode::Auto);
  // scc=0 saves the per-DC-block false flags, so auto <= off in bytes.
  assert!(
    auto.len() <= off.len(),
    "photo Auto ({}) should not exceed Off ({}): scc=0 drops flag bits",
    auto.len(),
    off.len()
  );
  let (dy, _, _) = decode(&auto, w, h);
  let (oy, _, _) = decode(&off, w, h);
  let (pa, po) = (psnr(&ny, &dy), psnr(&ny, &oy));
  assert!(
    (pa - po).abs() < 0.35,
    "photo Auto vs Off quality diverged: {pa:.2} vs {po:.2}"
  );
}

/// 10-bit palette roundtrip: the color coding writes bpc-bit literals and
/// delta widths derived from bit_depth; prove the whole path end-to-end at
/// depth 10 (encode + rav1d-safe decode + exactness on palette-exact
/// content).
#[test]
fn palette_10bit_roundtrip() {
  let (w, h) = (128usize, 128usize);
  let (sy8, su8, sv8) = synth_screen(w, h);

  let mut ss = SpeedSettings::from_preset(6);
  ss.prediction.palette = PaletteMode::Always;
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: 80,
    min_quantizer: 80,
    still_picture: true,
    bit_depth: 10,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u16> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  // Shift the 8-bit synthetic screen content to 10-bit. Plane data is
  // origin-padded; copy_from_raw_u8 handles the layout.
  let srcs: [(&[u8], usize); 3] = [(&sy8, w), (&su8, w / 2), (&sv8, w / 2)];
  for (plane, (src, pw)) in f.planes.iter_mut().zip(srcs) {
    let widened: Vec<u8> =
      src.iter().flat_map(|&v| (u16::from(v) << 2).to_le_bytes()).collect();
    plane.copy_from_raw_u8(&widened, pw * 2, 2);
  }
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut obu = Vec::new();
  while let Ok(pkt) = ctx.receive_packet() {
    obu.extend_from_slice(&pkt.data);
  }
  assert!(!obu.is_empty());
  eprintln!("10bit palette-on bytes: {}", obu.len());

  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut fr = dec.decode(&obu).expect("decode error");
  if fr.is_none() {
    fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
  }
  let frame = fr.expect("no decoded frame");
  assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
  // Luma must reconstruct the palette-exact content perfectly at q80.
  let mut exact = 0usize;
  let mut total = 0usize;
  match frame.planes() {
    rav1d_safe::Planes::Depth16(p) => {
      for (j, row) in p.y().rows().enumerate().take(h) {
        for (i, &px) in row[..w].iter().enumerate() {
          total += 1;
          if px == (sy8[j * w + i] as u16) << 2 {
            exact += 1;
          }
        }
      }
    }
    rav1d_safe::Planes::Depth8(_) => panic!("expected 10-bit output"),
  }
  assert!(
    exact as f64 >= total as f64 * 0.99,
    "10-bit palette luma should be (near-)exact on palette-exact content: \
     {exact}/{total}"
  );
}

// ---------------------------------------------------------------------------
// Chroma (UV) palette roundtrips.
// ---------------------------------------------------------------------------

/// Synthetic *chromatic* screen content: blocky few-color luma plus chroma
/// striped through four (U, V) pairs chosen to exercise every UV coding
/// path — a duplicate U value across two pairs (legal only because the U
/// deltas have minimum step 0) and a V jump of 190 (30 <-> 220) whose
/// wraparound distance (66) is shorter than the direct one, forcing the
/// complement-coded V delta branch.
fn synth_chroma_screen(
  w: usize, h: usize, xdec: usize, ydec: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  const PAIRS: [(u8, u8); 4] = [(60, 200), (100, 30), (100, 220), (180, 128)];
  let (cw, ch) = ((w + xdec) >> xdec, (h + ydec) >> ydec);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      // Blocky few-color luma (keeps screen-content coding plausible and
      // gives the luma palette something to win on too).
      let region = (i / 32 + (j / 16) * 3) % 5;
      y[j * w + i] = [16u8, 235, 80, 160, 40][region];
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      // Chroma stripes cycling through the four pairs every 8 chroma px in
      // both axes: several distinct pairs inside every block, so a flat DC
      // prediction is poor and the joint palette pays for itself.
      let sel = ((i / 8) + (j / 8) * 2) % PAIRS.len();
      u[j * cw + i] = PAIRS[sel].0;
      v[j * cw + i] = PAIRS[sel].1;
    }
  }
  (y, u, v)
}

fn encode_ss(
  sy: &[u8], su: &[u8], sv: &[u8], w: usize, h: usize, q: usize, speed: u8,
  palette: PaletteMode, cs: ChromaSampling,
) -> Vec<u8> {
  let (xdec, ydec) = match cs {
    ChromaSampling::Cs420 => (1, 1),
    ChromaSampling::Cs422 => (1, 0),
    ChromaSampling::Cs444 => (0, 0),
    ChromaSampling::Cs400 => unreachable!(),
  };
  let cw = (w + xdec) >> xdec;
  let _ = (h + ydec) >> ydec;
  let mut ss = SpeedSettings::from_preset(speed);
  ss.prediction.palette = palette;
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: q,
    min_quantizer: q as u8,
    still_picture: true,
    chroma_sampling: cs,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  f.planes[0].copy_from_raw_u8(sy, w, 1);
  f.planes[1].copy_from_raw_u8(su, cw, 1);
  f.planes[2].copy_from_raw_u8(sv, cw, 1);
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

/// The joint UV palette must fire on chroma-palettizable content and
/// round-trip the chroma planes (near-)exactly through rav1d-safe, at BOTH
/// chroma samplings (the 4:2:0 sliver zone is bug-class territory) and both
/// partition search paths (s6 topdown, s2 bottomup). Exact chroma at q80 is
/// decisive evidence the tool fired: hard chroma stripes DCT-coded under a
/// flat DC prediction ring and cannot reconstruct exactly, while a
/// palette-exact joint palette codes an all-zero residual.
#[test]
fn uv_palette_roundtrip_exact_chroma_both_samplings() {
  let (w, h) = (256usize, 256usize);
  for (cs, xdec, ydec) in
    [(ChromaSampling::Cs420, 1usize, 1usize), (ChromaSampling::Cs444, 0, 0)]
  {
    let (sy, su, sv) = synth_chroma_screen(w, h, xdec, ydec);
    for speed in [6u8, 2] {
      let off =
        encode_ss(&sy, &su, &sv, w, h, 80, speed, PaletteMode::Off, cs);
      let on =
        encode_ss(&sy, &su, &sv, w, h, 80, speed, PaletteMode::Always, cs);
      assert_ne!(
        on, off,
        "palette Always == Off bytes on chroma screen content \
         ({cs:?} s{speed}) — the palette search never fired"
      );

      // Additive dev hook (see the luma test): dump for external decoders.
      if let Ok(dir) = std::env::var("PALETTE_TEST_DUMP_IVF") {
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
        ivf.extend_from_slice(&(on.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&0u64.to_le_bytes());
        ivf.extend_from_slice(&on);
        std::fs::write(format!("{dir}/uvpal_{cs:?}_s{speed}.ivf"), ivf)
          .unwrap();
        let mut src = sy.clone();
        src.extend_from_slice(&su);
        src.extend_from_slice(&sv);
        std::fs::write(format!("{dir}/uvpal_{cs:?}_s{speed}.src.yuv"), src)
          .unwrap();
      }

      let (dy, du, dv) = decode_ss(&on, w, h, xdec, ydec);
      if let Ok(dir) = std::env::var("PALETTE_TEST_DUMP_IVF") {
        let mut yuv = dy.clone();
        yuv.extend_from_slice(&du);
        yuv.extend_from_slice(&dv);
        std::fs::write(format!("{dir}/uvpal_{cs:?}_s{speed}.rav1d.yuv"), yuv)
          .unwrap();
      }
      let exact_frac = |a: &[u8], b: &[u8]| -> f64 {
        let n = a.len().min(b.len());
        let eq = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        eq as f64 / n as f64
      };
      let (ey, eu, ev) =
        (exact_frac(&sy, &dy), exact_frac(&su, &du), exact_frac(&sv, &dv));
      println!(
        "{cs:?} s{speed}: off={} on={} bytes, exact y={ey:.4} u={eu:.4} \
         v={ev:.4}",
        off.len(),
        on.len()
      );
      assert!(
        eu >= 0.99 && ev >= 0.99,
        "chroma should be (near-)exact via the UV palette on palette-exact \
         content ({cs:?} s{speed}): u={eu:.4} v={ev:.4} — either the UV \
         palette never fired or its recon/coding diverged"
      );
      // And the decode must agree with the source luma too (the luma
      // palette handles that side).
      assert!(
        ey >= 0.99,
        "luma should be (near-)exact ({cs:?} s{speed}): {ey:.4}"
      );
    }
  }
}

/// Odd-dimension and tiny frames stay decodable with palette on (the
/// search skips partial edge blocks; the coded UV flag path must still be
/// conformant everywhere), at both samplings.
#[test]
fn uv_palette_odd_dims_and_tiny_frames_decodable() {
  for (w, h) in [(66usize, 34usize), (34, 18), (64, 64), (32, 32)] {
    for (cs, xdec, ydec) in
      [(ChromaSampling::Cs420, 1usize, 1usize), (ChromaSampling::Cs444, 0, 0)]
    {
      let (sy, su, sv) = synth_chroma_screen(w, h, xdec, ydec);
      let on = encode_ss(&sy, &su, &sv, w, h, 80, 6, PaletteMode::Always, cs);
      let (dy, _du, _dv) = decode_ss(&on, w, h, xdec, ydec);
      let py = psnr(&sy, &dy);
      assert!(
        py > 25.0,
        "implausible decode at {w}x{h} {cs:?}: y-psnr {py:.2}"
      );
    }
  }
}

/// 10-bit joint UV palette: V wraparound arithmetic and U duplicate coding
/// operate on bpc-wide literals; prove end-to-end at depth 10 with
/// (near-)exact chroma.
#[test]
fn uv_palette_10bit_roundtrip() {
  let (w, h) = (128usize, 128usize);
  let (sy8, su8, sv8) = synth_chroma_screen(w, h, 1, 1);

  let mut ss = SpeedSettings::from_preset(6);
  ss.prediction.palette = PaletteMode::Always;
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: 80,
    min_quantizer: 80,
    still_picture: true,
    bit_depth: 10,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u16> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  let srcs: [(&[u8], usize); 3] = [(&sy8, w), (&su8, w / 2), (&sv8, w / 2)];
  for (plane, (src, pw)) in f.planes.iter_mut().zip(srcs) {
    let widened: Vec<u8> =
      src.iter().flat_map(|&v| (u16::from(v) << 2).to_le_bytes()).collect();
    plane.copy_from_raw_u8(&widened, pw * 2, 2);
  }
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut obu = Vec::new();
  while let Ok(pkt) = ctx.receive_packet() {
    obu.extend_from_slice(&pkt.data);
  }
  assert!(!obu.is_empty());

  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut fr = dec.decode(&obu).expect("decode error");
  if fr.is_none() {
    fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
  }
  let frame = fr.expect("no decoded frame");
  let (cw, ch) = (w / 2, h / 2);
  let mut exact = 0usize;
  let mut total = 0usize;
  let mut mism: Vec<(usize, usize, usize, u16, u16)> = Vec::new();
  match frame.planes() {
    rav1d_safe::Planes::Depth16(p) => {
      for (pl, (plane, src)) in
        [(p.u(), &su8), (p.v(), &sv8)].into_iter().enumerate()
      {
        for (j, row) in
          plane.expect("chroma plane").rows().enumerate().take(ch)
        {
          for (i, &px) in row[..cw].iter().enumerate() {
            total += 1;
            if px == (src[j * cw + i] as u16) << 2 {
              exact += 1;
            } else if mism.len() < 24 {
              mism.push((pl, j, i, (src[j * cw + i] as u16) << 2, px));
            }
          }
        }
      }
    }
    rav1d_safe::Planes::Depth8(_) => panic!("expected 10-bit output"),
  }
  for (pl, j, i, s, d) in &mism {
    eprintln!("mismatch plane{} ({j},{i}): src={s} dec={d}", pl + 1);
  }
  assert!(
    exact as f64 >= total as f64 * 0.99,
    "10-bit UV palette chroma should be (near-)exact: {exact}/{total}"
  );
}
