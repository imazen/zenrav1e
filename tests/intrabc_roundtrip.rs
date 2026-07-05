//! Roundtrip gate for AV1 intra block copy
//! (`speed_settings.prediction.intrabc`): intraBC-on output must be valid,
//! decodable AV1 that reconstructs what the encoder's own recon holds —
//! proven against an independent decoder (rav1d-safe). Self-contained
//! (synthetic repeating screen-content frame, no corpus).
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Synthetic content intraBC exists for: an exactly-repeating "glyph sheet"
/// whose tiles carry far too many distinct values for palette mode (smooth
/// per-tile shading), so the only cheap description of a tile is a copy of
/// an earlier identical tile.
fn synth_repeating(
  w: usize, h: usize, xdec: usize, ydec: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  const P: usize = 64; // repeat period, px
  let (cw, ch) = ((w + xdec) >> xdec, (h + ydec) >> ydec);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      let (u, v) = (i % P, j % P);
      // "Glyph": diagonal shading + a few hard strokes; > 64 distinct
      // values per 16x16 region, exactly repeating with period P.
      let mut px = (40 + ((u * 3 + v * 2) % 173)) as u8;
      if (u / 7 + v / 5) % 4 == 0 {
        px = 16;
      }
      if u % 13 == 0 || v % 11 == 0 {
        px = 224;
      }
      y[j * w + i] = px;
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      let (a, b) = ((i << xdec) % P, (j << ydec) % P);
      u[j * cw + i] = (100 + ((a * 2 + b) % 97)) as u8;
      v[j * cw + i] = (90 + ((a + b * 2) % 101)) as u8;
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

#[allow(clippy::too_many_arguments)]
fn encode(
  sy: &[u8], su: &[u8], sv: &[u8], w: usize, h: usize, q: usize, speed: u8,
  intrabc: bool, cs: ChromaSampling,
) -> Vec<u8> {
  encode_hash(sy, su, sv, w, h, q, speed, intrabc, true, cs)
}

#[allow(clippy::too_many_arguments)]
fn encode_hash(
  sy: &[u8], su: &[u8], sv: &[u8], w: usize, h: usize, q: usize, speed: u8,
  intrabc: bool, intrabc_hash: bool, cs: ChromaSampling,
) -> Vec<u8> {
  let (xdec, _ydec) = match cs {
    ChromaSampling::Cs420 => (1, 1),
    ChromaSampling::Cs444 => (0, 0),
    _ => unreachable!(),
  };
  let cw = (w + xdec) >> xdec;
  let mut ss = SpeedSettings::from_preset(speed);
  ss.prediction.palette = PaletteMode::Always;
  ss.prediction.intrabc = intrabc;
  ss.prediction.intrabc_hash = intrabc_hash;
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

fn decode_y(data: &[u8], w: usize, h: usize) -> Vec<u8> {
  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut fr = dec.decode(data).expect("decode error");
  if fr.is_none() {
    fr = dec.flush().ok().and_then(|mut v| v.drain(..).next());
  }
  let frame = fr.expect("no decoded frame");
  assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
  let mut dy = vec![0u8; w * h];
  match frame.planes() {
    rav1d_safe::Planes::Depth8(p) => {
      for (j, row) in p.y().rows().enumerate().take(h) {
        dy[j * w..(j + 1) * w].copy_from_slice(&row[..w]);
      }
    }
    rav1d_safe::Planes::Depth16(_) => panic!("expected 8-bit"),
  }
  dy
}

/// intraBC must fire on exactly-repeating non-palettizable content, shrink
/// the stream, and round-trip through rav1d-safe at sane quality — at both
/// chroma samplings and both partition search paths.
#[test]
fn intrabc_fires_and_roundtrips_both_samplings() {
  let (w, h) = (256usize, 256usize);
  for (cs, xdec, ydec) in
    [(ChromaSampling::Cs420, 1usize, 1usize), (ChromaSampling::Cs444, 0, 0)]
  {
    let (sy, su, sv) = synth_repeating(w, h, xdec, ydec);
    for speed in [6u8, 2] {
      let off = encode(&sy, &su, &sv, w, h, 100, speed, false, cs);
      let on = encode(&sy, &su, &sv, w, h, 100, speed, true, cs);
      assert_ne!(
        on, off,
        "intraBC on == off bytes on repeating content ({cs:?} s{speed}) — \
         the search never chose the tool"
      );
      // Exactly-repeating content: block copies should shrink the stream.
      assert!(
        on.len() < off.len(),
        "intraBC should shrink exactly-repeating content ({cs:?} \
         s{speed}): on={} off={}",
        on.len(),
        off.len()
      );
      let dy = decode_y(&on, w, h);
      let py = psnr(&sy, &dy);
      let dy_off = decode_y(&off, w, h);
      let py_off = psnr(&sy, &dy_off);
      println!(
        "{cs:?} s{speed}: off={} on={} bytes ({:.3}x), y-psnr on={py:.2} \
         off={py_off:.2}",
        off.len(),
        on.len(),
        on.len() as f64 / off.len() as f64
      );
      assert!(
        py > 25.0,
        "implausible intraBC-on quality ({cs:?} s{speed}): {py:.2} — recon \
         likely diverged"
      );
      assert!(
        py >= py_off - 1.0,
        "intraBC-on decodes much worse than off ({cs:?} s{speed}): {py:.2} \
         vs {py_off:.2}"
      );
    }
  }
}

/// Odd-dimension and tiny frames stay decodable with intraBC on (the
/// search skips partial edge blocks; DV validity plus the coded flag path
/// must be conformant everywhere).
#[test]
fn intrabc_odd_dims_and_tiny_frames_decodable() {
  for (w, h) in [(130usize, 66usize), (66, 130), (64, 64)] {
    let (sy, su, sv) = synth_repeating(w, h, 1, 1);
    let on = encode(&sy, &su, &sv, w, h, 100, 6, true, ChromaSampling::Cs420);
    let dy = decode_y(&on, w, h);
    let py = psnr(&sy, &dy);
    assert!(py > 20.0, "implausible decode at {w}x{h}: y-psnr {py:.2}");
  }
}

/// A frame whose only repeat is one distinctive 64x64 texture stamp far
/// from its copy, surrounded by unrelated noise: the local seeds + diamond
/// have no gradient to follow, so finding the copy is the hash search's
/// job (chunk B). Hash-on must beat hash-off bytes and round-trip cleanly
/// at both samplings.
fn synth_long_range(
  w: usize, h: usize, xdec: usize, ydec: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (cw, ch) = ((w + xdec) >> xdec, (h + ydec) >> ydec);
  let mut state = 0x2545_f491u32;
  let mut rnd = move || {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    (state >> 24) as u8
  };
  let mut y = vec![0u8; w * h];
  for px in y.iter_mut() {
    *px = rnd();
  }
  // One deterministic "stamp" texture, placed at (0,0) and again at
  // (192, 192) — same tile-relative parity on both axes (chroma-fullpel
  // rule for 4:2:0), far outside any seed/diamond basin in pure noise.
  for j in 0..64 {
    for i in 0..64 {
      let v =
        (30 + ((i * 5 + j * 3) % 190) + ((i / 9 + j / 7) % 2) * 25) as u8;
      y[j * w + i] = v;
      y[(192 + j) * w + 192 + i] = v;
    }
  }
  // Flat chroma keeps the repeat exact in every plane.
  let u = vec![120u8; cw * ch];
  let v = vec![132u8; cw * ch];
  (y, u, v)
}

#[test]
fn intrabc_hash_finds_long_range_repeats() {
  let (w, h) = (320usize, 320usize);
  for (cs, xdec, ydec) in
    [(ChromaSampling::Cs420, 1usize, 1usize), (ChromaSampling::Cs444, 0, 0)]
  {
    let (sy, su, sv) = synth_long_range(w, h, xdec, ydec);
    let off = encode_hash(&sy, &su, &sv, w, h, 100, 6, true, false, cs);
    let on = encode_hash(&sy, &su, &sv, w, h, 100, 6, true, true, cs);
    println!(
      "{cs:?}: hash-off={} hash-on={} bytes ({:.3}x)",
      off.len(),
      on.len(),
      on.len() as f64 / off.len() as f64
    );
    assert!(
      on.len() < off.len(),
      "hash search should shrink the long-range repeat ({cs:?}): on={} \
       off={}",
      on.len(),
      off.len()
    );
    let dy = decode_y(&on, w, h);
    let py = psnr(&sy, &dy);
    assert!(py > 25.0, "implausible hash-on quality ({cs:?}): y-psnr {py:.2}");
  }
}
