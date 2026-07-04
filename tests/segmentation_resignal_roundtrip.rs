//! Regression + conformance gate for the segmentation re-signal fallback
//! (zenrav1e#31): an INTER frame whose primary ref carries no usable ALT_Q
//! segmentation data must re-signal fresh data (`update_data = 1`) instead of
//! panicking at `segmentation.rs` ("assertion `left != right` failed: 8 == 8").
//!
//! The deterministic trigger is the Tune::Ssimulacra2 variance-boost path: on
//! KEY/intra frames with `base_q_idx > 0` it codes per-SB delta-q and
//! dynamically disables segmentation, so the keyframe's `ReferenceFrame`
//! carries a default (all-features-false) `SegmentationState`. The next INTER
//! frame has segmentation enabled again, inherits that unusable state via
//! `get_initial_segmentation`, and — before the fix — hit the
//! `assert_ne!(min_segment, MAX_SEGMENTS)` in `segmentation_optimize`. The
//! same fallback also covers the rate-control shape (base_q_idx dropping
//! between frames lifts the lossless floor above every stored ALT_Q delta).
//!
//! Since the fallback writes new header bits (segmentation_update_data = 1 on
//! a frame with a primary ref), the encode is verified against an independent
//! decoder (rav1d-safe), not just for absence of panic.
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Synthetic photo-like content with per-frame motion so inter frames carry
/// real residual (an all-static sequence could degenerate to skip frames).
fn synth_frame(w: usize, h: usize, t: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (cw, ch) = (w / 2, h / 2);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      // Diagonal gradient sliding by 3px per frame + a variance-rich
      // texture band (the variance boost needs non-flat activity).
      let base = (i * 2 + j + t * 3) % 224;
      let tex = if (i / 4 + j / 4) % 2 == 0 { 24 } else { 0 };
      y[j * w + i] = (16 + base / 2 + tex) as u8;
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      u[j * cw + i] = (96 + (i + t) % 64) as u8;
      v[j * cw + i] = (160 - (j + t) % 64) as u8;
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
fn ssimulacra2_tune_inter_segmentation_resignals_and_decodes() {
  let (w, h) = (64, 64);
  let frames = 4usize;

  // Tune::Ssimulacra2 + lossy quantizer: the keyframe activates the
  // variance boost (disabling segmentation for that frame); the inter
  // frames keep the default SegmentationLevel (enabled) and must cope with
  // the keyframe's empty segmentation state. low_latency gives the plain
  // previous-frame primary-ref chain.
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: SpeedSettings::from_preset(6),
    quantizer: 100,
    still_picture: false,
    low_latency: true,
    tune: Tune::Ssimulacra2,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();

  let mut inputs = Vec::new();
  for t in 0..frames {
    let (sy, su, sv) = synth_frame(w, h, t);
    let mut f = ctx.new_frame();
    f.planes[0].copy_from_raw_u8(&sy, w, 1);
    f.planes[1].copy_from_raw_u8(&su, w / 2, 1);
    f.planes[2].copy_from_raw_u8(&sv, w / 2, 1);
    ctx.send_frame(f).unwrap();
    inputs.push(sy);
  }
  ctx.flush();

  // Pre-fix this loop panicked inside receive_packet() at
  // segmentation.rs (min_segment == MAX_SEGMENTS) on the first INTER frame.
  let mut packets = Vec::new();
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => packets.push(pkt.data),
      Err(EncoderStatus::LimitReached) => break,
      Err(EncoderStatus::Encoded) => {}
      Err(e) => panic!("encode error: {e:?}"),
    }
  }
  assert_eq!(packets.len(), frames, "expected one packet per input frame");

  // Independent-decoder conformance: every frame must decode, and the
  // decoded luma must resemble the input (a segmentation header desync
  // shows up as garbage pixels or a decoder error, not a subtle drift).
  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut raw_frames = Vec::new();
  for pkt in &packets {
    if let Some(frame) = dec.decode(pkt).expect("decode error (desync?)") {
      raw_frames.push(frame);
    }
  }
  raw_frames.extend(dec.flush().expect("flush error"));
  assert_eq!(raw_frames.len(), frames, "every coded frame must decode");

  let mut decoded: Vec<Vec<u8>> = Vec::new();
  for frame in &raw_frames {
    assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
    let mut dy = vec![0u8; w * h];
    match frame.planes() {
      rav1d_safe::Planes::Depth8(p) => {
        for (j, row) in p.y().rows().enumerate().take(h) {
          dy[j * w..(j + 1) * w].copy_from_slice(&row[..w]);
        }
      }
      _ => panic!("expected 8-bit planes"),
    }
    decoded.push(dy);
  }

  for (t, (input, output)) in inputs.iter().zip(&decoded).enumerate() {
    let p = psnr(input, output);
    assert!(
      p > 25.0,
      "frame {t}: luma PSNR {p:.2} dB too low — segmentation re-signal \
       likely desynced the bitstream"
    );
  }
}
