// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Regression test for issue #10:
//! <https://github.com/imazen/zenrav1e/issues/10>
//!
//! 8-bit content stored in `u16` (a `Context<u16>` with `bit_depth == 8`) must
//! not be routed to the high-bitdepth SIMD kernels. dav1d's 10/12bpc inverse
//! transform, 16bpc intra predictor, and 16bpc subpel (`put_8tap`) kernels are
//! only bit-accurate at their native bit depths (>= 10). Fed 8-bit content they
//! emit out-of-range reconstructed samples (e.g. 256 from the inverse
//! transform, 512 from the DC predictor), which then trip the
//! `p >> coeff_shift <= 255` debug assertion in CDEF direction search
//! (`cdef_find_dir`).
//!
//! This encodes a few frames of high-frequency 8-bit-in-`u16` content with
//! CDEF active. With debug assertions on (the default for `cargo test`) and the
//! SIMD `asm` feature enabled, a regression panics in `cdef_find_dir` during
//! encoding. On a scalar (no-`asm`) build the Rust kernels are already correct,
//! so the test simply confirms the encode succeeds.

use crate::prelude::*;

/// Deterministic per-pixel white noise spanning the full 0..=255 range. White
/// noise maximises residual energy, which is what pushed the buggy
/// high-bitdepth inverse transform out of range.
fn noise(x: usize, y: usize, f: u64) -> u16 {
  let mut v = (x as u64).wrapping_mul(2_654_435_761)
    ^ (y as u64).wrapping_mul(40_503)
    ^ f.wrapping_mul(2_246_822_519);
  v ^= v >> 13;
  v ^= v << 7;
  v ^= v >> 17;
  (v & 0xff) as u16
}

#[test]
fn encode_8bit_in_u16_does_not_trip_cdef_range_assert() {
  // speed 1 keeps CDEF + the loop-filter RDO decision on (where the assertion
  // fires); a 66x66 frame is non-8-aligned, exercising the frame-edge blocks
  // where the DC predictor left 512; three frames exercise intra and inter
  // prediction (so all of itx, intra-predict and put_8tap are hit).
  let mut enc = EncoderConfig::with_speed_preset(1);
  enc.width = 66;
  enc.height = 66;
  enc.bit_depth = 8;
  enc.chroma_sampling = ChromaSampling::Cs420;
  enc.quantizer = 100;
  enc.min_key_frame_interval = 1;
  enc.max_key_frame_interval = 5;
  enc.low_latency = true;

  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u16> = cfg.new_context().unwrap();

  for f in 0..3u64 {
    let mut frame = ctx.new_frame();
    for plane in &mut frame.planes {
      let stride = plane.cfg.stride;
      for (y, row) in plane.data.chunks_mut(stride).enumerate() {
        for (x, px) in row.iter_mut().enumerate() {
          *px = noise(x, y, f);
        }
      }
    }
    ctx.send_frame(frame).unwrap();
  }
  ctx.flush();

  let mut packets = 0;
  loop {
    match ctx.receive_packet() {
      Ok(_) => packets += 1,
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!("unexpected encoder status: {e:?}"),
    }
  }

  // Confirm the encode actually ran (and so actually exercised the kernels)
  // rather than silently producing nothing.
  assert_eq!(packets, 3, "expected one packet per input frame");
}
