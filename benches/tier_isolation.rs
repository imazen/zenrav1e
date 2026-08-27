//! SIMD-tier isolation: NEON assembly vs the pure-Rust fallback.
//!
//! zenrav1e's ARM SIMD is hand-written NEON **assembly** under `src/arm/`,
//! selected by `cfg(asm_neon)` which build.rs emits when the `asm` cargo
//! feature is on. There is no runtime token to toggle, so the comparison is a
//! COMPILE-TIME A/B: run this bench twice, once with `--features asm` and once
//! without, and compare.
//!
//! ```text
//! cargo bench --bench tier_isolation --features asm   # NEON asm
//! cargo bench --bench tier_isolation                  # pure Rust
//! ```
//!
//! This bench exists because `benches/bench.rs` does not compile against the
//! current API (it is behind `required-features = ["bench"]`, so no CI job
//! builds it, and it has drifted: `ts.qc.update` takes 7 args where it passes
//! 6, and a 25-arg function is called with 22). Until that is repaired there is
//! no working measurement of what the NEON assembly is worth on ARM.
//!
//! Encoding is deliberately single-threaded and still-picture so the number
//! reflects codec work rather than thread scheduling.

use criterion::{Criterion, criterion_group, criterion_main};
use zenrav1e::prelude::*;

/// Noise + patches. A gradient would produce degenerate residuals and
/// understate exactly the transform/prediction kernels this is measuring.
fn synth(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; w * h];
  let mut u = vec![0u8; (w / 2) * (h / 2)];
  let mut v = vec![0u8; (w / 2) * (h / 2)];
  let mut s = 0x9e37_79b9u32;
  let mut next = move || {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (s >> 24) as u8
  };
  for j in 0..h {
    for i in 0..w {
      let patch = ((i / 32 + j / 32) & 3) as u8;
      y[j * w + i] = next().wrapping_add(patch.wrapping_mul(40));
    }
  }
  for j in 0..h / 2 {
    for i in 0..w / 2 {
      let patch = ((i / 16 + j / 16) & 3) as u8;
      u[j * (w / 2) + i] = next().wrapping_add(patch.wrapping_mul(60));
      v[j * (w / 2) + i] = next().wrapping_add(patch.wrapping_mul(90));
    }
  }
  (y, u, v)
}

fn encode_once(
  w: usize, h: usize, planes: &(Vec<u8>, Vec<u8>, Vec<u8>), speed: u8,
) -> usize {
  let enc = EncoderConfig {
    width: w,
    height: h,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    still_picture: true,
    low_latency: true,
    quantizer: 100,
    speed_settings: SpeedSettings::from_preset(speed),
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  f.planes[0].copy_from_raw_u8(&planes.0, w, 1);
  f.planes[1].copy_from_raw_u8(&planes.1, w / 2, 1);
  f.planes[2].copy_from_raw_u8(&planes.2, w / 2, 1);
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut n = 0;
  while let Ok(pkt) = ctx.receive_packet() {
    n += pkt.data.len();
  }
  n
}

fn bench_encode(c: &mut Criterion) {
  // Label the arm by what was actually compiled in, so the two runs are not
  // confusable after the fact.
  let arm = if cfg!(asm_neon) {
    "neon_asm"
  } else if cfg!(nasm_x86_64) {
    "x86_asm"
  } else {
    "rust_fallback"
  };
  eprintln!("[tier_isolation] built with: {arm}");

  for &(label, w, h) in
    &[("256x256", 256usize, 256usize), ("512x512", 512, 512)]
  {
    let planes = synth(w, h);
    let mut group = c.benchmark_group(format!("encode_still/{label}"));
    group.sample_size(10);
    group.bench_function(arm, |b| {
      b.iter(|| encode_once(w, h, std::hint::black_box(&planes), 8))
    });
    group.finish();
  }
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
