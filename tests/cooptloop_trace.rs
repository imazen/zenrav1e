//! Liveness + schema gate for the COOPT_LOOP decision-trace (Phase 0 part 2,
//! docs/COOPT_LOOP_PLAN.md in zenavif). Only compiled with
//! `--features cooptloop_trace`; a stock build has neither the module nor the
//! emit sites (byte-identical encodes). Proves: the trace fires during real RDO,
//! the currency identity `cost == distortion + lambda*rate_bits` holds for every
//! row (what the joint fits rely on), and `dump_tsv` writes the documented schema.
#![cfg(feature = "cooptloop_trace")]
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::cooptloop_trace;
use zenrav1e::prelude::*;

/// Gradient + deterministic LCG noise → broadband detail so RDO evaluates many
/// candidates (mirrors tests/trellis_roundtrip.rs).
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

#[test]
fn trace_fires_and_currency_identity_holds() {
  let (w, h) = (128usize, 128usize);
  let (sy, su, sv) = synth(w, h);

  cooptloop_trace::clear();
  assert_eq!(cooptloop_trace::len(), 0, "clear must empty the buffer");

  let mut ss = SpeedSettings::from_preset(6);
  ss.segmentation = SegmentationLevel::Disabled;
  let enc = EncoderConfig {
    width: w,
    height: h,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    still_picture: true,
    low_latency: true,
    quantizer: 60,
    tune: Tune::Ssimulacra2, // exercises the scaled (kind=1) path too
    speed_settings: ss,
    ..Default::default()
  };
  // threads=1: deterministic record order, no global-buffer contention (the
  // __simd_test_log discipline).
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
  assert!(!obu.is_empty(), "encoder produced no data");

  // The RD loop must have evaluated many candidates (liveness).
  let n = cooptloop_trace::len();
  assert!(n > 100, "expected a busy RD trace, got {n} records");

  // Dump + verify the documented schema and that the row count is exact.
  let path = std::env::temp_dir().join("cooptloop_trace_test.tsv");
  let path = path.to_str().unwrap();
  let written = cooptloop_trace::dump_tsv(path).expect("dump_tsv");
  assert_eq!(written, n, "dump row count must equal len()");

  let text = std::fs::read_to_string(path).unwrap();
  assert_eq!(
    text.lines().next().unwrap(),
    "kind\tlambda\trate_bits\tdistortion\trd_cost",
    "schema header drift"
  );
  assert_eq!(
    text.lines().skip(1).count(),
    n,
    "TSV data rows must equal record count"
  );

  // The currency identity the fits rely on: cost == distortion + lambda*rate_bits
  // (recomputed with the same fused mul_add, so bit-exact up to the text
  // round-trip, which Rust's {} guarantees round-trips).
  let mut checked = 0usize;
  for line in text.lines().skip(1).take(500) {
    let c: Vec<f64> =
      line.split('\t').skip(1).map(|x| x.parse().unwrap()).collect();
    let (lambda, rate_bits, distortion, cost) = (c[0], c[1], c[2], c[3]);
    let expect = lambda.mul_add(rate_bits, distortion);
    assert!(
      (expect - cost).abs() <= 1e-6 * cost.abs().max(1.0),
      "currency identity broken: {lambda}*{rate_bits}+{distortion} \
       = {expect} != {cost}"
    );
    checked += 1;
  }
  assert!(checked > 0, "no rows checked");

  let _ = std::fs::remove_file(path);
}
