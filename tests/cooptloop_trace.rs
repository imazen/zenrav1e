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

  // The RD loop must have evaluated many candidates (liveness), with no
  // capped-out drops (a dropped-rows trace is INCOMPLETE by contract).
  let n = cooptloop_trace::len();
  assert!(n > 100, "expected a busy RD trace, got {n} records");
  assert_eq!(cooptloop_trace::dropped(), 0, "unexpected capped drops");

  // Dump + verify the documented schema and that the row count is exact.
  let path = std::env::temp_dir().join("cooptloop_trace_test.tsv");
  let path = path.to_str().unwrap();
  let written = cooptloop_trace::dump_tsv(path).expect("dump_tsv");
  assert_eq!(written, n, "dump row count must equal len()");

  let text = std::fs::read_to_string(path).unwrap();
  assert_eq!(
    text.lines().next().unwrap(),
    "block_seq\tbo_x\tbo_y\tbsize\trow\tlambda\trate_bits\tdistortion\t\
     rd_cost\tmode\ttx_size\tskip",
    "schema header drift"
  );
  assert_eq!(
    text.lines().skip(1).count(),
    n,
    "TSV data rows must equal record count"
  );

  // Parse rows: [block_seq, bo_x, bo_y, bsize, row] ints + 4 floats + 3 ints.
  let mut evals = 0usize;
  let mut decisions = 0usize;
  let mut eval_seqs = std::collections::HashSet::new();
  let mut decision_seqs = Vec::new();
  for line in text.lines().skip(1) {
    let c: Vec<&str> = line.split('\t').collect();
    assert_eq!(c.len(), 12, "row width");
    let block_seq: u64 = c[0].parse().unwrap();
    let row: u8 = c[4].parse().unwrap();
    match row {
      0 | 1 => {
        // Currency identity the fits rely on:
        // cost == distortion + lambda*rate_bits (same fused mul_add; text
        // round-trip is exact for Rust's {} float formatting).
        let (lambda, rate_bits): (f64, f64) =
          (c[5].parse().unwrap(), c[6].parse().unwrap());
        let (distortion, cost): (f64, f64) =
          (c[7].parse().unwrap(), c[8].parse().unwrap());
        let expect = lambda.mul_add(rate_bits, distortion);
        assert!(
          (expect - cost).abs() <= 1e-6 * cost.abs().max(1.0),
          "currency identity broken: {lambda}*{rate_bits}+{distortion} \
           = {expect} != {cost}"
        );
        evals += 1;
        eval_seqs.insert(block_seq);
      }
      2 => {
        // Decision rows carry the chosen (mode, tx_size, skip) + rd_cost.
        let cost: f64 = c[8].parse().unwrap();
        assert!(cost.is_finite() && cost >= 0.0, "bad decision rd_cost");
        assert_ne!(c[9], "255", "decision row must carry a real mode");
        assert!(block_seq > 0, "decision row outside any scope");
        decisions += 1;
        decision_seqs.push(block_seq);
      }
      other => panic!("unknown row kind {other}"),
    }
  }
  assert!(evals > 100, "expected many currency rows, got {evals}");
  assert!(decisions > 0, "expected block decision rows, got none");
  // Scope integrity: every decision's search emitted currency rows under the
  // same block_seq (the chosen-vs-evaluated join the analyzer relies on).
  let joined = decision_seqs.iter().filter(|s| eval_seqs.contains(s)).count();
  assert!(
    joined * 10 >= decision_seqs.len() * 9,
    "fewer than 90% of decision scopes have currency rows \
     ({joined}/{})",
    decision_seqs.len()
  );

  let _ = std::fs::remove_file(path);
}
