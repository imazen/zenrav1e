//! COOPT_LOOP decision-trace — Phase 0 part 2 (docs/COOPT_LOOP_PLAN.md in zenavif).
//!
//! A per-RD-evaluation log of the (lambda, rate, distortion, cost) currency that
//! [`crate::rdo::compute_rd_cost`] combines. This is the dataset generator for the
//! joint lambda-D-R fits of Phases 1-3: with a trace, "does this candidate D
//! predict the metric delta" and "is the fast-tier rate estimator consistent with
//! the exact in-trial tells" become OFFLINE regressions over cached encodes rather
//! than fresh sweeps — which is what makes the joint fit affordable.
//!
//! The whole module and its emit sites are behind the `cooptloop_trace` feature,
//! so a stock build is byte-identical (the emits do not exist — they are `#[cfg]`
//! -gated). Intended for single-threaded analysis encodes (`threads = 1`), matching
//! the `__simd_test_log` census discipline: the global buffer then has no lock
//! contention and the record order is deterministic.

use std::sync::Mutex;

/// One RD-cost evaluation: the currency `compute_rd_cost` weighed for one
/// candidate (a mode, a partition leaf, a transform choice — the caller varies).
#[derive(Clone, Copy, Debug)]
pub struct Record {
  /// The effective lambda applied — already multiplied by the per-block scale
  /// for the scaled call site — so `cost == distortion + lambda * rate_bits`.
  pub lambda: f64,
  /// Rate in bits: the encoder's integer rate divided by `2^OD_BITRES`.
  pub rate_bits: f64,
  /// Scaled distortion — the D the loop actually minimizes.
  pub distortion: f64,
  /// The resulting RD cost.
  pub cost: f64,
  /// 0 = `compute_rd_cost`, 1 = `compute_rd_cost_scaled` (the per-16x16
  /// ssim-rdmult path). Lets a fit separate the two currency regimes.
  pub kind: u8,
}

static BUF: Mutex<Vec<Record>> = Mutex::new(Vec::new());

/// Append one evaluation to the trace. Called from `compute_rd_cost[_scaled]`.
#[inline]
pub fn record(
  lambda: f64, rate_bits: f64, distortion: f64, cost: f64, kind: u8,
) {
  if let Ok(mut b) = BUF.lock() {
    b.push(Record { lambda, rate_bits, distortion, cost, kind });
  }
}

/// Number of records buffered so far.
pub fn len() -> usize {
  BUF.lock().map(|b| b.len()).unwrap_or(0)
}

/// Drop all buffered records (call between frames / at an analysis-run start so
/// one trace maps to one encode).
pub fn clear() {
  if let Ok(mut b) = BUF.lock() {
    b.clear();
  }
}

/// Write the buffered trace as a TSV (header + one row per evaluation) and return
/// the row count. Does not clear the buffer.
///
/// # Errors
/// Returns the underlying [`std::io::Error`] if `path` cannot be created or a
/// write fails, or an `Other` error if the trace buffer's lock was poisoned.
pub fn dump_tsv(path: &str) -> std::io::Result<usize> {
  use std::io::Write;
  let b = BUF
    .lock()
    .map_err(|_| std::io::Error::other("cooptloop_trace buffer poisoned"))?;
  let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
  writeln!(f, "kind\tlambda\trate_bits\tdistortion\trd_cost")?;
  for r in b.iter() {
    writeln!(
      f,
      "{}\t{}\t{}\t{}\t{}",
      r.kind, r.lambda, r.rate_bits, r.distortion, r.cost
    )?;
  }
  Ok(b.len())
}
