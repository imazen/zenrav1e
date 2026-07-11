//! COOPT_LOOP decision-trace — Phase 0 part 2 (docs/COOPT_LOOP_PLAN.md in zenavif).
//!
//! Two layers, one stream:
//!
//! - **Currency rows** (`row` 0/1): one per [`crate::rdo::compute_rd_cost`]
//!   (plain / ssim-rdmult-scaled) evaluation — the (lambda, rate, distortion,
//!   cost) every RD decision weighs.
//! - **Decision rows** (`row` 2): one per [`crate::rdo::rdo_mode_decision`]
//!   return — the block's chosen mode/tx/skip and final `rd_cost`.
//!
//! A thread-local **block scope** stamps every row with `(block_seq, bo, bsize)`:
//! `begin_block` opens the scope, the currency rows emitted during the search
//! inherit it, and `end_block` writes the decision row and closes it. An offline
//! analyzer can therefore reconstruct chosen-vs-evaluated per block without any
//! per-candidate plumbing inside the search internals. Rows outside any scope
//! (frame-level trials, CDEF/LRF search, etc.) carry `block_seq = 0`, `bsize =
//! 255`.
//!
//! The module and its emit sites exist only under the `cooptloop_trace` feature
//! — a stock build is byte-identical by construction. Intended for
//! single-threaded analysis encodes (`threads = 1`, the `__simd_test_log`
//! census discipline): deterministic record order, no lock contention.
//!
//! Memory: records accumulate in RAM (~64 B each; a busy 1 MP encode can emit
//! tens of millions). `COOPTLOOP_TRACE_CAP` (env, rows) bounds the buffer;
//! excess rows are counted in [`dropped`] — never silently truncated. Call
//! [`clear`] between encodes so one trace maps to one encode.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// One trace row. `row`: 0 = currency eval (plain), 1 = currency eval
/// (per-block scaled lambda), 2 = block decision. `mode`/`tx_size`/`skip` are
/// meaningful on decision rows only (255/255/255 on evals); `rate_bits` /
/// `distortion` are NaN on decision rows (the winner's composition lives in
/// its scope's eval rows).
#[derive(Clone, Copy, Debug)]
pub struct Record {
  pub block_seq: u64,
  pub bo_x: u16,
  pub bo_y: u16,
  pub bsize: u8,
  pub row: u8,
  pub lambda: f64,
  pub rate_bits: f64,
  pub distortion: f64,
  pub cost: f64,
  pub mode: u8,
  pub tx_size: u8,
  pub skip: u8,
}

static BUF: Mutex<Vec<Record>> = Mutex::new(Vec::new());
static DROPPED: AtomicU64 = AtomicU64::new(0);
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

thread_local! {
  /// (block_seq, bo_x, bo_y, bsize) of the open decision scope; seq 0 = none.
  static CTX: Cell<(u64, u16, u16, u8)> = const { Cell::new((0, 0, 0, 255)) };
}

fn cap() -> usize {
  use std::sync::OnceLock;
  static CAP: OnceLock<usize> = OnceLock::new();
  *CAP.get_or_init(|| {
    std::env::var("COOPTLOOP_TRACE_CAP")
      .ok()
      .and_then(|v| v.parse().ok())
      .unwrap_or(usize::MAX)
  })
}

fn push(r: Record) {
  if let Ok(mut b) = BUF.lock() {
    if b.len() < cap() {
      b.push(r);
    } else {
      DROPPED.fetch_add(1, Ordering::Relaxed);
    }
  }
}

/// Open a decision scope: subsequent currency rows on this thread are stamped
/// with this block's identity until [`end_block`].
pub fn begin_block(bo_x: u16, bo_y: u16, bsize: u8) {
  let seq = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
  CTX.with(|c| c.set((seq, bo_x, bo_y, bsize)));
}

/// Write the decision row for the open scope and close it. No-op stamp-wise if
/// no scope is open (the decision row then carries seq 0 — analyzer-visible).
pub fn end_block(
  mode: u8, tx_size: u8, skip: bool, rd_cost: f64, lambda: f64,
) {
  let (seq, bx, by, bs) = CTX.with(Cell::get);
  push(Record {
    block_seq: seq,
    bo_x: bx,
    bo_y: by,
    bsize: bs,
    row: 2,
    lambda,
    rate_bits: f64::NAN,
    distortion: f64::NAN,
    cost: rd_cost,
    mode,
    tx_size,
    skip: skip as u8,
  });
  CTX.with(|c| c.set((0, 0, 0, 255)));
}

/// Append one currency evaluation (called from `compute_rd_cost[_scaled]`).
/// `kind`: 0 = plain, 1 = scaled.
#[inline]
pub fn record(
  lambda: f64, rate_bits: f64, distortion: f64, cost: f64, kind: u8,
) {
  let (seq, bx, by, bs) = CTX.with(Cell::get);
  push(Record {
    block_seq: seq,
    bo_x: bx,
    bo_y: by,
    bsize: bs,
    row: kind,
    lambda,
    rate_bits,
    distortion,
    cost,
    mode: 255,
    tx_size: 255,
    skip: 255,
  });
}

/// Number of records buffered so far.
pub fn len() -> usize {
  BUF.lock().map(|b| b.len()).unwrap_or(0)
}

/// Rows dropped because the `COOPTLOOP_TRACE_CAP` bound was hit. Non-zero
/// means the trace is INCOMPLETE — an analyzer must check this before fitting.
pub fn dropped() -> u64 {
  DROPPED.load(Ordering::Relaxed)
}

/// Drop all buffered records and reset the drop counter (call between encodes
/// so one trace maps to one encode). Block seq numbering continues.
pub fn clear() {
  if let Ok(mut b) = BUF.lock() {
    b.clear();
  }
  DROPPED.store(0, Ordering::Relaxed);
}

/// Write the buffered trace as a TSV (header + one row per record) and return
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
  writeln!(
    f,
    "block_seq\tbo_x\tbo_y\tbsize\trow\tlambda\trate_bits\tdistortion\trd_cost\tmode\ttx_size\tskip"
  )?;
  for r in b.iter() {
    writeln!(
      f,
      "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
      r.block_seq,
      r.bo_x,
      r.bo_y,
      r.bsize,
      r.row,
      r.lambda,
      r.rate_bits,
      r.distortion,
      r.cost,
      r.mode,
      r.tx_size,
      r.skip
    )?;
  }
  Ok(b.len())
}
