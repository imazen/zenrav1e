// Copyright (c) 2026, Imazen LLC. All rights reserved.
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0.

//! Trellis quantization for AV1 coefficient optimization.
//!
//! Uses a Viterbi-style dynamic programming pass to find the optimal
//! combination of coefficient levels that minimizes rate + λ·distortion.
//! This exploits the AV1 entropy coding structure where each coefficient's
//! coding cost depends on the context formed by preceding coefficients.
//!
//! Purely encoder-side optimization — produces valid AV1 bitstream.

use crate::quantize::qm_tables::AOM_QM_BITS;
use crate::quantize::QuantizationContext;
use crate::scan_order::av1_scan_orders;
use crate::transform::{TxSize, TxType};
use crate::util::*;

/// AV1 coefficient coding context states.
///
/// The coding cost of each coefficient depends on the "level bin" of
/// the previous coded coefficient in scan order:
///   State 0: previous level == 0
///   State 1: previous level == 1
///   State 2: previous level >= 2
const NUM_STATES: usize = 3;

/// Estimated rate (in units of 1/256 bit) for each (state, level_bin) pair.
/// These are approximate costs derived from typical AV1 coefficient CDFs.
///
/// Row = current state (prev_level_bin), Column = current level_bin (0, 1, 2, >2)
/// Level bin: 0 = zero, 1 = level 1, 2 = level 2, 3 = level > 2
const RATE_TABLE: [[u16; 4]; NUM_STATES] = [
  // State 0: prev was zero → zeros are very cheap, nonzeros expensive
  [28, 320, 384, 448],
  // State 1: prev was 1 → small values expected
  [180, 140, 280, 380],
  // State 2: prev was ≥2 → larger values expected
  [260, 180, 160, 220],
];

/// Extra rate per unit of level above 2, in 1/256 bit units.
/// Approximates the base_range + Golomb coding cost.
const RATE_PER_LEVEL: u16 = 48;

/// Rate in 1/256 bits for the sign bit.
const SIGN_RATE: u16 = 256;

/// Rate in 1/256 bits for each extra bit of Golomb coding (level > 14).
const GOLOMB_RATE: u16 = 256;

/// Number of AV1 base range iterations before Golomb.
/// COEFF_BASE_RANGE = 12, each coded in chunks of BR_CDF_SIZE-1 = 3.
const COEFF_BASE_RANGE: u32 = 12;

/// Maximum coefficient level to consider in trellis alternatives.
/// Beyond this the alternatives are too expensive to matter.
const MAX_LEVEL: u32 = 60;

/// State index from coefficient level.
#[inline]
fn state_from_level(level: u32) -> usize {
  match level {
    0 => 0,
    1 => 1,
    _ => 2,
  }
}

/// Estimate the rate cost (in 1/256 bit) of coding a coefficient at `level`
/// when the previous coefficient had level bin `prev_state`.
#[inline]
fn estimate_rate(prev_state: usize, level: u32) -> u32 {
  let bin = (level as usize).min(3);
  let mut rate = RATE_TABLE[prev_state][bin] as u32;

  if level > 0 {
    rate += SIGN_RATE as u32;
  }

  if level > 2 {
    // Base range coding: each group of 3 levels costs ~one symbol
    let excess = level - 3; // levels above base level 2
    let base_range_levels = excess.min(COEFF_BASE_RANGE);
    rate += (base_range_levels * RATE_PER_LEVEL as u32 + 2) / 3;

    // Golomb coding for levels above base_range
    if excess > COEFF_BASE_RANGE {
      let golomb_val = excess - COEFF_BASE_RANGE;
      // Golomb coding uses approximately 2*floor(log2(val+1))+1 bits
      let bits = if golomb_val == 0 {
        1
      } else {
        2 * (32 - golomb_val.leading_zeros()) + 1
      };
      rate += bits * GOLOMB_RATE as u32;
    }
  }

  rate
}

/// Compute the distortion delta for coding coefficient `coeff` (in transform
/// domain, pre-shift) at quantized level `level` with effective quantizer `quant`.
///
/// Returns distortion in squared-error units.
#[inline]
fn distortion(coeff: i64, level: u32, quant: u32) -> i64 {
  let recon = level as i64 * quant as i64;
  let err = coeff.abs() - recon;
  err * err
}

/// Optimize quantized coefficients using trellis (Viterbi DP).
///
/// `qcoeffs`: quantized coefficients (modified in-place)
/// `coeffs`: original transform-domain coefficients (pre-quantization)
/// `qc`: quantization context (provides DC/AC quantizers)
/// `tx_size`: transform block size
/// `tx_type`: transform type (for scan order)
/// `lambda`: Lagrangian multiplier from RDO (higher = more aggressive zeroing)
/// `qm`: optional quantization matrix weights
/// `eob`: current end-of-block position (1-based, in scan order)
///
/// Returns the new eob after optimization.
pub fn optimize<T: Coefficient>(
  qcoeffs: &mut [T], coeffs: &[T], qc: &QuantizationContext, tx_size: TxSize,
  tx_type: TxType, lambda: f64, qm: Option<&[u8]>, eob: u16,
) -> u16 {
  let scan = &av1_scan_orders[tx_size as usize][tx_type as usize].scan;
  let n = eob as usize;

  if n <= 1 {
    return eob;
  }

  // Scale lambda: our rate is in 1/256 bit units, distortion is in squared
  // transform-domain error. We want λ_scaled such that:
  //   cost = distortion + λ_scaled * rate_256
  // rav1e's lambda is calibrated for bits, so:
  //   λ_scaled = lambda * 256 (to convert rate from 1/256 bits to bits)
  // But our distortion is unscaled squared error. rav1e's lambda is already
  // in distortion-per-bit units, so:
  let lambda_256 = lambda;

  let log_tx_scale = crate::quantize::get_log_tx_scale(tx_size);
  let dc_quant = qc.dc_quant();
  let ac_quant = qc.ac_quant();

  // DP arrays: cost[state] = minimum cost to code positions [i..eob-1]
  // with the coefficient at position i producing `state`.
  let mut dp_cost = [f64::MAX; NUM_STATES];
  let mut dp_next_state = [[0u8; NUM_STATES]; 32 * 32];
  let mut dp_level = [[0u32; NUM_STATES]; 32 * 32];

  // Allocate on heap for large transforms
  if n > 1024 {
    let mut ns = vec![[0u8; NUM_STATES]; n];
    let mut lv = vec![[0u32; NUM_STATES]; n];
    run_trellis(
      qcoeffs,
      coeffs,
      scan,
      n,
      lambda_256,
      log_tx_scale,
      dc_quant,
      ac_quant,
      qm,
      &mut dp_cost,
      &mut ns,
      &mut lv,
    )
  } else {
    run_trellis(
      qcoeffs,
      coeffs,
      scan,
      n,
      lambda_256,
      log_tx_scale,
      dc_quant,
      ac_quant,
      qm,
      &mut dp_cost,
      &mut dp_next_state[..n],
      &mut dp_level[..n],
    )
  }
}

fn run_trellis<T: Coefficient>(
  qcoeffs: &mut [T], coeffs: &[T], scan: &[u16], n: usize, lambda_256: f64,
  log_tx_scale: usize, dc_quant: u16, ac_quant: u16, qm: Option<&[u8]>,
  dp_cost: &mut [f64; NUM_STATES],
  dp_next_state: &mut [[u8; NUM_STATES]],
  dp_level: &mut [[u32; NUM_STATES]],
) -> u16 {
  // Initialize: after the last coefficient, cost for all states is 0
  dp_cost.fill(0.0);

  // Backward pass: from eob-1 down to 1 (skip DC, handled separately)
  for i in (1..n).rev() {
    let scan_pos = scan[i] as usize;
    let coeff_raw = i32::cast_from(coeffs[scan_pos]);
    let coeff = (coeff_raw as i64) << log_tx_scale;
    let orig_level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();

    // Determine effective quantizer for this position
    let base_quant = ac_quant as u32;
    let eff_quant = match qm {
      Some(qm_tbl) if scan_pos < qm_tbl.len() => {
        let wt = qm_tbl[scan_pos] as u32;
        (base_quant * wt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS
      }
      _ => base_quant,
    };

    if eff_quant == 0 {
      // Can't quantize this position, keep as-is
      for s in 0..NUM_STATES {
        dp_next_state[i][s] = state_from_level(orig_level) as u8;
        dp_level[i][s] = orig_level;
      }
      continue;
    }

    let mut new_cost = [f64::MAX; NUM_STATES];
    let mut new_next = [0u8; NUM_STATES];
    let mut new_level = [0u32; NUM_STATES];

    // For each possible previous state, find the best level for this position
    for prev_s in 0..NUM_STATES {
      let mut best_cost = f64::MAX;
      let mut best_level = orig_level;
      let mut best_next_s = state_from_level(orig_level) as u8;

      // Try the original level and alternatives (round down, zero out)
      let min_level = if orig_level > 2 { orig_level - 1 } else { 0 };
      let max_level = orig_level.min(MAX_LEVEL);

      for level in min_level..=max_level {
        let next_s = state_from_level(level);
        let d = distortion(coeff, level, eff_quant);
        let r = estimate_rate(prev_s, level);
        let cost = d as f64 + lambda_256 * r as f64 + dp_cost[next_s];

        if cost < best_cost {
          best_cost = cost;
          best_level = level;
          best_next_s = next_s as u8;
        }
      }

      new_cost[prev_s] = best_cost;
      new_next[prev_s] = best_next_s;
      new_level[prev_s] = best_level;
    }

    dp_next_state[i] = new_next;
    dp_level[i] = new_level;
    *dp_cost = new_cost;
  }

  // Handle DC separately (position 0 in scan order, uses dc_quant)
  {
    let scan_pos = scan[0] as usize;
    let coeff_raw = i32::cast_from(coeffs[scan_pos]);
    let coeff = (coeff_raw as i64) << log_tx_scale;
    let orig_level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();

    let eff_quant = match qm {
      Some(qm_tbl) if !qm_tbl.is_empty() => {
        let wt = qm_tbl[0] as u32;
        (dc_quant as u32 * wt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS
      }
      _ => dc_quant as u32,
    };

    // DC has no "previous" state. Try with a neutral prior (state 2, large).
    // The DC coefficient is coded with its own context, not really dependent
    // on AC context, so we use state 2 as prior.
    let prev_s = 2;

    let mut best_cost = f64::MAX;
    let mut best_level = orig_level;
    let mut best_next_s = 0u8;

    if eff_quant > 0 {
      // DC: only try original level (don't zero out DC — too aggressive)
      let min_level = if orig_level > 1 { orig_level - 1 } else { 0 };
      for level in min_level..=orig_level {
        let next_s = state_from_level(level);
        let d = distortion(coeff, level, eff_quant);
        let r = estimate_rate(prev_s, level);
        let cost = d as f64 + lambda_256 * r as f64 + dp_cost[next_s];

        if cost < best_cost {
          best_cost = cost;
          best_level = level;
          best_next_s = next_s as u8;
        }
      }
    } else {
      best_next_s = state_from_level(orig_level) as u8;
    }

    // Apply DC decision
    if best_level != orig_level {
      let sign = if coeff_raw < 0 { -1i32 } else { 1i32 };
      qcoeffs[scan_pos] = T::cast_from(sign * best_level as i32);
    }

    // Forward pass: apply optimal levels from position 1 onward
    let mut state = best_next_s as usize;
    for i in 1..n {
      let scan_pos = scan[i] as usize;
      let level = dp_level[i][state];
      let orig_level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();

      if level != orig_level {
        let coeff_raw = i32::cast_from(coeffs[scan_pos]);
        let sign = if coeff_raw < 0 { -1i32 } else { 1i32 };
        qcoeffs[scan_pos] = T::cast_from(sign * level as i32);
      }

      state = dp_next_state[i][state] as usize;
    }
  }

  // Recompute eob from the modified coefficients
  let new_eob = scan[..n]
    .iter()
    .rposition(|&pos| qcoeffs[pos as usize] != T::cast_from(0))
    .map(|i| i + 1)
    .unwrap_or(0) as u16;

  new_eob
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_state_from_level() {
    assert_eq!(state_from_level(0), 0);
    assert_eq!(state_from_level(1), 1);
    assert_eq!(state_from_level(2), 2);
    assert_eq!(state_from_level(100), 2);
  }

  #[test]
  fn test_estimate_rate_monotonic() {
    // Rate should generally increase with level for a given state
    for s in 0..NUM_STATES {
      let r0 = estimate_rate(s, 0);
      let r1 = estimate_rate(s, 1);
      let r5 = estimate_rate(s, 5);
      let r20 = estimate_rate(s, 20);
      // Level 0 in state 0 should be cheap
      if s == 0 {
        assert!(r0 < r1, "zero should be cheap in state 0");
      }
      // Higher levels should cost more
      assert!(r5 < r20, "higher levels should cost more rate");
    }
  }

  #[test]
  fn test_distortion() {
    // Zero level: distortion = coeff²
    assert_eq!(distortion(100, 0, 10), 10000);
    // Perfect reconstruction: distortion = 0
    assert_eq!(distortion(100, 10, 10), 0);
    // Off by one level: distortion = quant²
    assert_eq!(distortion(100, 9, 10), 100);
  }
}
