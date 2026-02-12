// Copyright (c) 2026, Imazen LLC. All rights reserved.
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0.

//! Trellis quantization for AV1 coefficient optimization.
//!
//! Performs two optimizations on quantized coefficients:
//!
//! 1. **EOB shrinkage**: Zeros trailing low-level coefficients where the
//!    distortion cost is outweighed by rate savings from a shorter block.
//!
//! 2. **Level round-down**: Reduces interior coefficient levels by 1 where
//!    the rate-distortion trade-off is favorable.
//!
//! Uses rav1e's RDO lambda and an approximate AV1 rate model.
//! Purely encoder-side — produces valid AV1 bitstream.

use crate::quantize::qm_tables::AOM_QM_BITS;
use crate::quantize::QuantizationContext;
use crate::scan_order::av1_scan_orders;
use crate::transform::{TxSize, TxType};
use crate::util::*;

/// Optimize quantized coefficients using rate-distortion optimization.
///
/// `qcoeffs`: quantized coefficients (modified in-place)
/// `coeffs`: original transform-domain coefficients (pre-quantization)
/// `qc`: quantization context (provides DC/AC quantizers)
/// `tx_size`: transform block size
/// `tx_type`: transform type (for scan order)
/// `lambda`: Lagrangian multiplier from RDO
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

  let log_tx_scale = crate::quantize::get_log_tx_scale(tx_size);
  let ac_quant = qc.ac_quant() as u32;

  // Lambda calibration for trellis:
  //
  // rav1e's RDO: cost = ScaledDist + lambda * rate_bits
  //   where ScaledDist = raw_sse >> tx_dist_scale_bits * bias * dist_scale
  //   and tx_dist_scale_bits = 2*(3 - log_tx_scale)
  //
  // For AVIF (no temporal RDO), bias ≈ 1.0 and dist_scale ≈ 1.0.
  // So ScaledDist ≈ raw_sse >> (6 - 2*lts) per coefficient.
  //
  // Our trellis distortion: sq_err returns (shifted_err²) >> (2*lts),
  // which equals raw_err² (the lts shifts cancel). This is 2^(6-2*lts)
  // times larger than rav1e's ScaledDist per coefficient.
  //
  // To match rav1e's R-D trade-off:
  //   our_dist + lambda_trellis * rate = 0
  //   ⟺ rav1e_dist * scale + lambda_trellis * rate = 0
  //   ⟺ rav1e_dist + (lambda_trellis/scale) * rate = 0
  //   Setting lambda_trellis/scale = lambda gives:
  //   lambda_trellis = lambda * scale = lambda * 2^(6-2*lts)
  let tx_dist_scale = (1u64 << (6 - 2 * log_tx_scale)) as f64;
  let lambda_trellis = lambda * tx_dist_scale;

  // Phase 1: EOB optimization via backward scan.
  // At each position, compute the cost of setting the new EOB here
  // (zeroing everything from this position to the current EOB).
  // The cost includes distortion increase from zeroing minus rate savings.
  //
  // Key: when we shrink EOB, we save not just the coefficient coding cost
  // but also the EOB signaling overhead (which uses a variable-length code).
  //
  // AV1 EOB coding: hierarchical prefix code where cost ≈ 2*log2(eob) bits.
  // So shrinking EOB from 47 to 30 saves roughly 2*(log2(47)-log2(30)) ≈ 1.3 bits.

  let eob_bits_current = eob_coding_cost(n);

  let mut best_new_eob = n;
  let mut best_net_cost = 0.0f64; // relative to keeping everything

  // Accumulated distortion from zeroing trailing coefficients
  let mut dist_delta_sum = 0.0f64;
  // Accumulated rate from coding the trailing coefficients (saved if zeroed)
  let mut rate_sum = 0.0f64;

  // Track context for rate estimation
  // Build the context state at each position in a forward pass first
  let mut states = vec![0u8; n + 1]; // states[i] = state BEFORE position i
  states[0] = 2; // assume large-level context before DC
  for i in 0..n {
    let scan_pos = scan[i] as usize;
    let level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();
    states[i + 1] = match level {
      0 => 0,
      1 => 1,
      _ => 2,
    };
  }

  for i in (1..n).rev() {
    let scan_pos = scan[i] as usize;
    let orig_level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();

    if orig_level == 0 {
      continue; // already zero, no savings from "zeroing" it
    }

    // Effective quantizer for this position
    let eff_quant = effective_ac_quant(ac_quant, scan_pos, qm);
    if eff_quant == 0 {
      continue;
    }

    // Distortion delta from zeroing
    let coeff_raw = i32::cast_from(coeffs[scan_pos]);
    let coeff = (coeff_raw as i64) << log_tx_scale;
    let dist_keep = sq_err(coeff, orig_level, eff_quant, log_tx_scale);
    let dist_zero = sq_err(coeff, 0, eff_quant, log_tx_scale);
    let dd = (dist_zero - dist_keep) as f64;

    // Rate saved from not coding this coefficient
    let prev_s = states[i] as usize;
    let coeff_rate = coeff_coding_cost(prev_s, orig_level);

    dist_delta_sum += dd;
    rate_sum += coeff_rate;

    // EOB coding cost if we moved EOB to position i
    let eob_bits_new = eob_coding_cost(i);
    let eob_rate_saved = eob_bits_current - eob_bits_new;

    // Net cost of zeroing positions [i..n):
    // positive = bad (distortion increase outweighs rate savings)
    // negative = good (rate savings outweigh distortion increase)
    let net = dist_delta_sum - lambda_trellis * (rate_sum + eob_rate_saved);

    if net < best_net_cost {
      best_net_cost = net;
      best_new_eob = i;
    }
  }

  // Apply EOB shrinkage
  if best_new_eob < n {
    for i in best_new_eob..n {
      let scan_pos = scan[i] as usize;
      qcoeffs[scan_pos] = T::cast_from(0);
    }
  }

  // Phase 2: Level round-down for interior coefficients [1..best_new_eob)
  // For each coefficient at level >= 2, check if reducing by 1 is cheaper.
  for i in 1..best_new_eob {
    let scan_pos = scan[i] as usize;
    let coeff_raw = i32::cast_from(coeffs[scan_pos]);
    let orig_level = i32::cast_from(qcoeffs[scan_pos]).unsigned_abs();

    if orig_level < 2 {
      continue; // don't round level-1 to 0 (too aggressive for interior)
    }

    let eff_quant = effective_ac_quant(ac_quant, scan_pos, qm);
    if eff_quant == 0 {
      continue;
    }

    let coeff = (coeff_raw as i64) << log_tx_scale;
    let new_level = orig_level - 1;

    let dist_orig = sq_err(coeff, orig_level, eff_quant, log_tx_scale);
    let dist_new = sq_err(coeff, new_level, eff_quant, log_tx_scale);
    let dd = (dist_new - dist_orig) as f64; // positive when near higher level

    let prev_s = states[i] as usize;
    let rate_orig = coeff_coding_cost(prev_s, orig_level);
    let rate_new = coeff_coding_cost(prev_s, new_level);
    let rate_saved = rate_orig - rate_new;

    if rate_saved > 0.0 && dd < lambda_trellis * rate_saved {
      let sign = if coeff_raw < 0 { -1i32 } else { 1 };
      qcoeffs[scan_pos] = T::cast_from(sign * new_level as i32);
      // Update state for next coefficient
      states[i + 1] = match new_level {
        0 => 0,
        1 => 1,
        _ => 2,
      };
    }
  }

  // Final EOB: find last non-zero
  scan[..n]
    .iter()
    .rposition(|&pos| qcoeffs[pos as usize] != T::cast_from(0))
    .map(|i| i + 1)
    .unwrap_or(0) as u16
}

/// Effective AC quantizer for a given scan position, accounting for QM.
#[inline]
fn effective_ac_quant(base_quant: u32, scan_pos: usize, qm: Option<&[u8]>) -> u32 {
  match qm {
    Some(qm_tbl) if scan_pos < qm_tbl.len() => {
      let wt = qm_tbl[scan_pos] as u32;
      (base_quant * wt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS
    }
    _ => base_quant,
  }
}

/// Squared error between original coefficient and reconstruction.
/// `coeff_shifted`: coeff << log_tx_scale, `level`: quantized abs level.
/// Returns distortion in unshifted squared-error units.
#[inline]
fn sq_err(coeff_shifted: i64, level: u32, quant: u32, log_tx_scale: usize) -> i64 {
  let recon = level as i64 * quant as i64;
  let err = coeff_shifted.abs() - recon;
  (err * err) >> (2 * log_tx_scale)
}

/// Approximate rate (in fractional bits) for coding a coefficient at `level`
/// given the previous coefficient's level bin (`prev_state`).
///
/// Based on typical AV1 coefficient coding costs:
/// - 0 → ~0.1 bits in zero context, ~0.7 bits in nonzero context
/// - 1 → ~1.25 bits in zero context, ~0.55 bits in small context
/// - >1 → base ~1.5 bits + ~0.19 bits per additional level
#[inline]
fn coeff_coding_cost(prev_state: usize, level: u32) -> f64 {
  // Base costs (in bits) for each state × level_bin combination
  const COSTS: [[f64; 4]; 3] = [
    // state 0 (prev=0): [0, 1, 2, >2]
    [0.11, 1.25, 1.50, 1.75],
    // state 1 (prev=1): [0, 1, 2, >2]
    [0.70, 0.55, 1.10, 1.50],
    // state 2 (prev>=2): [0, 1, 2, >2]
    [1.02, 0.70, 0.62, 0.86],
  ];

  let bin = (level as usize).min(3);
  let mut rate = COSTS[prev_state][bin];

  if level > 0 {
    rate += 1.0; // sign bit
  }

  if level > 2 {
    // Base range + Golomb coding
    let excess = level - 3;
    if excess <= 12 {
      // ~0.19 bits per base_range symbol (4 iterations of 3-way choice)
      rate += (excess as f64) * 0.19;
    } else {
      rate += 12.0 * 0.19;
      // Golomb: ~2*log2(val) + 1 bits
      let golomb = excess - 12;
      let bits = if golomb == 0 {
        1.0
      } else {
        2.0 * (golomb as f64).log2().ceil() + 1.0
      };
      rate += bits;
    }
  }

  rate
}

/// Approximate cost (in bits) of coding EOB at position `pos`.
/// AV1 uses a hierarchical code: eob_pt + eob_extra + eob_offset_bits.
/// The total cost is roughly 2 + log2(pos) bits.
#[inline]
fn eob_coding_cost(pos: usize) -> f64 {
  if pos <= 1 {
    return 1.0;
  }
  // eob_pt cost ≈ 1 + log2(pos) bits (prefix code over log-scaled groups)
  // eob_extra/offset ≈ 1 bit average
  1.0 + (pos as f64).log2() + 1.0
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sq_err() {
    assert_eq!(sq_err(100, 0, 10, 0), 10000);
    assert_eq!(sq_err(100, 10, 10, 0), 0);
    assert_eq!(sq_err(100, 9, 10, 0), 100);
    assert_eq!(sq_err(100, 0, 10, 1), 2500);
  }

  #[test]
  fn test_coeff_coding_cost_basic() {
    // Zero in zero context should be very cheap
    let cost_zero = coeff_coding_cost(0, 0);
    assert!(cost_zero < 0.5);

    // Level 1 should cost more than level 0
    let cost_one = coeff_coding_cost(0, 1);
    assert!(cost_one > cost_zero);
  }

  #[test]
  fn test_eob_coding_cost_monotonic() {
    let c1 = eob_coding_cost(1);
    let c10 = eob_coding_cost(10);
    let c100 = eob_coding_cost(100);
    assert!(c1 < c10);
    assert!(c10 < c100);
  }
}
