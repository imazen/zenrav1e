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
//! Uses rav1e's RDO lambda and CDF-based rate estimation from the actual
//! AV1 entropy coder state. Purely encoder-side — produces valid AV1 bitstream.

use crate::context::{
  BR_CDF_SIZE, CDFContext, COEFF_BASE_RANGE, ContextWriter, NUM_BASE_LEVELS,
  TX_PAD_2D, TX_PAD_HOR, TX_PAD_TOP, TxClass, TxClass::*,
  av1_get_coded_tx_size, tx_type_to_class,
};
use crate::quantize::QuantizationContext;
use crate::quantize::qm_tables::AOM_QM_BITS;
use crate::scan_order::av1_scan_orders;
use crate::transform::{TxSize, TxType};
use crate::util::*;

/// EC_PROB_SHIFT from ec.rs — lower 6 bits of CDF entries hold adaptation count.
const EC_PROB_SHIFT: u32 = 6;

/// CDF probability total after shifting: 32768 >> 6 = 512.
const CDF_TOTAL: u32 = 32768 >> EC_PROB_SHIFT;

/// Optimize quantized coefficients using CDF-based rate-distortion optimization.
///
/// `qcoeffs`: quantized coefficients (modified in-place)
/// `coeffs`: original transform-domain coefficients (pre-quantization)
/// `qc`: quantization context (provides DC/AC quantizers)
/// `tx_size`: transform block size
/// `tx_type`: transform type (for scan order)
/// `lambda`: Lagrangian multiplier from RDO
/// `qm`: optional quantization matrix weights
/// `eob`: current end-of-block position (1-based, in scan order)
/// `fc`: CDF context (frozen snapshot — not updated during trellis)
/// `plane_type`: 0 for luma, 1 for chroma
///
/// Returns the new eob after optimization.
pub fn optimize<T: Coefficient>(
  qcoeffs: &mut [T], coeffs: &[T], qc: &QuantizationContext, tx_size: TxSize,
  tx_type: TxType, lambda: f64, qm: Option<&[u8]>, eob: u16, fc: &CDFContext,
  plane_type: usize,
) -> u16 {
  let scan = &av1_scan_orders[tx_size as usize][tx_type as usize].scan;
  let n = eob as usize;

  if n <= 1 {
    return eob;
  }

  let log_tx_scale = crate::quantize::get_log_tx_scale(tx_size);
  let ac_quant = qc.ac_quant() as u32;

  // Lambda calibration: our distortion is 2^(6-2*lts) times rav1e's ScaledDist.
  // See previous comments in git history for full derivation.
  //
  // Quality-adaptive dampening: transform-domain MSE diverges from perceptual
  // quality (SSIMULACRA2) at low quality where quantizer steps are large.
  // Without dampening, the trellis over-optimizes at Q50 (-0.5 SS2).
  // Scale by ac_quant: small quantizer (high quality) → full strength,
  // large quantizer (low quality) → reduced strength.
  // ac_quant ~40 at Q95, ~200 at Q80, ~800 at Q50.
  let tx_dist_scale = (1u64 << (6 - 2 * log_tx_scale)) as f64;
  // Skip trellis when quantizer is too coarse for beneficial optimization.
  // At ac_quant >= 200 (~Q80 and below), the dampening would be ≤ 0.4
  // and the effect is empirically zero (< 0.01% BPP savings).
  if ac_quant >= 200 {
    return eob;
  }
  let dampening = (80.0 / (ac_quant as f64).max(80.0)).min(1.0);
  let lambda_trellis = lambda * tx_dist_scale * dampening;

  let tx_class = tx_type_to_class[tx_type as usize];
  let txs_ctx = ContextWriter::get_txsize_entropy_ctx(tx_size);
  let bhl = ContextWriter::get_txb_bhl(tx_size);
  let coded_size = av1_get_coded_tx_size(tx_size);
  let area = coded_size.area();
  let height = coded_size.height();
  let stride = height + TX_PAD_HOR;

  // Build padded levels buffer from current quantized coefficients.
  // Same layout as txb_init_levels: column-major with TX_PAD_HOR padding.
  let mut levels_buf = [0u8; TX_PAD_2D];
  let levels = &mut levels_buf[TX_PAD_TOP * stride..];
  init_levels(qcoeffs, height, levels, stride);

  // Pre-compute contexts for each scan position.
  // sig_ctx: non-EOB significance context (0..25 for 2D)
  // eob_ctx: EOB context (0..3, based on scan position quartile)
  // br_ctx:  base range context (0..20)
  let mut sig_ctx = [0usize; 32 * 32];
  let mut eob_ctx_arr = [0usize; 32 * 32];
  let mut br_ctx_arr = [0usize; 32 * 32];

  for i in 0..n {
    let pos = scan[i] as usize;
    sig_ctx[i] = ContextWriter::get_nz_map_ctx(
      levels, pos, bhl, area, i, false, tx_size, tx_class,
    );
    eob_ctx_arr[i] = ContextWriter::get_nz_map_ctx(
      levels, pos, bhl, area, i, true, tx_size, tx_class,
    );
    br_ctx_arr[i] = ContextWriter::get_br_ctx(levels, pos, bhl, tx_class);
  }

  // Current EOB position rate.
  let eob_rate_current =
    eob_position_rate(n as u16, tx_size, tx_class, fc, txs_ctx, plane_type);

  // Phase 1: EOB shrinkage via backward scan.
  //
  // For each candidate new_eob = i (from n-1 down to 1):
  //   - Accumulate distortion from zeroing positions i..n-1
  //   - Accumulate rate savings from not coding those coefficients
  //   - Account for EOB position coding change
  //   - Account for context switch: position i-1 changes from non-EOB to EOB
  let mut best_new_eob = n;
  let mut best_net_cost = 0.0f64;
  let mut rate_accum = 0.0f64;
  let mut dist_accum = 0.0f64;

  for i in (1..n).rev() {
    let pos = scan[i] as usize;
    let level = i32::cast_from(qcoeffs[pos]).unsigned_abs();

    if level == 0 {
      continue;
    }

    // Rate of this coefficient in its current role (EOB or non-EOB).
    let current_rate = if i == n - 1 {
      coeff_rate(
        level,
        eob_ctx_arr[i],
        br_ctx_arr[i],
        true,
        fc,
        txs_ctx,
        plane_type,
      )
    } else {
      coeff_rate(
        level,
        sig_ctx[i],
        br_ctx_arr[i],
        false,
        fc,
        txs_ctx,
        plane_type,
      )
    };
    rate_accum += current_rate;

    // Distortion increase from zeroing this coefficient.
    let eff_quant = effective_ac_quant(ac_quant, pos, qm);
    if eff_quant > 0 {
      let coeff_raw = i32::cast_from(coeffs[pos]);
      let coeff = (coeff_raw as i64) << log_tx_scale;
      let dist_keep = sq_err(coeff, level, eff_quant, log_tx_scale);
      let dist_zero = sq_err(coeff, 0, eff_quant, log_tx_scale);
      dist_accum += (dist_zero - dist_keep) as f64;
    }

    // Can only set new_eob = i if position i-1 is non-zero (EOB must be non-zero).
    let prev_pos = scan[i - 1] as usize;
    let prev_level = i32::cast_from(qcoeffs[prev_pos]).unsigned_abs();
    if prev_level == 0 {
      continue;
    }

    // Context switch cost: position i-1 changes from non-EOB to EOB role.
    let non_eob_rate_prev = coeff_rate(
      prev_level,
      sig_ctx[i - 1],
      br_ctx_arr[i - 1],
      false,
      fc,
      txs_ctx,
      plane_type,
    );
    let eob_rate_prev = coeff_rate(
      prev_level,
      eob_ctx_arr[i - 1],
      br_ctx_arr[i - 1],
      true,
      fc,
      txs_ctx,
      plane_type,
    );
    // Positive = EOB context costs more; negative = EOB context is cheaper.
    let switch_cost = eob_rate_prev - non_eob_rate_prev;

    // EOB position coding change.
    let eob_rate_new =
      eob_position_rate(i as u16, tx_size, tx_class, fc, txs_ctx, plane_type);
    let eob_pos_saved = eob_rate_current - eob_rate_new;

    // Total rate saved by moving EOB from n to i.
    let rate_saved = rate_accum + eob_pos_saved - switch_cost;

    // Net cost: positive = distortion increase outweighs savings.
    let net = dist_accum - lambda_trellis * rate_saved;
    if net < best_net_cost {
      best_net_cost = net;
      best_new_eob = i;
    }
  }

  // Apply EOB shrinkage.
  if best_new_eob < n {
    for i in best_new_eob..n {
      let pos = scan[i] as usize;
      qcoeffs[pos] = T::cast_from(0);
    }
  }

  // Phase 2: Level round-down for interior coefficients.
  //
  // Rebuild levels and recompute contexts after EOB shrinkage,
  // then check each coefficient >= 2 for beneficial round-down.
  if best_new_eob > 1 {
    // Rebuild levels from modified qcoeffs.
    levels_buf = [0u8; TX_PAD_2D];
    let levels = &mut levels_buf[TX_PAD_TOP * stride..];
    init_levels(qcoeffs, height, levels, stride);

    // Recompute contexts for the new block.
    for i in 0..best_new_eob {
      let pos = scan[i] as usize;
      let is_eob = i == best_new_eob - 1;
      if is_eob {
        eob_ctx_arr[i] = ContextWriter::get_nz_map_ctx(
          levels, pos, bhl, area, i, true, tx_size, tx_class,
        );
      } else {
        sig_ctx[i] = ContextWriter::get_nz_map_ctx(
          levels, pos, bhl, area, i, false, tx_size, tx_class,
        );
      }
      br_ctx_arr[i] = ContextWriter::get_br_ctx(levels, pos, bhl, tx_class);
    }

    for i in 1..best_new_eob {
      let pos = scan[i] as usize;
      let coeff_raw = i32::cast_from(coeffs[pos]);
      let orig_level = i32::cast_from(qcoeffs[pos]).unsigned_abs();

      if orig_level < 2 {
        continue;
      }

      let eff_quant = effective_ac_quant(ac_quant, pos, qm);
      if eff_quant == 0 {
        continue;
      }

      let coeff = (coeff_raw as i64) << log_tx_scale;
      let new_level = orig_level - 1;

      let dist_orig = sq_err(coeff, orig_level, eff_quant, log_tx_scale);
      let dist_new = sq_err(coeff, new_level, eff_quant, log_tx_scale);
      let dd = (dist_new - dist_orig) as f64;

      let is_eob_coeff = i == best_new_eob - 1;
      let ctx = if is_eob_coeff { eob_ctx_arr[i] } else { sig_ctx[i] };

      let rate_orig = coeff_rate(
        orig_level,
        ctx,
        br_ctx_arr[i],
        is_eob_coeff,
        fc,
        txs_ctx,
        plane_type,
      );
      let rate_new = coeff_rate(
        new_level,
        ctx,
        br_ctx_arr[i],
        is_eob_coeff,
        fc,
        txs_ctx,
        plane_type,
      );
      let rate_saved = rate_orig - rate_new;

      if rate_saved > 0.0 && dd < lambda_trellis * rate_saved {
        let sign = if coeff_raw < 0 { -1i32 } else { 1 };
        qcoeffs[pos] = T::cast_from(sign * new_level as i32);
      }
    }
  }

  // Final EOB: find last non-zero in scan order.
  scan[..n]
    .iter()
    .rposition(|&pos| qcoeffs[pos as usize] != T::cast_from(0))
    .map(|i| i + 1)
    .unwrap_or(0) as u16
}

// ---------------------------------------------------------------------------
// Rate estimation helpers
// ---------------------------------------------------------------------------

/// Approximate rate (in bits) for coding symbol `s` given a CDF table.
///
/// Uses the information-theoretic cost: -log2(p(s)) where probability is
/// derived from the CDF entries. This matches the actual entropy coder cost
/// to within ~0.1 bits on average (the encoder state `rng` introduces some
/// variation, but this averages out over a block).
#[inline]
fn cdf_rate(s: u32, cdf: &[u16]) -> f64 {
  let fh = (cdf[s as usize] >> EC_PROB_SHIFT) as u32;
  let fl = if s > 0 {
    (cdf[(s - 1) as usize] >> EC_PROB_SHIFT) as u32
  } else {
    CDF_TOTAL
  };
  let range = fl.saturating_sub(fh).max(1);
  // -log2(range / CDF_TOTAL) = log2(CDF_TOTAL / range)
  (CDF_TOTAL as f64 / range as f64).log2()
}

/// Rate (in bits) for coding one coefficient at a given level.
///
/// Includes: base level CDF + sign bit + base range coding + Golomb overflow.
/// Uses the actual CDF tables from the entropy coder state.
#[inline]
fn coeff_rate(
  level: u32, ctx: usize, br_ctx: usize, is_eob: bool, fc: &CDFContext,
  txs_ctx: usize, plane_type: usize,
) -> f64 {
  let mut rate = 0.0;

  // Base level coding (significance).
  if is_eob {
    // EOB coefficient: symbol = min(level, 3) - 1, 3 symbols (0,1,2).
    let sym = level.min(3) - 1;
    rate += cdf_rate(sym, &fc.coeff_base_eob_cdf[txs_ctx][plane_type][ctx]);
  } else {
    // Non-EOB: symbol = min(level, 3), 4 symbols (0,1,2,3).
    let sym = level.min(3);
    rate += cdf_rate(sym, &fc.coeff_base_cdf[txs_ctx][plane_type][ctx]);
  }

  // Sign bit (AC signs are raw bits; DC uses CDF but ~1 bit either way).
  if level > 0 {
    rate += 1.0;
  }

  // Base range coding for levels above NUM_BASE_LEVELS (2).
  if level > NUM_BASE_LEVELS as u32 {
    let base_range = level - 1 - NUM_BASE_LEVELS as u32;
    let br_cdf = &fc.coeff_br_cdf[txs_ctx.min(TxSize::TX_32X32 as usize)]
      [plane_type][br_ctx];
    let mut idx = 0u32;

    loop {
      if idx >= COEFF_BASE_RANGE as u32 {
        break;
      }
      let k = (base_range - idx).min(BR_CDF_SIZE as u32 - 1);
      rate += cdf_rate(k, br_cdf);
      if k < BR_CDF_SIZE as u32 - 1 {
        break;
      }
      idx += BR_CDF_SIZE as u32 - 1;
    }

    // Golomb coding for excess beyond COEFF_BASE_RANGE.
    if base_range >= COEFF_BASE_RANGE as u32 {
      let golomb_val = base_range - COEFF_BASE_RANGE as u32;
      rate += golomb_cost(golomb_val);
    }
  }

  rate
}

/// Rate (in bits) for coding an EOB position.
///
/// AV1 codes EOB as: eob_pt (from size-specific CDF) + optional eob_extra
/// (CDF for first bit, raw for remaining bits).
#[inline]
fn eob_position_rate(
  eob: u16, tx_size: TxSize, tx_class: TxClass, fc: &CDFContext,
  txs_ctx: usize, plane_type: usize,
) -> f64 {
  use crate::context::k_eob_offset_bits;

  let (eob_pt, eob_extra) = ContextWriter::get_eob_pos_token(eob);
  let eob_multi_size = tx_size.area_log2() - 4;
  let eob_multi_ctx = if tx_class != TX_CLASS_2D { 1 } else { 0 };

  let eob_sym = eob_pt - 1;
  let mut rate = match eob_multi_size {
    0 => cdf_rate(eob_sym, &fc.eob_flag_cdf16[plane_type][eob_multi_ctx]),
    1 => cdf_rate(eob_sym, &fc.eob_flag_cdf32[plane_type][eob_multi_ctx]),
    2 => cdf_rate(eob_sym, &fc.eob_flag_cdf64[plane_type][eob_multi_ctx]),
    3 => cdf_rate(eob_sym, &fc.eob_flag_cdf128[plane_type][eob_multi_ctx]),
    4 => cdf_rate(eob_sym, &fc.eob_flag_cdf256[plane_type][eob_multi_ctx]),
    5 => cdf_rate(eob_sym, &fc.eob_flag_cdf512[plane_type][eob_multi_ctx]),
    _ => cdf_rate(eob_sym, &fc.eob_flag_cdf1024[plane_type][eob_multi_ctx]),
  };

  // Extra bits for EOB offset within the group.
  let eob_offset_bits = k_eob_offset_bits[eob_pt as usize];
  if eob_offset_bits > 0 {
    // First extra bit via CDF.
    let eob_shift = eob_offset_bits - 1;
    let bit = (eob_extra >> eob_shift) & 1;
    rate += cdf_rate(
      bit,
      &fc.eob_extra_cdf[txs_ctx][plane_type][(eob_pt - 3) as usize],
    );
    // Remaining bits are raw (1 bit each).
    rate += (eob_offset_bits - 1) as f64;
  }

  rate
}

/// Cost (in bits) of Golomb coding a value.
#[inline]
fn golomb_cost(val: u32) -> f64 {
  let x = val + 1;
  let length = 32 - x.leading_zeros(); // bit_length
  (2 * length - 1) as f64
}

/// Build padded levels buffer from quantized coefficients.
/// Same layout as `txb_init_levels`: column-major with TX_PAD_HOR padding.
fn init_levels<T: Coefficient>(
  qcoeffs: &[T], height: usize, levels: &mut [u8], stride: usize,
) {
  for (col_coeffs, col_levels) in
    qcoeffs.chunks(height).zip(levels.chunks_mut(stride))
  {
    for (coeff, level) in col_coeffs.iter().zip(col_levels.iter_mut()) {
      *level = coeff.abs().min(T::cast_from(127)).as_();
    }
  }
}

/// Effective AC quantizer for a given coefficient position, accounting for QM.
#[inline]
fn effective_ac_quant(
  base_quant: u32, scan_pos: usize, qm: Option<&[u8]>,
) -> u32 {
  match qm {
    Some(qm_tbl) if scan_pos < qm_tbl.len() => {
      let wt = qm_tbl[scan_pos] as u32;
      (base_quant * wt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS
    }
    _ => base_quant,
  }
}

/// Squared error between original coefficient and reconstruction.
#[inline]
fn sq_err(
  coeff_shifted: i64, level: u32, quant: u32, log_tx_scale: usize,
) -> i64 {
  let recon = level as i64 * quant as i64;
  let err = coeff_shifted.abs() - recon;
  (err * err) >> (2 * log_tx_scale)
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
  fn test_cdf_rate_uniform() {
    // Uniform 4-symbol CDF: each symbol gets 1/4 probability.
    // CDF values (15-bit): [384<<6, 256<<6, 128<<6, count=5]
    let cdf = [384 << 6, 256 << 6, 128 << 6, 5];
    for s in 0..4 {
      let rate = cdf_rate(s, &cdf);
      assert!(
        (rate - 2.0).abs() < 0.01,
        "uniform 4-symbol: s={s}, rate={rate}, expected ~2.0"
      );
    }
  }

  #[test]
  fn test_cdf_rate_skewed() {
    // Skewed CDF: P(0) = 3/4, P(1) = 1/4.
    // CDF: [128<<6, count=5] → fh(0)=128, fl(0)=512, range=384
    // P(1): fl=128, fh=0, range=128
    let cdf = [128 << 6, 5];
    let r0 = cdf_rate(0, &cdf);
    let r1 = cdf_rate(1, &cdf);
    assert!(r0 < r1, "high-prob symbol should be cheaper: r0={r0}, r1={r1}");
    assert!((r0 - 0.415).abs() < 0.1, "P(0)=3/4 → ~0.415 bits, got {r0}");
    assert!((r1 - 2.0).abs() < 0.1, "P(1)=1/4 → ~2.0 bits, got {r1}");
  }

  #[test]
  fn test_golomb_cost() {
    assert_eq!(golomb_cost(0), 1.0); // x=1, length=1, cost=1
    assert_eq!(golomb_cost(1), 3.0); // x=2, length=2, cost=3
    assert_eq!(golomb_cost(5), 5.0); // x=6, length=3, cost=5
  }
}
