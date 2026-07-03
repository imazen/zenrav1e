// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! AV1 palette mode: encoder-side color search and the shared helpers used by
//! the bitstream writers in `context::block_unit`.
//!
//! Syntax references (all traced 2026-07-02):
//! - rav1d-safe `decode.rs` (`order_palette`, `read_pal_indices`) and
//!   `recon.rs` (`rav1d_read_pal_plane`) — the read-side dual of every writer
//!   here; rav1d is the conformance oracle for this module.
//! - libaom `av1/encoder/palette.c` (`av1_rd_pick_palette_intra_sby`,
//!   `av1_k_means` templates) and `av1/encoder/bitstream.c`
//!   (`write_palette_colors_y`, `pack_map_tokens`) at the pinned rev 632172a4.
//!
//! Scope: luma palette only. The UV palette flag is still written (as
//! "off") by `write_use_palette_mode` when the chroma mode is DC_PRED, which
//! is all a conforming bitstream needs; UV palette *search* is not
//! implemented.

use crate::util::Pixel;
use arrayvec::ArrayVec;

/// Minimum number of colors in a palette (AV1 spec).
pub const PALETTE_MIN_SIZE: usize = 2;
/// Maximum number of colors in a palette (AV1 spec).
pub const PALETTE_MAX_SIZE: usize = 8;

/// Maximum entries in the neighbor color cache (above 8 + left 8).
pub const PALETTE_CACHE_SIZE: usize = 2 * PALETTE_MAX_SIZE;

/// Largest block edge that palette mode applies to (AV1 spec: both
/// dimensions must be <= 64).
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub const MAX_PALETTE_BLOCK_SIZE: usize = 64;

/// Upper bound on candidates returned by [`palette_candidates_y`]:
/// 7 top-color sizes + 7 k-means sizes.
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub const MAX_PALETTE_CANDIDATES: usize = 14;

/// Encoder-side gate on the number of distinct colors in a block for the
/// palette search to run at all, mirroring libaom's RD-path default
/// (`x->color_palette_thresh = 64`). Blocks with more distinct (8-bit
/// domain) colors than this are photographic, not palettizable.
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub const PALETTE_COLOR_COUNT_THRESH: usize = 64;

/// A sorted, deduplicated set of palette colors for one plane.
pub type PaletteColors = ArrayVec<u16, PALETTE_MAX_SIZE>;

/// A chosen luma palette for one block: the colors plus the per-pixel color
/// index map (full block dimensions, row-major, stride = block width).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteData {
  /// Sorted, strictly increasing luma colors (2..=8 of them).
  pub colors: PaletteColors,
  /// Color index map, `block_width * block_height` entries, row-major.
  /// Values are indices into `colors`.
  pub map: Vec<u8>,
}

impl PaletteData {
  #[inline]
  pub fn size(&self) -> usize {
    self.colors.len()
  }
}

/// Merges the sorted palette color lists of the above and left neighbors
/// into the palette color cache, mirroring rav1d's `rav1d_read_pal_plane`
/// cache construction (== libaom `av1_get_palette_cache`): a sorted merge
/// that drops values equal to the previously emitted one.
///
/// Both inputs must be sorted non-decreasing (palette colors are coded
/// strictly increasing for luma, so this holds by construction).
pub fn merge_palette_cache(
  above: &[u16], left: &[u16],
) -> ArrayVec<u16, PALETTE_CACHE_SIZE> {
  let mut cache = ArrayVec::new();
  let push = |cache: &mut ArrayVec<u16, PALETTE_CACHE_SIZE>, v: u16| {
    if cache.last() != Some(&v) {
      cache.push(v);
    }
  };
  let (mut a, mut l) = (above, left);
  while !a.is_empty() && !l.is_empty() {
    if l[0] < a[0] {
      push(&mut cache, l[0]);
      l = &l[1..];
    } else {
      if a[0] == l[0] {
        l = &l[1..];
      }
      push(&mut cache, a[0]);
      a = &a[1..];
    }
  }
  for &v in l {
    push(&mut cache, v);
  }
  for &v in a {
    push(&mut cache, v);
  }
  cache
}

/// Computes the color-index context and the canonical color order for map
/// position `(r, c)`, from the already-assigned neighbors (left, top,
/// top-left) in `map`. Mirrors rav1d's `order_palette` inner step, which is
/// the compact equivalent of the AV1 spec's score-based
/// `get_palette_color_context` (equivalence unit-tested below against a
/// direct port of the spec formulation).
///
/// The coded symbol for position `(r, c)` is the position of
/// `map[r * stride + c]` in the returned order.
#[inline]
pub fn palette_color_ctx_and_order(
  map: &[u8], stride: usize, r: usize, c: usize,
) -> (usize, [u8; PALETTE_MAX_SIZE]) {
  debug_assert!(r > 0 || c > 0);
  let mut order = [0u8; PALETTE_MAX_SIZE];
  let mut mask = 0u8;
  let mut o_idx = 0;
  let add = |order: &mut [u8; PALETTE_MAX_SIZE],
             mask: &mut u8,
             o_idx: &mut usize,
             v: u8| {
    debug_assert!((v as usize) < PALETTE_MAX_SIZE);
    order[*o_idx] = v;
    *o_idx += 1;
    *mask |= 1 << v;
  };

  let ctx;
  if c == 0 {
    // Only a top neighbor.
    ctx = 0;
    add(&mut order, &mut mask, &mut o_idx, map[(r - 1) * stride + c]);
  } else if r == 0 {
    // Only a left neighbor.
    ctx = 0;
    add(&mut order, &mut mask, &mut o_idx, map[r * stride + c - 1]);
  } else {
    let l = map[r * stride + c - 1];
    let t = map[(r - 1) * stride + c];
    let tl = map[(r - 1) * stride + c - 1];
    let same_t_l = t == l;
    let same_t_tl = t == tl;
    let same_l_tl = l == tl;
    let same_all = same_t_l && same_t_tl && same_l_tl;
    if same_all {
      ctx = 4;
      add(&mut order, &mut mask, &mut o_idx, t);
    } else if same_t_l {
      ctx = 3;
      add(&mut order, &mut mask, &mut o_idx, t);
      add(&mut order, &mut mask, &mut o_idx, tl);
    } else if same_t_tl || same_l_tl {
      ctx = 2;
      add(&mut order, &mut mask, &mut o_idx, tl);
      add(&mut order, &mut mask, &mut o_idx, if same_t_tl { l } else { t });
    } else {
      ctx = 1;
      add(&mut order, &mut mask, &mut o_idx, t.min(l));
      add(&mut order, &mut mask, &mut o_idx, t.max(l));
      add(&mut order, &mut mask, &mut o_idx, tl);
    }
  }
  for bit in 0..PALETTE_MAX_SIZE as u8 {
    if mask & (1 << bit) == 0 {
      order[o_idx] = bit;
      o_idx += 1;
    }
  }
  debug_assert_eq!(o_idx, PALETTE_MAX_SIZE);
  (ctx, order)
}

/// Walks the color index map in bitstream order (the wavefront/antidiagonal
/// order shared by rav1d's `read_pal_indices` and libaom's
/// `cost_and_tokenize_map`), skipping position `(0, 0)` (coded separately
/// with the quasi-uniform code). Calls `f(ctx, coded_symbol)` for every
/// remaining position.
///
/// `rows`/`cols` are the *visible* dimensions (clipped to the frame edge);
/// `stride` is the full block width.
pub fn foreach_color_index_symbol(
  map: &[u8], stride: usize, rows: usize, cols: usize,
  mut f: impl FnMut(usize, u32),
) {
  for k in 1..rows + cols - 1 {
    let first = k.min(cols - 1);
    let last = (k + 1).saturating_sub(rows);
    for j in (last..=first).rev() {
      let r = k - j;
      let c = j;
      let (ctx, order) = palette_color_ctx_and_order(map, stride, r, c);
      let actual = map[r * stride + c];
      let sym = order.iter().position(|&o| o == actual).unwrap() as u32;
      f(ctx, sym);
    }
  }
}

/// `ceil(log2(n))` with `ceil_log2(0) == ceil_log2(1) == 0`, matching
/// libaom's `aom_ceil_log2`.
#[inline]
pub const fn ceil_log2(n: usize) -> u32 {
  if n < 2 { 0 } else { usize::BITS - (n - 1).leading_zeros() }
}

/// Splits `colors` into (indices of colors found in `cache`, colors not in
/// the cache), mirroring libaom's `av1_index_color_cache`. Returns the
/// per-cache-entry "found" flags (aligned with `cache`) and the out-of-cache
/// colors in ascending order.
pub fn index_color_cache(
  cache: &[u16], colors: &[u16],
) -> (ArrayVec<bool, PALETTE_CACHE_SIZE>, PaletteColors) {
  let mut found_flags: ArrayVec<bool, PALETTE_CACHE_SIZE> = ArrayVec::new();
  let mut in_cache = [false; PALETTE_MAX_SIZE];
  let mut n_in_cache = 0;
  for &cv in cache {
    let mut found = false;
    if n_in_cache < colors.len() {
      for (j, &c) in colors.iter().enumerate() {
        if !in_cache[j] && c == cv {
          in_cache[j] = true;
          n_in_cache += 1;
          found = true;
          break;
        }
      }
    }
    found_flags.push(found);
  }
  let mut rest = PaletteColors::new();
  for (j, &c) in colors.iter().enumerate() {
    if !in_cache[j] {
      rest.push(c);
    }
  }
  (found_flags, rest)
}

/// Number of distinct color values in the flattened block data, using a
/// caller-provided histogram buffer of at least `1 << bit_depth` entries.
/// The buffer is zeroed here. Also returns the (lower, upper) value bounds.
pub fn count_colors(
  data: &[i16], bit_depth: usize, histogram: &mut [u32],
) -> (usize, i16, i16) {
  let hist = &mut histogram[..1 << bit_depth];
  hist.fill(0);
  let mut lower = i16::MAX;
  let mut upper = i16::MIN;
  let mut colors = 0usize;
  for &v in data {
    debug_assert!(v >= 0 && (v as usize) < (1 << bit_depth));
    let slot = &mut hist[v as usize];
    if *slot == 0 {
      colors += 1;
    }
    *slot += 1;
    lower = lower.min(v);
    upper = upper.max(v);
  }
  (colors, lower, upper)
}

/// The up-to-8 most frequent colors, sorted by count descending (ties:
/// lower color value first) — libaom's `find_top_colors`.
fn find_top_colors(histogram: &[u32], n: usize) -> ArrayVec<i16, 8> {
  // (count, color) pairs; sort by count desc then color asc.
  let mut top: ArrayVec<(u32, i16), 8> = ArrayVec::new();
  for (color, &count) in histogram.iter().enumerate() {
    if count == 0 {
      continue;
    }
    let key = (count, color as i16);
    if top.len() < n {
      top.push(key);
      if top.len() == n {
        top.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
      }
    } else if count > top[n - 1].0 {
      // Insert maintaining order (count desc, color asc for equal counts;
      // a strictly greater count displaces the tail).
      let mut j = n - 1;
      while j >= 1 && count > top[j - 1].0 {
        j -= 1;
      }
      top.pop();
      top.insert(j, key);
    }
  }
  top.iter().map(|&(_, c)| c).collect()
}

/// libaom's `lcg_rand16` (used only to re-seed empty k-means clusters,
/// deterministically).
#[inline]
fn lcg_rand16(state: &mut u32) -> u16 {
  *state = state.wrapping_mul(1103515245).wrapping_add(12345);
  ((*state / 65536) % 32768) as u16
}

/// Nearest-centroid assignment (1-D). Strict `<` so the lowest index wins
/// ties, matching libaom's `av1_calc_indices_dim1`. Returns the summed
/// squared distance.
fn calc_indices(data: &[i16], centroids: &[i16], indices: &mut [u8]) -> u64 {
  let mut dist = 0u64;
  for (v, idx) in data.iter().zip(indices.iter_mut()) {
    let mut min_d = (v - centroids[0]).unsigned_abs() as u32;
    let mut min_i = 0u8;
    for (j, &cent) in centroids.iter().enumerate().skip(1) {
      let d = (v - cent).unsigned_abs() as u32;
      if d < min_d {
        min_d = d;
        min_i = j as u8;
      }
    }
    *idx = min_i;
    dist += u64::from(min_d) * u64::from(min_d);
  }
  dist
}

/// Centroid update step (1-D), matching libaom's `calc_centroids_dim1`:
/// rounded mean per cluster; empty clusters re-seeded from a deterministic
/// LCG-selected data point.
fn calc_centroids(data: &[i16], centroids: &mut [i16], indices: &[u8]) {
  let k = centroids.len();
  let mut count = [0u32; PALETTE_MAX_SIZE];
  let mut sum = [0i64; PALETTE_MAX_SIZE];
  let mut rand_state = data[0] as u16 as u32;
  for (&v, &idx) in data.iter().zip(indices.iter()) {
    let idx = idx as usize;
    debug_assert!(idx < k);
    count[idx] += 1;
    sum[idx] += i64::from(v);
  }
  for i in 0..k {
    if count[i] == 0 {
      centroids[i] = data[lcg_rand16(&mut rand_state) as usize % data.len()];
    } else {
      // libaom's DIVIDE_AND_ROUND; sums are non-negative (pixel values).
      debug_assert!(sum[i] >= 0);
      centroids[i] =
        ((sum[i] + i64::from(count[i] / 2)) / i64::from(count[i])) as i16;
    }
  }
}

/// 1-D k-means, ported from libaom's `av1_k_means_dim1` template: iterate
/// assign/update up to `max_itr` times, keeping the best-distortion state
/// (the search stops early when centroids converge or distortion increases).
fn k_means(data: &[i16], centroids: &mut [i16], max_itr: usize) {
  let k = centroids.len();
  let mut cent_a: ArrayVec<i16, PALETTE_MAX_SIZE> =
    centroids.iter().copied().collect();
  let mut cent_b = cent_a.clone();
  let mut idx_a = vec![0u8; data.len()];
  let mut idx_b = vec![0u8; data.len()];

  let mut this_dist = calc_indices(data, &cent_a, &mut idx_a);
  // `l` selects which of (a, b) is current, starting at a; `best_l` tracks
  // the state with the lowest seen distortion, exactly like libaom's
  // meta_centroids/meta_indices juggling.
  let mut l = 0usize;
  let mut best_l = 0usize;
  let mut i = 0usize;
  while i < max_itr {
    let prev_dist = this_dist;
    let prev_l = l;
    l = 1 - l;
    {
      let (cur, prev): (&mut ArrayVec<_, 8>, &ArrayVec<_, 8>) =
        if l == 1 { (&mut cent_b, &cent_a) } else { (&mut cent_a, &cent_b) };
      cur.clear();
      cur.extend(prev.iter().copied());
      let prev_idx = if prev_l == 1 { &idx_b } else { &idx_a };
      calc_centroids(data, cur, prev_idx);
    }
    let (cur_cent, prev_cent) =
      if l == 1 { (&cent_b, &cent_a) } else { (&cent_a, &cent_b) };
    if cur_cent[..k] == prev_cent[..k] {
      break;
    }
    let cur_idx = if l == 1 { &mut idx_b } else { &mut idx_a };
    this_dist = calc_indices(data, cur_cent, cur_idx);
    if this_dist > prev_dist {
      best_l = prev_l;
      break;
    }
    i += 1;
  }
  if i == max_itr {
    best_l = l;
  }
  let best = if best_l == 1 { &cent_b } else { &cent_a };
  centroids.copy_from_slice(&best[..k]);
}

/// Snaps centroids to nearby cache colors (within `4 << (bit_depth - 8)`),
/// mirroring libaom's `optimize_palette_colors`: reusing a cached color is
/// nearly free to code, so close-enough centroids move onto cache entries.
fn snap_to_cache(centroids: &mut [i16], cache: &[u16], bit_depth: usize) {
  if cache.is_empty() {
    return;
  }
  let min_threshold = 4i32 << (bit_depth - 8);
  for cent in centroids.iter_mut() {
    let mut best = i32::MAX;
    let mut best_val = 0u16;
    for &cv in cache {
      let d = (i32::from(*cent) - i32::from(cv)).abs();
      if d < best {
        best = d;
        best_val = cv;
      }
    }
    if best <= min_threshold {
      *cent = best_val as i16;
    }
  }
}

/// Sorts, clamps to the pixel range, and deduplicates centroids into a final
/// color set (libaom's `remove_duplicates` after clipping). Returns `None`
/// if fewer than [`PALETTE_MIN_SIZE`] unique colors remain.
fn finish_candidate(
  centroids: &mut [i16], bit_depth: usize,
) -> Option<PaletteColors> {
  let max = (1i16 << bit_depth) - 1;
  for c in centroids.iter_mut() {
    *c = (*c).clamp(0, max);
  }
  centroids.sort_unstable();
  let mut out = PaletteColors::new();
  for &c in centroids.iter() {
    if out.last() != Some(&(c as u16)) {
      out.push(c as u16);
    }
  }
  if out.len() >= PALETTE_MIN_SIZE { Some(out) } else { None }
}

/// Flattens a block's source pixels into row-major `i16` data (libaom's
/// `fill_data_and_get_bounds` shape; bounds come from [`count_colors`]).
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub fn flatten_block<T: Pixel>(
  src: &crate::tiling::PlaneRegion<'_, T>, rows: usize, cols: usize,
) -> Vec<i16> {
  use crate::util::CastFromPrimitive;
  let mut data = Vec::with_capacity(rows * cols);
  for y in 0..rows {
    let row = &src[y];
    for x in 0..cols {
      data.push(i16::cast_from(row[x]));
    }
  }
  data
}

/// Runs the luma palette color search on a flattened block, producing
/// candidate palettes for RD evaluation. This is the libaom
/// `av1_rd_pick_palette_intra_sby` candidate generation (both the top-color
/// and k-means families, all sizes 2..=min(8, colors)), without the
/// header-rate pruning (zenrav1e trials candidates with real rates).
///
/// Returns an empty list when the block is not palettizable
/// (`colors <= 1 || colors > PALETTE_COLOR_COUNT_THRESH`).
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub fn palette_candidates_y(
  data: &[i16], bit_depth: usize, cache: &[u16], histogram: &mut [u32],
) -> ArrayVec<PaletteColors, MAX_PALETTE_CANDIDATES> {
  let mut out: ArrayVec<PaletteColors, MAX_PALETTE_CANDIDATES> =
    ArrayVec::new();
  debug_assert!(!data.is_empty());
  let (colors, lower, upper) = count_colors(data, bit_depth, histogram);
  // Threshold counting happens in the 8-bit domain like libaom's
  // `av1_count_colors_highbd` threshold path, so high bit depths don't
  // spuriously exceed the gate on smooth gradients.
  let colors_threshold = if bit_depth > 8 {
    let shift = bit_depth - 8;
    let mut bins = [false; 256];
    let mut n = 0usize;
    for (v, &count) in histogram[..1 << bit_depth].iter().enumerate() {
      if count > 0 && !std::mem::replace(&mut bins[v >> shift], true) {
        n += 1;
      }
    }
    n
  } else {
    colors
  };
  if colors_threshold <= 1 || colors_threshold > PALETTE_COLOR_COUNT_THRESH {
    return out;
  }

  let max_n = colors.min(PALETTE_MAX_SIZE);
  let top_colors = find_top_colors(&histogram[..1 << bit_depth], max_n);

  let push_unique =
    |out: &mut ArrayVec<PaletteColors, MAX_PALETTE_CANDIDATES>,
     cand: PaletteColors| {
      if !out.contains(&cand) {
        out.push(cand);
      }
    };

  // Family 1: dominant colors, ascending palette size.
  for n in PALETTE_MIN_SIZE..=max_n {
    let mut centroids: ArrayVec<i16, PALETTE_MAX_SIZE> =
      top_colors[..n].iter().copied().collect();
    snap_to_cache(&mut centroids, cache, bit_depth);
    if let Some(cand) = finish_candidate(&mut centroids, bit_depth) {
      push_unique(&mut out, cand);
    }
  }

  // Family 2: k-means refinement.
  if colors == PALETTE_MIN_SIZE {
    // Two distinct colors are their own optimal centroids.
    let mut centroids = [lower, upper];
    snap_to_cache(&mut centroids, cache, bit_depth);
    if let Some(cand) = finish_candidate(&mut centroids, bit_depth) {
      push_unique(&mut out, cand);
    }
  } else {
    const MAX_ITR: usize = 50;
    for n in PALETTE_MIN_SIZE..=max_n {
      let mut centroids: ArrayVec<i16, PALETTE_MAX_SIZE> = (0..n)
        .map(|i| {
          (i32::from(lower)
            + (2 * i as i32 + 1) * i32::from(upper - lower) / n as i32 / 2)
            as i16
        })
        .collect();
      k_means(data, &mut centroids, MAX_ITR);
      snap_to_cache(&mut centroids, cache, bit_depth);
      if let Some(cand) = finish_candidate(&mut centroids, bit_depth) {
        push_unique(&mut out, cand);
      }
    }
  }

  out
}

/// Builds the color index map for a block given its final palette: nearest
/// color, lowest index winning ties (libaom's `av1_calc_indices` on the
/// final integer palette).
#[allow(dead_code)] // used by the RDO palette search (landing next)
pub fn build_index_map(data: &[i16], colors: &[u16]) -> Vec<u8> {
  let cents: ArrayVec<i16, PALETTE_MAX_SIZE> =
    colors.iter().map(|&c| c as i16).collect();
  let mut map = vec![0u8; data.len()];
  calc_indices(data, &cents, &mut map);
  map
}

#[cfg(test)]
mod test {
  use super::*;

  /// Direct port of the AV1 spec / libaom score-based
  /// `av1_get_palette_color_index_context`, used only to verify the compact
  /// rav1d-derived formulation above.
  fn spec_ctx_and_order(
    map: &[u8], stride: usize, r: usize, c: usize,
  ) -> (usize, [u8; PALETTE_MAX_SIZE], usize) {
    const NUM_PALETTE_NEIGHBORS: usize = 3;
    let color_neighbors: [i32; NUM_PALETTE_NEIGHBORS] = [
      if c > 0 { map[r * stride + c - 1] as i32 } else { -1 },
      if c > 0 && r > 0 { map[(r - 1) * stride + c - 1] as i32 } else { -1 },
      if r > 0 { map[(r - 1) * stride + c] as i32 } else { -1 },
    ];
    let weights = [2i32, 1, 2];
    let mut scores = [0i32; PALETTE_MAX_SIZE];
    for i in 0..NUM_PALETTE_NEIGHBORS {
      if color_neighbors[i] >= 0 {
        scores[color_neighbors[i] as usize] += weights[i];
      }
    }
    let mut color_order: [u8; PALETTE_MAX_SIZE] = [0, 1, 2, 3, 4, 5, 6, 7];
    // Partial selection sort of the top 3 scores, stable shift like libaom.
    for i in 0..NUM_PALETTE_NEIGHBORS {
      let mut max = scores[i];
      let mut max_idx = i;
      for j in (i + 1)..PALETTE_MAX_SIZE {
        if scores[j] > max {
          max = scores[j];
          max_idx = j;
        }
      }
      if max_idx != i {
        let max_score = scores[max_idx];
        let max_color_order = color_order[max_idx];
        let mut k = max_idx;
        while k > i {
          scores[k] = scores[k - 1];
          color_order[k] = color_order[k - 1];
          k -= 1;
        }
        scores[i] = max_score;
        color_order[i] = max_color_order;
      }
    }
    let color_idx =
      color_order.iter().position(|&o| o == map[r * stride + c]).unwrap();
    // Context hash -> context via the spec lookup.
    let hash_multipliers = [1i32, 2, 2];
    let mut hash = 0i32;
    for i in 0..NUM_PALETTE_NEIGHBORS {
      hash += scores[i] * hash_multipliers[i];
    }
    const CONTEXT_LOOKUP: [i32; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];
    let ctx = CONTEXT_LOOKUP[hash as usize];
    assert!(ctx >= 0);
    (ctx as usize, color_order, color_idx)
  }

  /// Exhaustive equivalence of the compact ctx/order derivation vs the
  /// spec's score-based formulation: every (l, t, tl) neighbor combination,
  /// every palette size, plus the two edge cases (no-left / no-top).
  ///
  /// The full 8-element orders can differ beyond the neighbor-derived
  /// prefix only in ways that never matter: positions holding equal scores
  /// (zero) are tie-broken ascending in both. We check ctx and the coded
  /// symbol (position of the actual index), which is the bitstream-visible
  /// pair, for every possible actual index.
  #[test]
  fn ctx_and_order_matches_spec_formulation() {
    // Interior: all neighbor combinations, all palette sizes.
    for pal_sz in PALETTE_MIN_SIZE..=PALETTE_MAX_SIZE {
      let n = pal_sz as u8;
      for l in 0..n {
        for t in 0..n {
          for tl in 0..n {
            // 2x2 map, position (1,1): neighbors l=(1,0), t=(0,1), tl=(0,0).
            for actual in 0..n {
              let map = [tl, t, l, actual];
              let (ctx, order) = palette_color_ctx_and_order(&map, 2, 1, 1);
              let (sctx, _sorder, sidx) = spec_ctx_and_order(&map, 2, 1, 1);
              let idx = order.iter().position(|&o| o == actual).unwrap();
              assert_eq!(
                (ctx, idx),
                (sctx, sidx),
                "interior l={l} t={t} tl={tl} actual={actual} n={n}"
              );
            }
          }
        }
      }
    }
    // Edges: first row (no top) and first column (no left).
    for n in 2..=8u8 {
      for nb in 0..n {
        for actual in 0..n {
          // (0, 1): only left neighbor.
          let map_row = [nb, actual, 0, 0];
          let (ctx, order) = palette_color_ctx_and_order(&map_row, 2, 0, 1);
          let (sctx, _, sidx) = spec_ctx_and_order(&map_row, 2, 0, 1);
          let idx = order.iter().position(|&o| o == actual).unwrap();
          assert_eq!((ctx, idx), (sctx, sidx), "row0 nb={nb} actual={actual}");
          // (1, 0): only top neighbor.
          let map_col = [nb, 0, actual, 0];
          let (ctx, order) = palette_color_ctx_and_order(&map_col, 2, 1, 0);
          let (sctx, _, sidx) = spec_ctx_and_order(&map_col, 2, 1, 0);
          let idx = order.iter().position(|&o| o == actual).unwrap();
          assert_eq!((ctx, idx), (sctx, sidx), "col0 nb={nb} actual={actual}");
        }
      }
    }
  }

  /// The wavefront traversal must visit every non-(0,0) position exactly
  /// once, in an order where each position's left/top/topleft neighbors
  /// were already visited (or are (0,0)).
  #[test]
  fn wavefront_order_is_causal_and_complete() {
    for &(rows, cols) in
      &[(4usize, 4usize), (8, 4), (4, 8), (16, 16), (8, 32), (64, 64)]
    {
      let stride = cols;
      let map = vec![0u8; rows * cols];
      let mut visited = vec![false; rows * cols];
      visited[0] = true;
      let mut count = 0usize;
      // Reimplement the traversal shell to check coordinates; the symbol
      // helper itself is exercised via foreach_color_index_symbol below.
      for k in 1..rows + cols - 1 {
        let first = k.min(cols - 1);
        let last = (k + 1).saturating_sub(rows);
        for j in (last..=first).rev() {
          let (r, c) = (k - j, j);
          assert!(r < rows && c < cols, "({r},{c}) in {rows}x{cols}");
          assert!(!visited[r * stride + c]);
          if c > 0 {
            assert!(visited[r * stride + c - 1], "left of ({r},{c})");
          }
          if r > 0 {
            assert!(visited[(r - 1) * stride + c], "top of ({r},{c})");
          }
          if r > 0 && c > 0 {
            assert!(visited[(r - 1) * stride + c - 1], "topleft of ({r},{c})");
          }
          visited[r * stride + c] = true;
          count += 1;
        }
      }
      assert_eq!(count, rows * cols - 1);
      let mut n = 0;
      foreach_color_index_symbol(&map, stride, rows, cols, |_, _| n += 1);
      assert_eq!(n, rows * cols - 1);
    }
  }

  /// Cache merge fixtures mirroring rav1d's semantics: sorted merge, dedup
  /// against the previously emitted entry, left-first on strictly smaller.
  #[test]
  fn cache_merge_matches_rav1d_semantics() {
    // Disjoint.
    assert_eq!(
      merge_palette_cache(&[10, 30], &[20, 40]).as_slice(),
      &[10, 20, 30, 40]
    );
    // Equal heads collapse.
    assert_eq!(
      merge_palette_cache(&[10, 30], &[10, 20]).as_slice(),
      &[10, 20, 30]
    );
    // One side empty.
    assert_eq!(merge_palette_cache(&[], &[5, 6]).as_slice(), &[5, 6]);
    assert_eq!(merge_palette_cache(&[5, 6], &[]).as_slice(), &[5, 6]);
    // Both empty.
    assert!(merge_palette_cache(&[], &[]).is_empty());
    // Duplicates within the tail drain also collapse.
    assert_eq!(
      merge_palette_cache(&[1, 2, 3], &[3, 4, 4]).as_slice(),
      &[1, 2, 3, 4]
    );
    // Full 8+8 disjoint keeps all 16.
    let a: Vec<u16> = (0..8).map(|i| 2 * i).collect();
    let l: Vec<u16> = (0..8).map(|i| 2 * i + 1).collect();
    let m = merge_palette_cache(&a, &l);
    assert_eq!(m.len(), 16);
    assert!(m.windows(2).all(|w| w[0] < w[1]));
  }

  #[test]
  fn index_color_cache_splits() {
    let cache = [10u16, 20, 30, 40];
    let colors = [20u16, 25, 40];
    let (flags, rest) = index_color_cache(&cache, &colors);
    assert_eq!(flags.as_slice(), &[false, true, false, true]);
    assert_eq!(rest.as_slice(), &[25]);
    // All colors in cache.
    let (flags, rest) = index_color_cache(&cache, &[10u16, 30]);
    assert_eq!(flags.as_slice(), &[true, false, true, false]);
    assert!(rest.is_empty());
    // Empty cache.
    let (flags, rest) = index_color_cache(&[], &[7u16, 8]);
    assert!(flags.is_empty());
    assert_eq!(rest.as_slice(), &[7, 8]);
  }

  #[test]
  fn ceil_log2_matches_aom() {
    assert_eq!(ceil_log2(0), 0);
    assert_eq!(ceil_log2(1), 0);
    assert_eq!(ceil_log2(2), 1);
    assert_eq!(ceil_log2(3), 2);
    assert_eq!(ceil_log2(4), 2);
    assert_eq!(ceil_log2(5), 3);
    assert_eq!(ceil_log2(8), 3);
    assert_eq!(ceil_log2(9), 4);
    assert_eq!(ceil_log2(255), 8);
    assert_eq!(ceil_log2(256), 8);
    assert_eq!(ceil_log2(257), 9);
  }

  #[test]
  fn candidates_two_color_block() {
    let mut hist = vec![0u32; 256];
    // A synthetic 8x8 two-value block.
    let mut data = vec![10i16; 32];
    data.extend(vec![200i16; 32]);
    let cands = palette_candidates_y(&data, 8, &[], &mut hist);
    assert!(!cands.is_empty());
    // The two-color candidate must be exactly the two values.
    assert!(cands.iter().any(|c| c.as_slice() == [10u16, 200]));
    // Map assignment is exact for candidate [10, 200].
    let map = build_index_map(&data, &[10, 200]);
    assert!(map[..32].iter().all(|&i| i == 0));
    assert!(map[32..].iter().all(|&i| i == 1));
  }

  #[test]
  fn candidates_gate_rejects_photo_like() {
    let mut hist = vec![0u32; 256];
    // 65+ distinct values -> not palettizable.
    let data: Vec<i16> = (0..128).map(|i| (i * 2) as i16).collect();
    assert!(palette_candidates_y(&data, 8, &[], &mut hist).is_empty());
    // Single color -> nothing to do.
    let flat = vec![42i16; 64];
    assert!(palette_candidates_y(&flat, 8, &[], &mut hist).is_empty());
  }

  #[test]
  fn candidates_snap_to_cache() {
    let mut hist = vec![0u32; 256];
    let mut data = vec![50i16; 32];
    data.extend(vec![150i16; 32]);
    // Cache color 52 is within the snap threshold (4) of 50.
    let cands = palette_candidates_y(&data, 8, &[52, 200], &mut hist);
    assert!(cands.iter().any(|c| c.as_slice() == [52u16, 150]));
  }

  #[test]
  fn kmeans_converges_on_clusters() {
    // Three tight clusters; k-means with k=3 should land near them.
    let mut data = Vec::new();
    for &(center, n) in &[(20i16, 40usize), (128, 40), (240, 40)] {
      for i in 0..n {
        data.push(center + (i % 3) as i16 - 1);
      }
    }
    let mut hist = vec![0u32; 256];
    let cands = palette_candidates_y(&data, 8, &[], &mut hist);
    let three: Vec<_> = cands.iter().filter(|c| c.len() == 3).collect();
    assert!(!three.is_empty());
    assert!(three.iter().any(|c| {
      (c[0] as i32 - 20).abs() <= 2
        && (c[1] as i32 - 128).abs() <= 2
        && (c[2] as i32 - 240).abs() <= 2
    }));
  }
}
