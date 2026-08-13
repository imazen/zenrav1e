// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! AV1 intra block copy (intraBC): encoder-side displacement-vector rules
//! and search for screen content on intra frames.
//!
//! Syntax/semantics references (traced 2026-07-03):
//! - rav1d-safe `decode.rs` (the `allow_intrabc` branch of `decode_b`: the
//!   `intrabc` flag, DV prediction from the ref-MV stack with the two
//!   default fallbacks, `read_mv_residual(.., -1)` fullpel coding, and the
//!   decoder-side DV clip) — the read-side dual of the write path.
//! - libaom `av1/common/mvref_common.h` `av1_is_dv_valid` at the pinned rev
//!   632172a4 — the bitstream-conformance validity rule (stricter than the
//!   decoder clip: 256-px delay + wavefront constraint). The encoder only
//!   ever emits DVs that pass it, so the decoder clip is a no-op.
//!
//! Scope: intra (key) frames, 64x64 superblocks, fullpel DVs; when chroma is
//! subsampled the search additionally restricts DVs to chroma-fullpel
//! alignment (even luma offsets on the subsampled axes) so every plane is a
//! pure block copy — subpel (bilinear) chroma DV prediction is a conformant
//! encoder-side omission, not implemented.

use crate::context::{CandidateMV, MI_SIZE_LOG2, TileBlockOffset};
use crate::mc::MotionVector;
use crate::partition::BlockSize;
use crate::tiling::TileStateMut;
use crate::util::{CastFromPrimitive, Pixel};

/// The AV1 intraBC reference delay: the referenced area must trail the
/// current position by at least this many pixels of 64-px superblock
/// columns (libaom `INTRABC_DELAY_PIXELS`).
pub const INTRABC_DELAY_PIXELS: usize = 256;
const INTRABC_DELAY_SB64: i32 = (INTRABC_DELAY_PIXELS / 64) as i32;

/// DV prediction for an intraBC block, mirroring rav1d `decode_b`'s
/// selection exactly: the first nonzero of the two top ref-MV stack
/// entries, else a position-dependent default (one superblock up, or one
/// superblock plus the delay to the left when still in the first
/// superblock row of the tile). All units are 1/8 pel.
pub fn dv_prediction(
  mv_stack: &[CandidateMV], tile_bo: TileBlockOffset, sb128: bool,
) -> MotionVector {
  let zero = MotionVector::default();
  if let Some(c) = mv_stack.first()
    && c.this_mv != zero
  {
    return c.this_mv;
  }
  if let Some(c) = mv_stack.get(1)
    && c.this_mv != zero
  {
    return c.this_mv;
  }
  let sb128 = sb128 as i16;
  // rav1d: `if t.b.y - (16 << sb128) < ts.tiling.row_start` — tile-relative
  // block y within the first superblock row.
  if (tile_bo.0.y as i16) < (16 << sb128) {
    MotionVector { row: 0, col: -(512 << sb128) - 2048 }
  } else {
    MotionVector { row: -(512 << sb128), col: 0 }
  }
}

/// Bitstream-conformance DV validity (port of libaom `av1_is_dv_valid`,
/// keyed to a single-tile 64-px-superblock intra frame): fullpel, source
/// block fully inside the tile, sub-8px chroma-pairing border respected,
/// and the referenced area at least `INTRABC_DELAY_SB64` 64-px superblock
/// units behind the current one along the wavefront. `mi_rows`/`mi_cols`
/// bound the tile in 4-px block units.
///
/// `dv` is in 1/8 pel; `(mi_row, mi_col)` locate the block, tile-relative.
pub fn is_dv_valid(
  dv: MotionVector, bsize: BlockSize, mi_row: usize, mi_col: usize,
  mi_rows: usize, mi_cols: usize, is_chroma_ref: bool, xdec: usize,
  ydec: usize, sb128: bool,
) -> bool {
  let bw = bsize.width() as i32;
  let bh = bsize.height() as i32;
  const SCALE_PX_TO_MV: i32 = 8;
  let (row, col) = (i32::from(dv.row), i32::from(dv.col));
  // Fullpel only.
  if (row & (SCALE_PX_TO_MV - 1)) != 0 || (col & (SCALE_PX_TO_MV - 1)) != 0 {
    return false;
  }
  const MI_SIZE: i32 = 1 << MI_SIZE_LOG2;
  let (mi_row, mi_col) = (mi_row as i32, mi_col as i32);
  // Tile bounds (single-tile: 0..mi_rows/mi_cols).
  let src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + row;
  let tile_top_edge = 0;
  if src_top_edge < tile_top_edge {
    return false;
  }
  let src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + col;
  let tile_left_edge = 0;
  if src_left_edge < tile_left_edge {
    return false;
  }
  let src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + row;
  let tile_bottom_edge = mi_rows as i32 * MI_SIZE * SCALE_PX_TO_MV;
  if src_bottom_edge > tile_bottom_edge {
    return false;
  }
  let src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + col;
  let tile_right_edge = mi_cols as i32 * MI_SIZE * SCALE_PX_TO_MV;
  if src_right_edge > tile_right_edge {
    return false;
  }

  // Sub-8px chroma pairing: the merged chroma block reaches 4 px further
  // up/left than the luma block, so the source must keep that margin.
  if is_chroma_ref {
    if bw < 8
      && xdec != 0
      && src_left_edge < tile_left_edge + 4 * SCALE_PX_TO_MV
    {
      return false;
    }
    if bh < 8 && ydec != 0 && src_top_edge < tile_top_edge + 4 * SCALE_PX_TO_MV
    {
      return false;
    }
  }

  // Already-coded-superblock + hardware-decoder delay constraints.
  let mib_size_log2 = if sb128 { 5 } else { 4 };
  let sb_size = (1 << mib_size_log2) * MI_SIZE;
  let active_sb_row = mi_row >> mib_size_log2;
  let active_sb64_col = (mi_col * MI_SIZE) >> 6;
  let src_sb_row = ((src_bottom_edge >> 3) - 1) / sb_size;
  let src_sb64_col = ((src_right_edge >> 3) - 1) >> 6;
  let total_sb64_per_row = ((mi_cols as i32 - 1) >> 4) + 1;
  let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
  let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
  if src_sb64 >= active_sb64 - INTRABC_DELAY_SB64 {
    return false;
  }

  // Wavefront constraint: only the top-left gradient region is legal.
  let gradient = 1 + INTRABC_DELAY_SB64 + (sb_size > 64) as i32;
  let wf_offset = gradient * (active_sb_row - src_sb_row);
  if src_sb_row > active_sb_row
    || src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset
  {
    return false;
  }
  true
}

/// Predicts every plane of an intraBC block into the reconstruction by
/// fullpel copy from the same tile's already-reconstructed area at `dv`
/// (1/8 pel, luma units) — the write-side dual of rav1d's intraBC `mc`:
/// with the search restricted to plane-fullpel DVs the bilinear filter
/// reduces to a copy. DV validity guarantees the source region is fully
/// inside the already-reconstructed part of the tile. The residual path
/// after this is the normal inter coding path.
pub fn intrabc_compensate<T: Pixel>(
  ts: &mut TileStateMut<'_, T>, tile_bo: TileBlockOffset, bsize: BlockSize,
  dv: MotionVector, is_chroma_block: bool,
) {
  let planes = if is_chroma_block { 3 } else { 1 };
  let dv_px_x = isize::from(dv.col) >> 3;
  let dv_px_y = isize::from(dv.row) >> 3;
  for p in 0..planes {
    let rec = &mut ts.rec.planes[p];
    let (xdec, ydec) = (rec.plane_cfg.xdec, rec.plane_cfg.ydec);
    debug_assert!(
      dv_px_x & ((1 << xdec) - 1) == 0 && dv_px_y & ((1 << ydec) - 1) == 0,
      "intraBC DVs must be plane-fullpel (search restriction)"
    );
    let dst_x = (tile_bo.0.x >> xdec) << MI_SIZE_LOG2;
    let dst_y = (tile_bo.0.y >> ydec) << MI_SIZE_LOG2;
    let plane_bsize = if p == 0 {
      bsize
    } else {
      bsize.subsampled_size(xdec, ydec).expect("subsampable block size")
    };
    let w = plane_bsize.width();
    let h = plane_bsize.height();
    let src_x = (dst_x as isize + (dv_px_x >> xdec)) as usize;
    let src_y = (dst_y as isize + (dv_px_y >> ydec)) as usize;
    // Source and destination live in the same plane region; buffer one row
    // at a time (blocks are <= 64 px, the copy cost is trivial next to
    // RDO). The DV validity rules exclude any overlap of source and
    // destination, so row order does not matter.
    let mut row_buf = vec![T::cast_from(0u8); w];
    for y in 0..h {
      row_buf.copy_from_slice(&rec[src_y + y][src_x..src_x + w]);
      rec[dst_y + y][dst_x..dst_x + w].copy_from_slice(&row_buf);
    }
  }
}

/// Luma SAD between the source block and the reconstruction at a DV offset
/// (both tile-relative), used to rank candidate DVs before the full-rate RD
/// trials.
pub fn sad_at_dv<T: Pixel>(
  ts: &TileStateMut<'_, T>, tile_bo: TileBlockOffset, bsize: BlockSize,
  dv: MotionVector,
) -> u64 {
  use crate::tiling::Area;
  let src =
    ts.input_tile.planes[0].subregion(Area::BlockStartingAt { bo: tile_bo.0 });
  let rec = &ts.rec.planes[0];
  let base_x = tile_bo.0.x << MI_SIZE_LOG2;
  let base_y = tile_bo.0.y << MI_SIZE_LOG2;
  let sx = (base_x as isize + (isize::from(dv.col) >> 3)) as usize;
  let sy = (base_y as isize + (isize::from(dv.row) >> 3)) as usize;
  let (w, h) = (bsize.width(), bsize.height());
  let mut sad = 0u64;
  for y in 0..h {
    let src_row = &src[y];
    let rec_row = &rec[sy + y][sx..sx + w];
    for x in 0..w {
      let a = i32::cast_from(src_row[x]);
      let b = i32::cast_from(rec_row[x]);
      sad += (a - b).unsigned_abs() as u64;
    }
  }
  sad
}

#[cfg(test)]
mod test {
  use super::*;

  fn mv(row: i16, col: i16) -> MotionVector {
    MotionVector { row, col }
  }

  #[test]
  fn dv_default_prediction_matches_rav1d() {
    // Empty stack, first SB row of the tile: one SB + delay to the left.
    let bo = TileBlockOffset(crate::context::BlockOffset { x: 40, y: 4 });
    assert_eq!(dv_prediction(&[], bo, false), mv(0, -512 - 2048));
    // Below the first SB row: one SB up.
    let bo = TileBlockOffset(crate::context::BlockOffset { x: 40, y: 16 });
    assert_eq!(dv_prediction(&[], bo, false), mv(-512, 0));
  }

  #[test]
  fn dv_validity_basics() {
    let bs = BlockSize::BLOCK_16X16;
    // 1024x1024 frame = 256x256 mi, single tile.
    let (mi_rows, mi_cols) = (256, 256);
    // Subpel DV invalid.
    assert!(!is_dv_valid(
      mv(-4, 0),
      bs,
      32,
      32,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
    // Out of tile (above).
    assert!(!is_dv_valid(
      mv(-8 * 1000, 0),
      bs,
      32,
      32,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
    // The block directly above, one SB row up but same SB64 column: legal
    // only when far enough behind the wavefront. (mi_row 32 = SB row 2 for
    // 64px SBs; source bottom lands in SB row 1; gradient = 5 =>
    // src_sb64_col (8) < 8 - 4 + 5 = 9: valid.)
    assert!(is_dv_valid(
      mv(-8 * 64, 0),
      bs,
      32,
      32,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
    // Immediately left of the block, same SB: invalid (delay).
    assert!(!is_dv_valid(
      mv(0, -8 * 16),
      bs,
      32,
      32,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
    // Far left on the same SB row: still invalid (256px delay covers the
    // whole row start for the first SB rows).
    assert!(!is_dv_valid(
      mv(0, -8 * 128),
      bs,
      32,
      4,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
  }

  #[test]
  fn dv_validity_first_row_left_region() {
    // On the first SB row, only sources >= 256+ px to the left are legal.
    let bs = BlockSize::BLOCK_16X16;
    let (mi_rows, mi_cols) = (256, 256);
    // mi_col 96 = px 384 = SB64 col 6; delay 4 => src_sb64_col must be
    // <= 6 - 4 - 1 = 1 (wf_offset 0 on the same SB row).
    assert!(is_dv_valid(
      mv(0, -8 * 320),
      bs,
      0,
      96,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
    assert!(!is_dv_valid(
      mv(0, -8 * 240),
      bs,
      0,
      96,
      mi_rows,
      mi_cols,
      true,
      1,
      1,
      false
    ));
  }
}
