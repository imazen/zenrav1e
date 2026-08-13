// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use super::*;
use crate::predict::{FilterIntraMode, PaletteColor, PaletteSize};
use std::marker::PhantomData;

pub const CDF_LEN_MAX: usize = 16;

#[derive(Clone)]
pub struct CDFContextCheckpoint {
  small: usize,
  large: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CDFContext {
  pub comp_bwd_ref_cdf: [[[u16; 2]; BWD_REFS - 1]; REF_CONTEXTS],
  pub comp_mode_cdf: [[u16; 2]; COMP_INTER_CONTEXTS],
  pub comp_ref_cdf: [[[u16; 2]; FWD_REFS - 1]; REF_CONTEXTS],
  pub comp_ref_type_cdf: [[u16; 2]; COMP_REF_TYPE_CONTEXTS],
  pub dc_sign_cdf: [[[u16; 2]; DC_SIGN_CONTEXTS]; PLANE_TYPES],
  pub drl_cdfs: [[u16; 2]; DRL_MODE_CONTEXTS],
  pub eob_extra_cdf:
    [[[[u16; 2]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub filter_intra_cdfs: [[u16; 2]; BlockSize::BLOCK_SIZES_ALL],
  pub filter_intra_mode_cdf:
    [u16; FilterIntraMode::FILTER_INTRA_MODES as usize],
  pub intra_inter_cdfs: [[u16; 2]; INTRA_INTER_CONTEXTS],
  pub intrabc_cdf: [u16; 2],
  pub lrf_sgrproj_cdf: [u16; 2],
  pub lrf_wiener_cdf: [u16; 2],
  pub newmv_cdf: [[u16; 2]; NEWMV_MODE_CONTEXTS],
  pub palette_uv_mode_cdfs: [[u16; 2]; PALETTE_UV_MODE_CONTEXTS],
  pub palette_y_mode_cdfs:
    [[[u16; 2]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BSIZE_CTXS],
  pub refmv_cdf: [[u16; 2]; REFMV_MODE_CONTEXTS],
  pub single_ref_cdfs: [[[u16; 2]; SINGLE_REFS - 1]; REF_CONTEXTS],
  pub skip_cdfs: [[u16; 2]; SKIP_CONTEXTS],
  pub txb_skip_cdf: [[[u16; 2]; TXB_SKIP_CONTEXTS]; TxSize::TX_SIZES],
  pub txfm_partition_cdf: [[u16; 2]; TXFM_PARTITION_CONTEXTS],
  pub zeromv_cdf: [[u16; 2]; GLOBALMV_MODE_CONTEXTS],
  pub tx_size_8x8_cdf: [[u16; MAX_TX_DEPTH]; TX_SIZE_CONTEXTS],
  pub inter_tx_3_cdf: [[u16; 2]; TX_SIZE_SQR_CONTEXTS],

  pub coeff_base_eob_cdf:
    [[[[u16; 3]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub lrf_switchable_cdf: [u16; 3],
  pub tx_size_cdf: [[[u16; MAX_TX_DEPTH + 1]; TX_SIZE_CONTEXTS]; BIG_TX_CATS],

  pub coeff_base_cdf:
    [[[[u16; 4]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub coeff_br_cdf:
    [[[[u16; BR_CDF_SIZE]; LEVEL_CONTEXTS]; PLANE_TYPES]; TxSize::TX_SIZES],
  pub deblock_delta_cdf: [u16; DELTA_LF_PROBS + 1],
  pub deblock_delta_multi_cdf: [[u16; DELTA_LF_PROBS + 1]; FRAME_LF_COUNT],
  pub delta_q_cdf: [u16; DELTA_Q_PROBS + 1],
  pub partition_w8_cdf: [[u16; 4]; PARTITION_TYPES],

  pub eob_flag_cdf16: [[[u16; 5]; 2]; PLANE_TYPES],
  pub intra_tx_2_cdf: [[[u16; 5]; INTRA_MODES]; TX_SIZE_SQR_CONTEXTS],

  pub eob_flag_cdf32: [[[u16; 6]; 2]; PLANE_TYPES],

  pub angle_delta_cdf: [[u16; 2 * MAX_ANGLE_DELTA + 1]; DIRECTIONAL_MODES],
  pub eob_flag_cdf64: [[[u16; 7]; 2]; PLANE_TYPES],
  pub intra_tx_1_cdf: [[[u16; 7]; INTRA_MODES]; TX_SIZE_SQR_CONTEXTS],
  pub palette_y_size_cdf:
    [[u16; PaletteSize::PALETTE_SIZES as usize]; PALETTE_BSIZE_CTXS],
  pub palette_uv_size_cdf:
    [[u16; PaletteSize::PALETTE_SIZES as usize]; PALETTE_BSIZE_CTXS],

  // Color-index CDFs: the used alphabet is the palette size (2..=8), i.e.
  // only the first `size` entries of each row (count slot at `size - 1`);
  // the writers take fixed-size sub-array views per palette size.
  pub palette_y_color_index_cdf: [[[u16; PaletteColor::PALETTE_COLORS as usize];
    PALETTE_COLOR_INDEX_CONTEXTS];
    PaletteSize::PALETTE_SIZES as usize],
  pub palette_uv_color_index_cdf:
    [[[u16; PaletteColor::PALETTE_COLORS as usize];
      PALETTE_COLOR_INDEX_CONTEXTS]; PaletteSize::PALETTE_SIZES as usize],

  pub cfl_sign_cdf: [u16; CFL_JOINT_SIGNS],
  pub compound_mode_cdf: [[u16; INTER_COMPOUND_MODES]; INTER_MODE_CONTEXTS],
  pub eob_flag_cdf128: [[[u16; 8]; 2]; PLANE_TYPES],
  pub spatial_segmentation_cdfs: [[u16; 8]; 3],
  pub partition_w128_cdf: [[u16; 8]; PARTITION_TYPES],

  pub eob_flag_cdf256: [[[u16; 9]; 2]; PLANE_TYPES],

  pub eob_flag_cdf512: [[[u16; 10]; 2]; PLANE_TYPES],
  pub partition_cdf: [[u16; EXT_PARTITION_TYPES]; 3 * PARTITION_TYPES],

  pub eob_flag_cdf1024: [[[u16; 11]; 2]; PLANE_TYPES],

  pub inter_tx_2_cdf: [[u16; 12]; TX_SIZE_SQR_CONTEXTS],

  pub kf_y_cdf: [[[u16; INTRA_MODES]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
  pub y_mode_cdf: [[u16; INTRA_MODES]; BLOCK_SIZE_GROUPS],
  pub uv_mode_cdf: [[u16; INTRA_MODES]; INTRA_MODES],

  pub uv_mode_cfl_cdf: [[u16; UV_INTRA_MODES]; INTRA_MODES],

  pub cfl_alpha_cdf: [[u16; CFL_ALPHABET_SIZE]; CFL_ALPHA_CONTEXTS],
  pub inter_tx_1_cdf: [[u16; TX_TYPES]; TX_SIZE_SQR_CONTEXTS],

  pub nmv_context: NMVContext,
}

pub struct CDFOffset<const CDF_LEN: usize> {
  offset: usize,
  phantom: PhantomData<[u16; CDF_LEN]>,
}

impl CDFContext {
  pub fn new(quantizer: u8) -> CDFContext {
    let qctx = match quantizer {
      0..=20 => 0,
      21..=60 => 1,
      61..=120 => 2,
      _ => 3,
    };
    CDFContext {
      partition_w8_cdf: default_partition_w8_cdf,
      partition_w128_cdf: default_partition_w128_cdf,
      partition_cdf: default_partition_cdf,
      kf_y_cdf: default_kf_y_mode_cdf,
      y_mode_cdf: default_if_y_mode_cdf,
      uv_mode_cdf: default_uv_mode_cdf,
      uv_mode_cfl_cdf: default_uv_mode_cfl_cdf,
      cfl_sign_cdf: default_cfl_sign_cdf,
      cfl_alpha_cdf: default_cfl_alpha_cdf,
      newmv_cdf: default_newmv_cdf,
      zeromv_cdf: default_zeromv_cdf,
      refmv_cdf: default_refmv_cdf,
      intra_tx_2_cdf: default_intra_tx_2_cdf,
      intra_tx_1_cdf: default_intra_tx_1_cdf,
      inter_tx_3_cdf: default_inter_tx_3_cdf,
      inter_tx_2_cdf: default_inter_tx_2_cdf,
      inter_tx_1_cdf: default_inter_tx_1_cdf,
      tx_size_8x8_cdf: default_tx_size_8x8_cdf,
      tx_size_cdf: default_tx_size_cdf,
      txfm_partition_cdf: default_txfm_partition_cdf,
      skip_cdfs: default_skip_cdfs,
      intra_inter_cdfs: default_intra_inter_cdf,
      intrabc_cdf: default_intrabc_cdf,
      angle_delta_cdf: default_angle_delta_cdf,
      filter_intra_cdfs: default_filter_intra_cdfs,
      filter_intra_mode_cdf: default_filter_intra_mode_cdf,
      palette_y_mode_cdfs: default_palette_y_mode_cdfs,
      palette_uv_mode_cdfs: default_palette_uv_mode_cdfs,
      palette_y_size_cdf: default_palette_y_size_cdf,
      palette_uv_size_cdf: default_palette_uv_size_cdf,
      palette_y_color_index_cdf: default_palette_y_color_index_cdf,
      palette_uv_color_index_cdf: default_palette_uv_color_index_cdf,
      comp_mode_cdf: default_comp_mode_cdf,
      comp_ref_type_cdf: default_comp_ref_type_cdf,
      comp_ref_cdf: default_comp_ref_cdf,
      comp_bwd_ref_cdf: default_comp_bwdref_cdf,
      single_ref_cdfs: default_single_ref_cdf,
      drl_cdfs: default_drl_cdf,
      compound_mode_cdf: default_compound_mode_cdf,
      nmv_context: default_nmv_context,
      deblock_delta_multi_cdf: default_delta_lf_multi_cdf,
      deblock_delta_cdf: default_delta_lf_cdf,
      delta_q_cdf: default_delta_q_cdf,
      spatial_segmentation_cdfs: default_spatial_pred_seg_tree_cdf,
      lrf_switchable_cdf: default_switchable_restore_cdf,
      lrf_sgrproj_cdf: default_sgrproj_restore_cdf,
      lrf_wiener_cdf: default_wiener_restore_cdf,

      // lv_map
      txb_skip_cdf: av1_default_txb_skip_cdfs[qctx],
      dc_sign_cdf: av1_default_dc_sign_cdfs[qctx],
      eob_extra_cdf: av1_default_eob_extra_cdfs[qctx],

      eob_flag_cdf16: av1_default_eob_multi16_cdfs[qctx],
      eob_flag_cdf32: av1_default_eob_multi32_cdfs[qctx],
      eob_flag_cdf64: av1_default_eob_multi64_cdfs[qctx],
      eob_flag_cdf128: av1_default_eob_multi128_cdfs[qctx],
      eob_flag_cdf256: av1_default_eob_multi256_cdfs[qctx],
      eob_flag_cdf512: av1_default_eob_multi512_cdfs[qctx],
      eob_flag_cdf1024: av1_default_eob_multi1024_cdfs[qctx],

      coeff_base_eob_cdf: av1_default_coeff_base_eob_multi_cdfs[qctx],
      coeff_base_cdf: av1_default_coeff_base_multi_cdfs[qctx],
      coeff_br_cdf: av1_default_coeff_lps_multi_cdfs[qctx],
    }
  }

  pub fn reset_counts(&mut self) {
    macro_rules! reset_1d {
      ($field:expr) => {
        let r = $field.last_mut().unwrap();
        *r = 0;
      };
    }
    macro_rules! reset_2d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_1d!(x);
        }
      };
    }
    macro_rules! reset_3d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_2d!(x);
        }
      };
    }
    macro_rules! reset_4d {
      ($field:expr) => {
        for x in $field.iter_mut() {
          reset_3d!(x);
        }
      };
    }

    reset_2d!(self.partition_w8_cdf);
    reset_2d!(self.partition_w128_cdf);
    reset_2d!(self.partition_cdf);

    reset_3d!(self.kf_y_cdf);
    reset_2d!(self.y_mode_cdf);

    reset_2d!(self.uv_mode_cdf);
    reset_2d!(self.uv_mode_cfl_cdf);
    reset_1d!(self.cfl_sign_cdf);
    reset_2d!(self.cfl_alpha_cdf);
    reset_2d!(self.newmv_cdf);
    reset_2d!(self.zeromv_cdf);
    reset_2d!(self.refmv_cdf);

    reset_3d!(self.intra_tx_2_cdf);
    reset_3d!(self.intra_tx_1_cdf);

    reset_2d!(self.inter_tx_3_cdf);
    reset_2d!(self.inter_tx_2_cdf);
    reset_2d!(self.inter_tx_1_cdf);

    reset_2d!(self.tx_size_8x8_cdf);
    reset_3d!(self.tx_size_cdf);

    reset_2d!(self.txfm_partition_cdf);

    reset_2d!(self.skip_cdfs);
    reset_2d!(self.intra_inter_cdfs);
    reset_1d!(self.intrabc_cdf);
    reset_2d!(self.angle_delta_cdf);
    reset_2d!(self.filter_intra_cdfs);
    reset_1d!(self.filter_intra_mode_cdf);
    reset_3d!(self.palette_y_mode_cdfs);
    reset_2d!(self.palette_uv_mode_cdfs);
    reset_2d!(self.palette_y_size_cdf);
    reset_2d!(self.palette_uv_size_cdf);
    // The color-index CDFs use a per-palette-size partial alphabet: for
    // palette size index `i` (palette size `i + 2`) the adaptation count
    // lives at `[i + 1]`, not at the row's last slot. Mirrors rav1d's
    // frame-end `update_cdf_2d!(5, k + 1, color_map[l][k])`.
    for (i, per_size) in self.palette_y_color_index_cdf.iter_mut().enumerate()
    {
      for cdf in per_size.iter_mut() {
        cdf[i + 1] = 0;
      }
    }
    for (i, per_size) in self.palette_uv_color_index_cdf.iter_mut().enumerate()
    {
      for cdf in per_size.iter_mut() {
        cdf[i + 1] = 0;
      }
    }
    reset_2d!(self.comp_mode_cdf);
    reset_2d!(self.comp_ref_type_cdf);
    reset_3d!(self.comp_ref_cdf);
    reset_3d!(self.comp_bwd_ref_cdf);
    reset_3d!(self.single_ref_cdfs);
    reset_2d!(self.drl_cdfs);
    reset_2d!(self.compound_mode_cdf);
    reset_2d!(self.deblock_delta_multi_cdf);
    reset_1d!(self.deblock_delta_cdf);
    reset_1d!(self.delta_q_cdf);
    reset_2d!(self.spatial_segmentation_cdfs);
    reset_1d!(self.lrf_switchable_cdf);
    reset_1d!(self.lrf_sgrproj_cdf);
    reset_1d!(self.lrf_wiener_cdf);

    reset_1d!(self.nmv_context.joints_cdf);
    for i in 0..2 {
      reset_1d!(self.nmv_context.comps[i].classes_cdf);
      reset_2d!(self.nmv_context.comps[i].class0_fp_cdf);
      reset_1d!(self.nmv_context.comps[i].fp_cdf);
      reset_1d!(self.nmv_context.comps[i].sign_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_hp_cdf);
      reset_1d!(self.nmv_context.comps[i].hp_cdf);
      reset_1d!(self.nmv_context.comps[i].class0_cdf);
      reset_2d!(self.nmv_context.comps[i].bits_cdf);
    }

    // lv_map
    reset_3d!(self.txb_skip_cdf);
    reset_3d!(self.dc_sign_cdf);
    reset_4d!(self.eob_extra_cdf);

    reset_3d!(self.eob_flag_cdf16);
    reset_3d!(self.eob_flag_cdf32);
    reset_3d!(self.eob_flag_cdf64);
    reset_3d!(self.eob_flag_cdf128);
    reset_3d!(self.eob_flag_cdf256);
    reset_3d!(self.eob_flag_cdf512);
    reset_3d!(self.eob_flag_cdf1024);

    reset_4d!(self.coeff_base_eob_cdf);
    reset_4d!(self.coeff_base_cdf);
    reset_4d!(self.coeff_br_cdf);
  }

  /// # Panics
  ///
  /// - If any of the CDF arrays are uninitialized.
  ///   This should never happen and indicates a development error.
  pub fn build_map(&self) -> Vec<(&'static str, usize, usize)> {
    use std::mem::size_of_val;

    let partition_w8_cdf_start =
      self.partition_w8_cdf.first().unwrap().as_ptr() as usize;
    let partition_w8_cdf_end =
      partition_w8_cdf_start + size_of_val(&self.partition_w8_cdf);
    let partition_w128_cdf_start =
      self.partition_w128_cdf.first().unwrap().as_ptr() as usize;
    let partition_w128_cdf_end =
      partition_w128_cdf_start + size_of_val(&self.partition_w128_cdf);
    let partition_cdf_start =
      self.partition_cdf.first().unwrap().as_ptr() as usize;
    let partition_cdf_end =
      partition_cdf_start + size_of_val(&self.partition_cdf);
    let kf_y_cdf_start = self.kf_y_cdf.first().unwrap().as_ptr() as usize;
    let kf_y_cdf_end = kf_y_cdf_start + size_of_val(&self.kf_y_cdf);
    let y_mode_cdf_start = self.y_mode_cdf.first().unwrap().as_ptr() as usize;
    let y_mode_cdf_end = y_mode_cdf_start + size_of_val(&self.y_mode_cdf);
    let uv_mode_cdf_start =
      self.uv_mode_cdf.first().unwrap().as_ptr() as usize;
    let uv_mode_cdf_end = uv_mode_cdf_start + size_of_val(&self.uv_mode_cdf);
    let uv_mode_cfl_cdf_start =
      self.uv_mode_cfl_cdf.first().unwrap().as_ptr() as usize;
    let uv_mode_cfl_cdf_end =
      uv_mode_cfl_cdf_start + size_of_val(&self.uv_mode_cfl_cdf);
    let cfl_sign_cdf_start = self.cfl_sign_cdf.as_ptr() as usize;
    let cfl_sign_cdf_end =
      cfl_sign_cdf_start + size_of_val(&self.cfl_sign_cdf);
    let cfl_alpha_cdf_start =
      self.cfl_alpha_cdf.first().unwrap().as_ptr() as usize;
    let cfl_alpha_cdf_end =
      cfl_alpha_cdf_start + size_of_val(&self.cfl_alpha_cdf);
    let newmv_cdf_start = self.newmv_cdf.first().unwrap().as_ptr() as usize;
    let newmv_cdf_end = newmv_cdf_start + size_of_val(&self.newmv_cdf);
    let zeromv_cdf_start = self.zeromv_cdf.first().unwrap().as_ptr() as usize;
    let zeromv_cdf_end = zeromv_cdf_start + size_of_val(&self.zeromv_cdf);
    let refmv_cdf_start = self.refmv_cdf.first().unwrap().as_ptr() as usize;
    let refmv_cdf_end = refmv_cdf_start + size_of_val(&self.refmv_cdf);
    let intra_tx_2_cdf_start =
      self.intra_tx_2_cdf.first().unwrap().as_ptr() as usize;
    let intra_tx_2_cdf_end =
      intra_tx_2_cdf_start + size_of_val(&self.intra_tx_2_cdf);
    let intra_tx_1_cdf_start =
      self.intra_tx_1_cdf.first().unwrap().as_ptr() as usize;
    let intra_tx_1_cdf_end =
      intra_tx_1_cdf_start + size_of_val(&self.intra_tx_1_cdf);
    let inter_tx_3_cdf_start =
      self.inter_tx_3_cdf.first().unwrap().as_ptr() as usize;
    let inter_tx_3_cdf_end =
      inter_tx_3_cdf_start + size_of_val(&self.inter_tx_3_cdf);
    let inter_tx_2_cdf_start =
      self.inter_tx_2_cdf.first().unwrap().as_ptr() as usize;
    let inter_tx_2_cdf_end =
      inter_tx_2_cdf_start + size_of_val(&self.inter_tx_2_cdf);
    let inter_tx_1_cdf_start =
      self.inter_tx_1_cdf.first().unwrap().as_ptr() as usize;
    let inter_tx_1_cdf_end =
      inter_tx_1_cdf_start + size_of_val(&self.inter_tx_1_cdf);
    let tx_size_8x8_cdf_start =
      self.tx_size_8x8_cdf.first().unwrap().as_ptr() as usize;
    let tx_size_8x8_cdf_end =
      tx_size_8x8_cdf_start + size_of_val(&self.tx_size_8x8_cdf);
    let tx_size_cdf_start =
      self.tx_size_cdf.first().unwrap().as_ptr() as usize;
    let tx_size_cdf_end = tx_size_cdf_start + size_of_val(&self.tx_size_cdf);
    let txfm_partition_cdf_start =
      self.txfm_partition_cdf.first().unwrap().as_ptr() as usize;
    let txfm_partition_cdf_end =
      txfm_partition_cdf_start + size_of_val(&self.txfm_partition_cdf);
    let skip_cdfs_start = self.skip_cdfs.first().unwrap().as_ptr() as usize;
    let skip_cdfs_end = skip_cdfs_start + size_of_val(&self.skip_cdfs);
    let intrabc_cdf_start = self.intrabc_cdf.as_ptr() as usize;
    let intrabc_cdf_end = intrabc_cdf_start + size_of_val(&self.intrabc_cdf);
    let intra_inter_cdfs_start =
      self.intra_inter_cdfs.first().unwrap().as_ptr() as usize;
    let intra_inter_cdfs_end =
      intra_inter_cdfs_start + size_of_val(&self.intra_inter_cdfs);
    let angle_delta_cdf_start =
      self.angle_delta_cdf.first().unwrap().as_ptr() as usize;
    let angle_delta_cdf_end =
      angle_delta_cdf_start + size_of_val(&self.angle_delta_cdf);
    let filter_intra_cdfs_start =
      self.filter_intra_cdfs.first().unwrap().as_ptr() as usize;
    let filter_intra_cdfs_end =
      filter_intra_cdfs_start + size_of_val(&self.filter_intra_cdfs);
    let palette_y_mode_cdfs_start =
      self.palette_y_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_y_mode_cdfs_end =
      palette_y_mode_cdfs_start + size_of_val(&self.palette_y_mode_cdfs);
    let palette_uv_mode_cdfs_start =
      self.palette_uv_mode_cdfs.first().unwrap().as_ptr() as usize;
    let palette_uv_mode_cdfs_end =
      palette_uv_mode_cdfs_start + size_of_val(&self.palette_uv_mode_cdfs);
    let palette_y_size_cdf_start =
      self.palette_y_size_cdf.first().unwrap().as_ptr() as usize;
    let palette_y_size_cdf_end =
      palette_y_size_cdf_start + size_of_val(&self.palette_y_size_cdf);
    let palette_uv_size_cdf_start =
      self.palette_uv_size_cdf.first().unwrap().as_ptr() as usize;
    let palette_uv_size_cdf_end =
      palette_uv_size_cdf_start + size_of_val(&self.palette_uv_size_cdf);
    let palette_y_color_index_cdf_start =
      self.palette_y_color_index_cdf.first().unwrap().as_ptr() as usize;
    let palette_y_color_index_cdf_end = palette_y_color_index_cdf_start
      + size_of_val(&self.palette_y_color_index_cdf);
    let palette_uv_color_index_cdf_start =
      self.palette_uv_color_index_cdf.first().unwrap().as_ptr() as usize;
    let palette_uv_color_index_cdf_end = palette_uv_color_index_cdf_start
      + size_of_val(&self.palette_uv_color_index_cdf);
    let comp_mode_cdf_start =
      self.comp_mode_cdf.first().unwrap().as_ptr() as usize;
    let comp_mode_cdf_end =
      comp_mode_cdf_start + size_of_val(&self.comp_mode_cdf);
    let comp_ref_type_cdf_start =
      self.comp_ref_type_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_type_cdf_end =
      comp_ref_type_cdf_start + size_of_val(&self.comp_ref_type_cdf);
    let comp_ref_cdf_start =
      self.comp_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_ref_cdf_end =
      comp_ref_cdf_start + size_of_val(&self.comp_ref_cdf);
    let comp_bwd_ref_cdf_start =
      self.comp_bwd_ref_cdf.first().unwrap().as_ptr() as usize;
    let comp_bwd_ref_cdf_end =
      comp_bwd_ref_cdf_start + size_of_val(&self.comp_bwd_ref_cdf);
    let single_ref_cdfs_start =
      self.single_ref_cdfs.first().unwrap().as_ptr() as usize;
    let single_ref_cdfs_end =
      single_ref_cdfs_start + size_of_val(&self.single_ref_cdfs);
    let drl_cdfs_start = self.drl_cdfs.first().unwrap().as_ptr() as usize;
    let drl_cdfs_end = drl_cdfs_start + size_of_val(&self.drl_cdfs);
    let compound_mode_cdf_start =
      self.compound_mode_cdf.first().unwrap().as_ptr() as usize;
    let compound_mode_cdf_end =
      compound_mode_cdf_start + size_of_val(&self.compound_mode_cdf);
    let nmv_context_start = &self.nmv_context as *const NMVContext as usize;
    let nmv_context_end = nmv_context_start + size_of_val(&self.nmv_context);
    let deblock_delta_multi_cdf_start =
      self.deblock_delta_multi_cdf.first().unwrap().as_ptr() as usize;
    let deblock_delta_multi_cdf_end = deblock_delta_multi_cdf_start
      + size_of_val(&self.deblock_delta_multi_cdf);
    let deblock_delta_cdf_start = self.deblock_delta_cdf.as_ptr() as usize;
    let deblock_delta_cdf_end =
      deblock_delta_cdf_start + size_of_val(&self.deblock_delta_cdf);
    let delta_q_cdf_start = self.delta_q_cdf.as_ptr() as usize;
    let delta_q_cdf_end = delta_q_cdf_start + size_of_val(&self.delta_q_cdf);
    let spatial_segmentation_cdfs_start =
      self.spatial_segmentation_cdfs.first().unwrap().as_ptr() as usize;
    let spatial_segmentation_cdfs_end = spatial_segmentation_cdfs_start
      + size_of_val(&self.spatial_segmentation_cdfs);
    let lrf_switchable_cdf_start = self.lrf_switchable_cdf.as_ptr() as usize;
    let lrf_switchable_cdf_end =
      lrf_switchable_cdf_start + size_of_val(&self.lrf_switchable_cdf);
    let lrf_sgrproj_cdf_start = self.lrf_sgrproj_cdf.as_ptr() as usize;
    let lrf_sgrproj_cdf_end =
      lrf_sgrproj_cdf_start + size_of_val(&self.lrf_sgrproj_cdf);
    let lrf_wiener_cdf_start = self.lrf_wiener_cdf.as_ptr() as usize;
    let lrf_wiener_cdf_end =
      lrf_wiener_cdf_start + size_of_val(&self.lrf_wiener_cdf);

    let txb_skip_cdf_start =
      self.txb_skip_cdf.first().unwrap().as_ptr() as usize;
    let txb_skip_cdf_end =
      txb_skip_cdf_start + size_of_val(&self.txb_skip_cdf);
    let dc_sign_cdf_start =
      self.dc_sign_cdf.first().unwrap().as_ptr() as usize;
    let dc_sign_cdf_end = dc_sign_cdf_start + size_of_val(&self.dc_sign_cdf);
    let eob_extra_cdf_start =
      self.eob_extra_cdf.first().unwrap().as_ptr() as usize;
    let eob_extra_cdf_end =
      eob_extra_cdf_start + size_of_val(&self.eob_extra_cdf);
    let eob_flag_cdf16_start =
      self.eob_flag_cdf16.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf16_end =
      eob_flag_cdf16_start + size_of_val(&self.eob_flag_cdf16);
    let eob_flag_cdf32_start =
      self.eob_flag_cdf32.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf32_end =
      eob_flag_cdf32_start + size_of_val(&self.eob_flag_cdf32);
    let eob_flag_cdf64_start =
      self.eob_flag_cdf64.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf64_end =
      eob_flag_cdf64_start + size_of_val(&self.eob_flag_cdf64);
    let eob_flag_cdf128_start =
      self.eob_flag_cdf128.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf128_end =
      eob_flag_cdf128_start + size_of_val(&self.eob_flag_cdf128);
    let eob_flag_cdf256_start =
      self.eob_flag_cdf256.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf256_end =
      eob_flag_cdf256_start + size_of_val(&self.eob_flag_cdf256);
    let eob_flag_cdf512_start =
      self.eob_flag_cdf512.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf512_end =
      eob_flag_cdf512_start + size_of_val(&self.eob_flag_cdf512);
    let eob_flag_cdf1024_start =
      self.eob_flag_cdf1024.first().unwrap().as_ptr() as usize;
    let eob_flag_cdf1024_end =
      eob_flag_cdf1024_start + size_of_val(&self.eob_flag_cdf1024);
    let coeff_base_eob_cdf_start =
      self.coeff_base_eob_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_eob_cdf_end =
      coeff_base_eob_cdf_start + size_of_val(&self.coeff_base_eob_cdf);
    let coeff_base_cdf_start =
      self.coeff_base_cdf.first().unwrap().as_ptr() as usize;
    let coeff_base_cdf_end =
      coeff_base_cdf_start + size_of_val(&self.coeff_base_cdf);
    let coeff_br_cdf_start =
      self.coeff_br_cdf.first().unwrap().as_ptr() as usize;
    let coeff_br_cdf_end =
      coeff_br_cdf_start + size_of_val(&self.coeff_br_cdf);

    vec![
      ("partition_w8_cdf", partition_w8_cdf_start, partition_w8_cdf_end),
      ("partition_w128_cdf", partition_w128_cdf_start, partition_w128_cdf_end),
      ("partition_cdf", partition_cdf_start, partition_cdf_end),
      ("kf_y_cdf", kf_y_cdf_start, kf_y_cdf_end),
      ("y_mode_cdf", y_mode_cdf_start, y_mode_cdf_end),
      ("uv_mode_cdf", uv_mode_cdf_start, uv_mode_cdf_end),
      ("uv_mode_cfl_cdf", uv_mode_cfl_cdf_start, uv_mode_cfl_cdf_end),
      ("cfl_sign_cdf", cfl_sign_cdf_start, cfl_sign_cdf_end),
      ("cfl_alpha_cdf", cfl_alpha_cdf_start, cfl_alpha_cdf_end),
      ("newmv_cdf", newmv_cdf_start, newmv_cdf_end),
      ("zeromv_cdf", zeromv_cdf_start, zeromv_cdf_end),
      ("refmv_cdf", refmv_cdf_start, refmv_cdf_end),
      ("intra_tx_2_cdf", intra_tx_2_cdf_start, intra_tx_2_cdf_end),
      ("intra_tx_1_cdf", intra_tx_1_cdf_start, intra_tx_1_cdf_end),
      ("inter_tx_3_cdf", inter_tx_3_cdf_start, inter_tx_3_cdf_end),
      ("inter_tx_2_cdf", inter_tx_2_cdf_start, inter_tx_2_cdf_end),
      ("inter_tx_1_cdf", inter_tx_1_cdf_start, inter_tx_1_cdf_end),
      ("tx_size_8x8_cdf", tx_size_8x8_cdf_start, tx_size_8x8_cdf_end),
      ("tx_size_cdf", tx_size_cdf_start, tx_size_cdf_end),
      ("txfm_partition_cdf", txfm_partition_cdf_start, txfm_partition_cdf_end),
      ("skip_cdfs", skip_cdfs_start, skip_cdfs_end),
      ("intra_inter_cdfs", intra_inter_cdfs_start, intra_inter_cdfs_end),
      ("intrabc_cdf", intrabc_cdf_start, intrabc_cdf_end),
      ("angle_delta_cdf", angle_delta_cdf_start, angle_delta_cdf_end),
      ("filter_intra_cdfs", filter_intra_cdfs_start, filter_intra_cdfs_end),
      (
        "palette_y_mode_cdfs",
        palette_y_mode_cdfs_start,
        palette_y_mode_cdfs_end,
      ),
      (
        "palette_uv_mode_cdfs",
        palette_uv_mode_cdfs_start,
        palette_uv_mode_cdfs_end,
      ),
      ("palette_y_size_cdf", palette_y_size_cdf_start, palette_y_size_cdf_end),
      (
        "palette_uv_size_cdf",
        palette_uv_size_cdf_start,
        palette_uv_size_cdf_end,
      ),
      (
        "palette_y_color_index_cdf",
        palette_y_color_index_cdf_start,
        palette_y_color_index_cdf_end,
      ),
      (
        "palette_uv_color_index_cdf",
        palette_uv_color_index_cdf_start,
        palette_uv_color_index_cdf_end,
      ),
      ("comp_mode_cdf", comp_mode_cdf_start, comp_mode_cdf_end),
      ("comp_ref_type_cdf", comp_ref_type_cdf_start, comp_ref_type_cdf_end),
      ("comp_ref_cdf", comp_ref_cdf_start, comp_ref_cdf_end),
      ("comp_bwd_ref_cdf", comp_bwd_ref_cdf_start, comp_bwd_ref_cdf_end),
      ("single_ref_cdfs", single_ref_cdfs_start, single_ref_cdfs_end),
      ("drl_cdfs", drl_cdfs_start, drl_cdfs_end),
      ("compound_mode_cdf", compound_mode_cdf_start, compound_mode_cdf_end),
      ("nmv_context", nmv_context_start, nmv_context_end),
      (
        "deblock_delta_multi_cdf",
        deblock_delta_multi_cdf_start,
        deblock_delta_multi_cdf_end,
      ),
      ("deblock_delta_cdf", deblock_delta_cdf_start, deblock_delta_cdf_end),
      ("delta_q_cdf", delta_q_cdf_start, delta_q_cdf_end),
      (
        "spatial_segmentation_cdfs",
        spatial_segmentation_cdfs_start,
        spatial_segmentation_cdfs_end,
      ),
      ("lrf_switchable_cdf", lrf_switchable_cdf_start, lrf_switchable_cdf_end),
      ("lrf_sgrproj_cdf", lrf_sgrproj_cdf_start, lrf_sgrproj_cdf_end),
      ("lrf_wiener_cdf", lrf_wiener_cdf_start, lrf_wiener_cdf_end),
      ("txb_skip_cdf", txb_skip_cdf_start, txb_skip_cdf_end),
      ("dc_sign_cdf", dc_sign_cdf_start, dc_sign_cdf_end),
      ("eob_extra_cdf", eob_extra_cdf_start, eob_extra_cdf_end),
      ("eob_flag_cdf16", eob_flag_cdf16_start, eob_flag_cdf16_end),
      ("eob_flag_cdf32", eob_flag_cdf32_start, eob_flag_cdf32_end),
      ("eob_flag_cdf64", eob_flag_cdf64_start, eob_flag_cdf64_end),
      ("eob_flag_cdf128", eob_flag_cdf128_start, eob_flag_cdf128_end),
      ("eob_flag_cdf256", eob_flag_cdf256_start, eob_flag_cdf256_end),
      ("eob_flag_cdf512", eob_flag_cdf512_start, eob_flag_cdf512_end),
      ("eob_flag_cdf1024", eob_flag_cdf1024_start, eob_flag_cdf1024_end),
      ("coeff_base_eob_cdf", coeff_base_eob_cdf_start, coeff_base_eob_cdf_end),
      ("coeff_base_cdf", coeff_base_cdf_start, coeff_base_cdf_end),
      ("coeff_br_cdf", coeff_br_cdf_start, coeff_br_cdf_end),
    ]
  }

  pub fn offset<const CDF_LEN: usize>(
    &self, cdf: *const [u16; CDF_LEN],
  ) -> CDFOffset<CDF_LEN> {
    CDFOffset {
      offset: cdf as usize - self as *const _ as usize,
      phantom: PhantomData,
    }
  }
}

impl fmt::Debug for CDFContext {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "CDFContext contains too many numbers to print :-(")
  }
}

macro_rules! symbol_with_update {
  ($self:ident, $w:ident, $s:expr, $cdf:expr) => {
    let cdf = $self.fc.offset($cdf);
    #[cfg(feature = "desync_finder")]
    let _cdf_ptr = ($cdf).as_ptr() as usize;
    $w.symbol_with_update($s, cdf, &mut $self.fc_log, &mut $self.fc);
    #[cfg(feature = "desync_finder")]
    {
      if let Some(map) = $self.fc_map.as_ref() {
        map.lookup(_cdf_ptr);
      }
    }
  };
  ($self:ident, $cdf:expr) => {
    #[cfg(feature = "desync_finder")]
    {
      let cdf: &[_] = $cdf;
      if let Some(map) = $self.fc_map.as_ref() {
        map.lookup(cdf.as_ptr() as usize);
      }
    }
  };
}

#[derive(Clone)]
pub struct ContextWriterCheckpoint {
  pub fc: CDFContextCheckpoint,
  pub bc: BlockContextCheckpoint,
}

/// One undo-log partition. Each entry is `[u16; ENTRY_LEN]` holding an
/// EXACT-length CDF snapshot: `len` payload words at `[0..len]`, the
/// `CDFContext` byte offset at `[ENTRY_LEN - 2]`, and `len` itself at
/// `[ENTRY_LEN - 1]`.
///
/// Exact lengths are load-bearing, not an optimization: the log previously
/// captured and restored a fixed `ENTRY_LEN - 1` words regardless of the
/// CDF's real length, so a push near the end of one CDF table spilled its
/// snapshot into the FIRST WORDS OF THE NEXT FIELD — and because the small
/// and large partitions roll back sequentially (all of `small`, then all of
/// `large`) rather than in one global LIFO order, a large-partition entry's
/// overspilled restore could land after the small partition had already
/// correctly restored those bytes, resurrecting mid-trial CDF state. With
/// `palette_y_color_index_cdf` (8-wide rows, large log) directly followed
/// by `palette_uv_color_index_cdf` (whose n=2..4 rows adapt through the
/// small log), a luma n=8 color-index update at the last row `[6][4]`
/// clobbered the UV n=2 row `[0][0]` on rollback — a real bitstream desync
/// once the UV palette began adapting those bytes (silent while they held
/// constant defaults, which is why luma-only palette streams never hit it).
struct CDFContextLogPartition<const ENTRY_LEN: usize> {
  pub data: Vec<[u16; ENTRY_LEN]>,
}

impl<const ENTRY_LEN: usize> CDFContextLogPartition<ENTRY_LEN> {
  fn new(capacity: usize) -> Self {
    Self { data: Vec::with_capacity(capacity) }
  }
  #[inline(always)]
  fn push<const CDF_LEN: usize>(
    &mut self, fc: &mut CDFContext, cdf: CDFOffset<CDF_LEN>,
  ) -> &mut [u16; CDF_LEN] {
    // (The hard compile-time bound lives in `CDFContextLog::push`, which
    // knows the partition dispatch; this debug_assert covers direct
    // callers. Both partition monomorphizations exist for every CDF_LEN,
    // so an unconditional const assert here would fire for the
    // unreachable small-partition instantiations.)
    debug_assert!(CDF_LEN + 2 <= ENTRY_LEN);
    debug_assert!(cdf.offset <= u16::MAX.into());
    // SAFETY: Maintain an invariant of non-zero spare capacity, so that
    // branching may be deferred until writes are issued. Benchmarks indicate
    // this is faster than first testing capacity and possibly reallocating.
    unsafe {
      let len = self.data.len();
      let new_len = len + 1;
      let capacity = self.data.capacity();
      debug_assert!(new_len <= capacity);
      let dst = self.data.as_mut_ptr().add(len) as *mut u16;
      let base = fc as *mut _ as *mut u8;
      let src = base.add(cdf.offset) as *const u16;
      // Exact-length snapshot: never read (or later restore) bytes beyond
      // this CDF — see the partition doc comment.
      dst.copy_from_nonoverlapping(src, CDF_LEN);
      *dst.add(ENTRY_LEN - 2) = cdf.offset as u16;
      *dst.add(ENTRY_LEN - 1) = CDF_LEN as u16;
      self.data.set_len(new_len);
      if ENTRY_LEN > capacity.wrapping_sub(new_len) {
        self.data.reserve(ENTRY_LEN);
      }
      let cdf = base.add(cdf.offset) as *mut [u16; CDF_LEN];
      &mut *cdf
    }
  }
  #[inline(always)]
  fn rollback(&mut self, fc: &mut CDFContext, checkpoint: usize) {
    let base = fc as *mut _ as *mut u8;
    let mut len = self.data.len();
    // SAFETY: We use unchecked pointers here for performance.
    // Since we know the length, we can ensure not to go OOB.
    unsafe {
      let mut src = self.data.as_mut_ptr().add(len);
      while len > checkpoint {
        len -= 1;
        src = src.sub(1);
        let src = src as *mut u16;
        let offset = *src.add(ENTRY_LEN - 2) as usize;
        let cdf_len = *src.add(ENTRY_LEN - 1) as usize;
        debug_assert!(cdf_len + 2 <= ENTRY_LEN);
        let dst = base.add(offset) as *mut u16;
        dst.copy_from_nonoverlapping(src, cdf_len);
      }
      self.data.set_len(len);
    }
  }
}

const CDF_LEN_SMALL: usize = 4;

pub struct CDFContextLog {
  small: CDFContextLogPartition<{ CDF_LEN_SMALL + 2 }>,
  large: CDFContextLogPartition<{ CDF_LEN_MAX + 2 }>,
}

impl Default for CDFContextLog {
  fn default() -> Self {
    Self {
      small: CDFContextLogPartition::new(1 << 16),
      large: CDFContextLogPartition::new(1 << 9),
    }
  }
}

impl CDFContextLog {
  fn checkpoint(&self) -> CDFContextCheckpoint {
    CDFContextCheckpoint {
      small: self.small.data.len(),
      large: self.large.data.len(),
    }
  }
  #[inline(always)]
  pub fn push<const CDF_LEN: usize>(
    &mut self, fc: &mut CDFContext, cdf: CDFOffset<CDF_LEN>,
  ) -> &mut [u16; CDF_LEN] {
    // Hard compile-time bounds (corruption-class guards, zero runtime
    // cost): every CDF length must fit its partition's entry (payload +
    // offset + length words), and every CDF offset in the context must
    // fit the u16 offset slot -- silent truncation of either would
    // corrupt CDF state on rollback.
    const {
      assert!(
        CDF_LEN + 2
          <= if CDF_LEN <= CDF_LEN_SMALL {
            CDF_LEN_SMALL + 2
          } else {
            CDF_LEN_MAX + 2
          }
      );
      assert!(size_of::<CDFContext>() <= u16::MAX as usize);
    }
    if CDF_LEN <= CDF_LEN_SMALL {
      self.small.push(fc, cdf)
    } else {
      self.large.push(fc, cdf)
    }
  }
  #[inline(always)]
  pub fn rollback(
    &mut self, fc: &mut CDFContext, checkpoint: &CDFContextCheckpoint,
  ) {
    self.small.rollback(fc, checkpoint.small);
    self.large.rollback(fc, checkpoint.large);
  }
  pub fn clear(&mut self) {
    self.small.data.clear();
    self.large.data.clear();
  }
}

pub struct ContextWriter<'a> {
  pub bc: BlockContext<'a>,
  pub fc: &'a mut CDFContext,
  pub fc_log: CDFContextLog,
  #[cfg(feature = "desync_finder")]
  pub fc_map: Option<FieldMap>, // For debugging purposes
}

impl<'a> ContextWriter<'a> {
  pub fn new(fc: &'a mut CDFContext, bc: BlockContext<'a>) -> Self {
    let fc_log = CDFContextLog::default();
    #[allow(unused_mut)]
    let mut cw = ContextWriter {
      bc,
      fc,
      fc_log,
      #[cfg(feature = "desync_finder")]
      fc_map: Default::default(),
    };
    #[cfg(feature = "desync_finder")]
    {
      if std::env::var_os("RAV1E_DEBUG").is_some() {
        cw.fc_map = Some(FieldMap { map: cw.fc.build_map() });
      }
    }

    cw
  }

  pub const fn cdf_element_prob(cdf: &[u16], element: usize) -> u16 {
    (if element > 0 { cdf[element - 1] } else { 32768 })
      - (if element + 1 < cdf.len() { cdf[element] } else { 0 })
  }

  pub fn checkpoint(
    &self, tile_bo: &TileBlockOffset, chroma_sampling: ChromaSampling,
  ) -> ContextWriterCheckpoint {
    ContextWriterCheckpoint {
      fc: self.fc_log.checkpoint(),
      bc: self.bc.checkpoint(tile_bo, chroma_sampling),
    }
  }

  pub fn rollback(&mut self, checkpoint: &ContextWriterCheckpoint) {
    self.fc_log.rollback(self.fc, &checkpoint.fc);
    self.bc.rollback(&checkpoint.bc);
    #[cfg(feature = "desync_finder")]
    {
      if self.fc_map.is_some() {
        self.fc_map = Some(FieldMap { map: self.fc.build_map() });
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  /// The undo log must capture and restore EXACTLY the updated CDF's words.
  /// The old fixed-width log spilled a large-partition snapshot from the
  /// last `palette_y_color_index_cdf` row into the first words of
  /// `palette_uv_color_index_cdf`; because the small partition rolls back
  /// before the large one, the spilled restore resurrected stale UV
  /// color-index state — a real, content-dependent bitstream desync once
  /// the UV palette adapts those bytes (silent for luma-only palettes,
  /// whose adjacent bytes never changed).
  #[test]
  fn cdf_log_rollback_is_exact_length_across_field_boundaries() {
    let mut fc = Box::new(CDFContext::new(50));
    let mut log = CDFContextLog::default();
    let checkpoint = log.checkpoint();

    let uv_default = fc.palette_uv_color_index_cdf[0][0];
    let y_default = fc.palette_y_color_index_cdf[6][4];

    // 1. Adapt the UV n=2 row (small partition) FIRST...
    {
      let uv_row: *const [u16; 2] =
        fc.palette_uv_color_index_cdf[0][0].first_chunk::<2>().unwrap();
      let off = fc.offset(uv_row);
      let cdf = log.push(&mut fc, off);
      cdf[0] = cdf[0].wrapping_add(1000);
      cdf[1] += 1;
    }
    let uv_dirty = fc.palette_uv_color_index_cdf[0][0];

    // 2. ...then update the LAST luma color-index row (8 wide -> large
    // partition), whose old fixed-width snapshot would capture the dirty
    // UV words above.
    {
      let y_row: *const [u16; 8] = &fc.palette_y_color_index_cdf[6][4];
      let off = fc.offset(y_row);
      let cdf = log.push(&mut fc, off);
      cdf[0] = cdf[0].wrapping_add(777);
      cdf[7] += 1;
    }

    // 3. Roll back: both rows must return exactly to their defaults. The
    // old log left the UV row at its step-1 ("dirty") state: the small
    // partition restored it, then the large partition's 16-word restore
    // re-clobbered it with the dirty snapshot.
    log.rollback(&mut fc, &checkpoint);
    std::assert_eq!(
      fc.palette_y_color_index_cdf[6][4],
      y_default,
      "luma color-index row must restore exactly"
    );
    std::assert_eq!(
      fc.palette_uv_color_index_cdf[0][0],
      uv_default,
      "UV color-index row must restore exactly (no cross-field overspill)"
    );
    std::assert_ne!(
      uv_dirty,
      uv_default,
      "sanity: the UV row was actually modified mid-log"
    );
  }
}
