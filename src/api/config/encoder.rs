// Copyright (c) 2020-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use itertools::*;

use crate::api::color::*;
use crate::api::config::GrainTableSegment;
use crate::api::{Rational, SpeedSettings};
use crate::encoder::Tune;
use crate::serialize::{Deserialize, Serialize};

use std::fmt;

// We add 1 to rdo_lookahead_frames in a bunch of places.
// Capped at 256 to prevent unbounded memory allocation.
// Still images use 1 frame; video rarely needs more than 64.
pub(crate) const MAX_RDO_LOOKAHEAD_FRAMES: usize = 256;
// Due to the math in RCState::new() regarding the reservoir frame delay.
pub(crate) const MAX_MAX_KEY_FRAME_INTERVAL: u64 = i32::MAX as u64 / 3;
// Maximum supported frame rate (fps), where the effective rate is
// `time_base.den / time_base.num`. The scene-change detector (`scenechange`
// feature, on by default) forwards this rate into `av-scenechange`'s
// `TilingInfo::from_target_tiles`, which derives `min_tile_rows_ratelimit_log2`
// from `(w * h) * fps / MAX_TILE_RATE`. A pathological fps pushes that minimum
// above `max_tile_rows_log2`, so the subsequent `clamp(min, max)` runs with
// `min > max` and panics (av-scenechange-0.14.1 `data/tile.rs:314`). For the
// smallest encodable frame the panic begins at fps ~= MAX_TILE_RATE / 4096
// (~143616 fps); this ceiling sits well below that and far above any real
// rate (broadcast/web <= 240, high-speed capture <= a few thousand). Mirrors
// the sane-fps bound the fuzz harness applies in `src/fuzzing.rs` (`time_base`
// num/den each constrained to 1..=120, i.e. fps <= 120) so the library rejects
// the same pathological rates the harness avoids, rather than letting them
// reach the dependency's panic at encode time.
pub(crate) const MAX_FRAME_RATE: u64 = 65536;

/// Parameters for the composed coefficient-level RD valuation stack
/// (`EncoderConfig::coeff_rd_stack`): libaom's coupled FP-quantization +
/// per-coefficient RD descent posture as one A/B unit. See the field's
/// documentation and zenavif docs/COEFF_RD_STACK.md for the mechanism map
/// and measured context.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoeffRdStack {
  /// Flat forward-quant rounding offset in 1/256 quantizer-step units,
  /// applied to DC, AC and the EOB dead-zone alike (same mechanics as
  /// `quant_rounding_bias`, which this overrides while armed). 128 = the
  /// aom FP path's 0.5 round-to-nearest. Valid 1..=128.
  pub rounding_bias: u8,
  /// Trellis lambda relative to the block-RDO lambda. aom's tune=iq/ss2
  /// posture is 0.1328 (plane_rd_mult 17 >> 7); aom's default-tune posture
  /// is 4.25 (17 * 8 >> 5). Replaces the opt-in trellis's quality
  /// dampening AND its `ac_quant >= 200` disable. Valid finite, > 0.0,
  /// <= 8.0.
  pub trellis_lambda_scale: f64,
  /// Map aom's `sharpness != 0` preserve gates into the descent: never
  /// zero level-1 coefficients, require level > 2 to lower scan positions
  /// <= 5, floor the level descent at 1, and only pull the EOB in to >= 5
  /// kept coefficients.
  pub preserve_guards: bool,
  /// aom's per-TU zero-out counterweight: after the descent, zero the
  /// whole TU when its coded RD loses to the zero block at the BLOCK
  /// lambda (aom tx_search.c:3294-3311, which runs un-neutered even under
  /// the tunes).
  pub tu_zero_out: bool,
}

impl Default for CoeffRdStack {
  /// The aom tune=ssimulacra2 posture verbatim: FP 0.5-rounding, trellis
  /// lambda 17/128, preserve guards on, no TU zero-out.
  fn default() -> Self {
    Self {
      rounding_bias: 128,
      trellis_lambda_scale: 17.0 / 128.0,
      preserve_guards: true,
      tu_zero_out: false,
    }
  }
}

/// Encoder settings which impact the produced bitstream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderConfig {
  // output size
  /// Width of the frames in pixels.
  pub width: usize,
  /// Height of the frames in pixels.
  pub height: usize,
  /// Sample aspect ratio (for anamorphic video).
  pub sample_aspect_ratio: Rational,
  /// Video time base.
  pub time_base: Rational,

  // data format and ancillary color information
  /// Bit depth.
  pub bit_depth: usize,
  /// Chroma subsampling.
  pub chroma_sampling: ChromaSampling,
  /// Chroma sample position.
  pub chroma_sample_position: ChromaSamplePosition,
  /// Pixel value range.
  pub pixel_range: PixelRange,
  /// Content color description (primaries, transfer characteristics, matrix).
  pub color_description: Option<ColorDescription>,
  /// HDR mastering display parameters.
  pub mastering_display: Option<MasteringDisplay>,
  /// HDR content light parameters.
  pub content_light: Option<ContentLight>,

  /// AV1 level index to target (0-31).
  /// If None, allow the encoder to decide.
  /// Currently, rav1e is unable to guarantee that the output bitstream
  /// meets the rate limitations of the specified level.
  pub level_idx: Option<u8>,

  /// Enable signaling timing info in the bitstream.
  pub enable_timing_info: bool,

  /// Still picture mode flag.
  pub still_picture: bool,

  /// Flag to force all frames to be error resilient.
  pub error_resilient: bool,

  /// Interval between switch frames (0 to disable)
  pub switch_frame_interval: u64,

  // encoder configuration
  /// The *minimum* interval between two keyframes
  pub min_key_frame_interval: u64,
  /// The *maximum* interval between two keyframes
  pub max_key_frame_interval: u64,
  /// The number of temporal units over which to distribute the reservoir
  /// usage.
  pub reservoir_frame_delay: Option<i32>,
  /// Flag to enable low latency mode.
  ///
  /// In this mode the frame reordering is disabled.
  pub low_latency: bool,
  /// The base quantizer to use.
  pub quantizer: usize,
  /// The minimum allowed base quantizer to use in bitrate mode.
  pub min_quantizer: u8,
  /// The target bitrate for the bitrate mode.
  pub bitrate: i32,
  /// Metric to tune the quality for.
  pub tune: Tune,
  /// Parameters for grain synthesis.
  pub film_grain_params: Option<Vec<GrainTableSegment>>,
  /// Number of tiles horizontally. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub tile_cols: usize,
  /// Number of tiles vertically. Must be a power of two.
  ///
  /// Overridden by [`tiles`], if present.
  ///
  /// [`tiles`]: #structfield.tiles
  pub tile_rows: usize,
  /// Total number of tiles desired.
  ///
  /// Encoder will try to optimally split to reach this number of tiles,
  /// rounded up. Overrides [`tile_cols`] and [`tile_rows`].
  ///
  /// [`tile_cols`]: #structfield.tile_cols
  /// [`tile_rows`]: #structfield.tile_rows
  pub tiles: usize,

  /// Enable AV1 quantization matrices for perceptual quality improvement.
  /// QM applies frequency-dependent quantization weights based on contrast
  /// sensitivity, giving ~10% BD-rate improvement for photographic content.
  pub enable_qm: bool,

  /// Enable variance adaptive quantization (VAQ).
  /// Allocates more bits to smooth/flat regions (where artifacts are visible)
  /// and fewer bits to textured regions (where texture masks distortion).
  /// Uses AV1 segmentation with SSIM-weighted activity masking.
  /// Works independently of `tune` mode.
  pub enable_vaq: bool,

  /// VAQ strength (0.0 to 4.0, default 1.0).
  /// Controls how aggressively bits are redistributed based on variance.
  /// 0.0 = no redistribution, 1.0 = default SSIM weighting,
  /// >1.0 = stronger redistribution toward smooth areas.
  /// > Only effective when `enable_vaq` is true.
  pub vaq_strength: f64,

  /// Segmentation boost power (valid range 0.5..=4.0, default 1.0 = disabled).
  /// When > 1.0, amplifies the dynamic range of segmentation QP offsets
  /// independently of RDO distortion weighting. This allows wider bit
  /// redistribution without inflating total bitrate through RDO.
  /// Typical values: 1.0 (off), 1.5–2.5 (moderate–aggressive boost).
  /// Values outside `0.5..=4.0` are rejected by `validate()` as `InvalidSegBoost`.
  pub seg_boost: f64,

  /// Enable trellis quantization (Viterbi DP coefficient optimization).
  /// Uses rate-distortion optimization to find the globally optimal
  /// combination of coefficient levels, exploiting AV1 entropy coding
  /// dependencies between coefficients. Encoder-only, bitstream compatible.
  pub enable_trellis: bool,

  /// Override the [`Tune::Ssimulacra2`] per-superblock Variance Boost
  /// strength (libaom units; the tune's fitted default is 1.0, libaom's own
  /// default maps to 3.0). `None` keeps the fitted constant and is
  /// byte-identical to builds without this knob. Valid range 0.0..=6.0
  /// (0.0 disables the boost map entirely). Only effective under
  /// [`Tune::Ssimulacra2`] on KEY/intra-only frames (the only path that
  /// computes the boost map).
  ///
  /// Measured context (zenavif `benchmarks/rd_gap_deltaq_2026-07-02.tsv`):
  /// the global strength response is an inverted-U on ssim2 with monotone
  /// butteraugli decay — strengths >= 3 are butteraugli-vetoed corpus-wide,
  /// but deep-AQ content (1-bit scans / ornate interiors / smooth
  /// illustrations, the P3 iq-AQ residual class) measures monotone gains to
  /// 4.5. This knob exists for per-image heads and A/B arms, not as a
  /// global default.
  ///
  /// [`Tune::Ssimulacra2`]: crate::api::Tune::Ssimulacra2
  pub variance_boost_strength: Option<f64>,

  /// Deep-flat Variance Boost ramp: `(deep_strength, ceil_log2)`. When set,
  /// the effective boost strength for a superblock with smoothed 8x8
  /// variance `v` ramps linearly in `log2(v)` from `deep_strength` at
  /// `v = 1` to the base strength (the tune default or
  /// [`variance_boost_strength`](Self::variance_boost_strength)) at
  /// `v >= 2^ceil_log2`; above the ceiling the base strength applies
  /// unchanged. This reproduces libaom tune=iq's DEEPER per-SB qindex
  /// spread on near-flat content (the aom {36,64}-vs-{42,61} bimodal-map
  /// evidence on 1-bit rescans, zenavif docs/RD_GAP_VS_LIBAOM.md
  /// "Near-lossless rescans residual") without re-boosting the mid-variance
  /// superblocks that the global strength fit measured as
  /// butteraugli-vetoed on photos. `None` = flat strength (byte-identical).
  /// Valid: strength finite 0.0..=6.0, ceil_log2 1..=10.
  pub variance_boost_deep: Option<(f64, u8)>,

  /// Flat quantizer rounding bias, in 1/256 units of the quantizer step.
  /// When set, replaces the fitted rounding offsets (DC 109, AC 98/109,
  /// EOB 88 — Valin-method RD-derived, see `QuantizationContext::update`)
  /// with a single flat offset `k/256` for DC, AC and the EOB dead-zone
  /// alike. `Some(128)` is 0.5-rounding: every coefficient at or above half
  /// a quantizer step codes and extends the EOB — the libaom
  /// `sharpness != 0` quantizer path (`av1_build_quantizer` qrounding
  /// 48->64 of 128, i.e. dead-zone removal) that aom tune=iq (at qindex
  /// <= 112, via the adaptive-sharpness clamp) and tune=ssimulacra2
  /// (everywhere, sharpness=7 unclamped) run with. `None` keeps the fitted
  /// offsets (byte-identical). Valid range 1..=128.
  ///
  /// This is the P3 "6096 coefficient-level no-skip" probe (zenavif
  /// docs/RD_GAP_VS_LIBAOM.md "Near-lossless rescans residual"): at
  /// byte-matched near-lossless cells aom codes coefficients on 100% of
  /// 4x4 cells at baseQ 64 while the fitted offsets skip 57.5% at baseQ 54.
  /// Encoder-side value choice only — no bitstream syntax changes. Applies
  /// to every frame this encoder quantizes (not tune-gated), including the
  /// trellis-off path zenavif ships.
  pub quant_rounding_bias: Option<u8>,

  /// Per-16×16 ssim-rdmult λ scaling strength, a port of libaom's
  /// `av1_set_mb_ssim_rdmult_scaling` + `av1_set_ssim_rdmult`
  /// (encoder_utils.c / encodeframe_utils.c at rev 632172a4, shared by
  /// aom `--tune={ssim,iq,ssimulacra2}`): per 16×16 source block, the mean
  /// per-pixel 8×8 variance feeds
  /// `factor = 67.035434·(1 − exp(−0.0021489·var)) + 17.492222`, factors
  /// are normalized by the frame geometric mean (range ≈ [0.207, 4.832]),
  /// and every coding block's RD **rate** term is scaled by the geometric
  /// mean of the factors it covers — bits cost more in high-variance
  /// (masked) areas and less in flat ones. This is the λ-side counterpart
  /// of the distortion-side `ssim_boost` activity masking the psy tunes
  /// already run; the two compose.
  ///
  /// The value is an exponent blend on the normalized factor
  /// (`factor^strength`): `1.0` = the aom curve verbatim, `0.5` = its
  /// geometric half, `0.0` = off. Exponentiation preserves the geomean-1
  /// normalization, so any strength is a pure spatial reallocation with no
  /// global λ shift (the frame-level aom rdmult weight was measured
  /// +4.41% BD — this knob deliberately cannot reproduce that failure
  /// mode). `None` = off (byte-identical to builds without this knob).
  /// Only effective under [`Tune::Ssimulacra2`]. Valid range 0.0..=4.0.
  ///
  /// [`Tune::Ssimulacra2`]: crate::api::Tune::Ssimulacra2
  pub ssim_rdmult_strength: Option<f64>,

  /// The composed coefficient-level RD valuation stack — libaom's coupled
  /// "FP round-to-nearest quantization + always-on per-coefficient RD
  /// descent" posture, ported as ONE knob (zenavif
  /// docs/COEFF_RD_STACK.md; aom rev 632172a4: `av1_build_quantizer`
  /// round_fp = 64/128, `skip_trellis ? B : FP` coupling in tx_search.c,
  /// `av1_optimize_txb` with the tune's rdmult posture and sharpness
  /// guards). When set:
  ///
  /// - forward quantization uses the flat `rounding_bias`/256 offset for
  ///   DC, AC and the EOB dead-zone (identical mechanics to
  ///   [`quant_rounding_bias`](Self::quant_rounding_bias), which this
  ///   overrides; 128 = aom FP parity);
  /// - the trellis runs on every transform block regardless of
  ///   [`enable_trellis`](Self::enable_trellis), WITHOUT the `ac_quant >=
  ///   200` disable and WITHOUT the `80/ac_quant` quality dampening, at
  ///   `lambda_trellis = lambda * trellis_lambda_scale` (aom's ss2-tune
  ///   posture is 0.1328 = plane_rd_mult 17 / 128; aom's default-tune
  ///   posture is 4.25 = 17 * 8 / 32);
  /// - `preserve_guards` maps aom's `sharpness != 0` trellis gates:
  ///   level-1 coefficients are never zeroed, near-DC (scan pos <= 5)
  ///   coefficients need level > 2 to be lowered, level descent floors at
  ///   1 instead of 0, and the EOB may only be pulled in to >= 5 kept
  ///   coefficients;
  /// - `tu_zero_out` adds aom's per-TU counterweight (tx_search.c
  ///   :3294-3311): after the trellis, the whole TU is zeroed when the
  ///   coded rate-distortion loses to the zero block at the BLOCK lambda.
  ///
  /// `None` = byte-identical to builds without this knob. The two prior
  /// half-stack probes are both measured rejections (flat rounding alone
  /// +2.67% med / 20-of-23 butteraugli vetoes; forced trellis alone
  /// +0.32..0.55%) — this knob exists to measure the composition aom
  /// actually ships, which neither probe reached.
  pub coeff_rd_stack: Option<CoeffRdStack>,

  /// Maximum pixel count (width * height). Default 120_000_000 (120 megapixels).
  /// Set to 0 to disable the limit. Validated in `Config::validate()`.
  pub max_pixel_count: u64,

  /// Settings which affect the encoding speed vs. quality trade-off.
  pub speed_settings: SpeedSettings,
}

/// Default preset for `EncoderConfig`: it is a balance between quality and
/// speed. See [`with_speed_preset()`].
///
/// [`with_speed_preset()`]: struct.EncoderConfig.html#method.with_speed_preset
impl Default for EncoderConfig {
  fn default() -> Self {
    const DEFAULT_SPEED: u8 = 6;
    Self::with_speed_preset(DEFAULT_SPEED)
  }
}

impl EncoderConfig {
  /// This is a preset which provides default settings according to a speed
  /// value in the specific range 0–10. Each speed value corresponds to a
  /// different preset. See [`from_preset()`]. If the input value is greater
  /// than 10, it will result in the same settings as 10.
  ///
  /// [`from_preset()`]: struct.SpeedSettings.html#method.from_preset
  pub fn with_speed_preset(speed: u8) -> Self {
    EncoderConfig {
      width: 640,
      height: 480,
      sample_aspect_ratio: Rational { num: 1, den: 1 },
      time_base: Rational { num: 1, den: 30 },

      bit_depth: 8,
      chroma_sampling: ChromaSampling::Cs420,
      chroma_sample_position: ChromaSamplePosition::Unknown,
      pixel_range: Default::default(),
      color_description: None,
      mastering_display: None,
      content_light: None,

      level_idx: None,

      enable_timing_info: false,

      still_picture: false,

      error_resilient: false,
      switch_frame_interval: 0,

      min_key_frame_interval: 12,
      max_key_frame_interval: 240,
      min_quantizer: 0,
      reservoir_frame_delay: None,
      low_latency: false,
      quantizer: 100,
      bitrate: 0,
      tune: Tune::default(),
      film_grain_params: None,
      tile_cols: 0,
      tile_rows: 0,
      tiles: 0,
      enable_qm: false,
      enable_vaq: false,
      vaq_strength: 1.0,
      seg_boost: 1.0,
      enable_trellis: false,
      variance_boost_strength: None,
      variance_boost_deep: None,
      quant_rounding_bias: None,
      ssim_rdmult_strength: None,
      coeff_rd_stack: None,
      max_pixel_count: 120_000_000, // 120 megapixels (admits 108 MP phone photos)
      speed_settings: SpeedSettings::from_preset(speed),
    }
  }

  /// Sets the minimum and maximum keyframe interval, handling special cases as needed.
  pub fn set_key_frame_interval(
    &mut self, min_interval: u64, max_interval: u64,
  ) {
    self.min_key_frame_interval = min_interval;

    // Map an input value of 0 to an infinite interval
    self.max_key_frame_interval = if max_interval == 0 {
      MAX_MAX_KEY_FRAME_INTERVAL
    } else {
      max_interval
    };
  }

  /// Returns the video frame rate computed from [`time_base`].
  ///
  /// [`time_base`]: #structfield.time_base
  pub fn frame_rate(&self) -> f64 {
    Rational::from_reciprocal(self.time_base).as_f64()
  }

  /// Computes the render width and height of the stream based
  /// on [`width`], [`height`], and [`sample_aspect_ratio`].
  ///
  /// [`width`]: #structfield.width
  /// [`height`]: #structfield.height
  /// [`sample_aspect_ratio`]: #structfield.sample_aspect_ratio
  pub fn render_size(&self) -> (usize, usize) {
    let sar = self.sample_aspect_ratio.as_f64();

    if sar > 1.0 {
      ((self.width as f64 * sar).round() as usize, self.height)
    } else {
      (self.width, (self.height as f64 / sar).round() as usize)
    }
  }

  /// Is temporal RDO enabled ?
  #[inline]
  pub const fn temporal_rdo(&self) -> bool {
    // Note: This function is called frequently, unlike most other functions here.

    // `compute_distortion_scale` computes a scaling factor for the distortion
    // of an 8x8 block (4x4 blocks simply use the scaling of the enclosing 8x8
    // block). As long as distortion is always computed on <= 8x8 blocks, this
    // has the property that the scaled distortion of a 2Nx2N block is always
    // equal to the sum of the scaled distortions of the NxN sub-blocks it's
    // made of, this is a necessary property to be able to do RDO between
    // multiple partition sizes properly. Unfortunately, when tx domain
    // distortion is used, distortion is only known at the tx block level which
    // might be bigger than 8x8. So temporal RDO is always disabled in that case.
    !self.speed_settings.transform.tx_domain_distortion
  }

  /// Describes whether the output is targeted as HDR
  pub fn is_hdr(&self) -> bool {
    self
      .color_description
      .map(|colors| {
        colors.transfer_characteristics == TransferCharacteristics::SMPTE2084
      })
      .unwrap_or(false)
  }

  pub(crate) fn get_film_grain_at(
    &self, timestamp: u64,
  ) -> Option<&GrainTableSegment> {
    self.film_grain_params.as_ref().and_then(|entries| {
      entries.iter().find(|entry| {
        timestamp >= entry.start_time && timestamp < entry.end_time
      })
    })
  }

  pub(crate) fn get_film_grain_mut_at(
    &mut self, timestamp: u64,
  ) -> Option<&mut GrainTableSegment> {
    self.film_grain_params.as_mut().and_then(|entries| {
      entries.iter_mut().find(|entry| {
        timestamp >= entry.start_time && timestamp < entry.end_time
      })
    })
  }
}

impl fmt::Display for EncoderConfig {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    let pairs = [
      ("keyint_min", self.min_key_frame_interval.to_string()),
      ("keyint_max", self.max_key_frame_interval.to_string()),
      ("quantizer", self.quantizer.to_string()),
      ("bitrate", self.bitrate.to_string()),
      ("min_quantizer", self.min_quantizer.to_string()),
      ("low_latency", self.low_latency.to_string()),
      ("tune", self.tune.to_string()),
      (
        "rdo_lookahead_frames",
        self.speed_settings.rdo_lookahead_frames.to_string(),
      ),
      (
        "multiref",
        (!self.low_latency || self.speed_settings.multiref).to_string(),
      ),
      ("fast_deblock", self.speed_settings.fast_deblock.to_string()),
      (
        "scene_detection_mode",
        self.speed_settings.scene_detection_mode.to_string(),
      ),
      ("cdef", self.speed_settings.cdef.to_string()),
      ("lrf", self.speed_settings.lrf.to_string()),
      ("enable_timing_info", self.enable_timing_info.to_string()),
      ("enable_qm", self.enable_qm.to_string()),
      ("enable_vaq", self.enable_vaq.to_string()),
      ("vaq_strength", self.vaq_strength.to_string()),
      ("seg_boost", self.seg_boost.to_string()),
      (
        "min_block_size",
        self.speed_settings.partition.partition_range.min.to_string(),
      ),
      (
        "max_block_size",
        self.speed_settings.partition.partition_range.max.to_string(),
      ),
      (
        "encode_bottomup",
        self.speed_settings.partition.encode_bottomup.to_string(),
      ),
      (
        "non_square_partition_max_threshold",
        self
          .speed_settings
          .partition
          .non_square_partition_max_threshold
          .to_string(),
      ),
      (
        "reduced_tx_set",
        self.speed_settings.transform.reduced_tx_set.to_string(),
      ),
      (
        "tx_domain_distortion",
        self.speed_settings.transform.tx_domain_distortion.to_string(),
      ),
      (
        "tx_domain_rate",
        self.speed_settings.transform.tx_domain_rate.to_string(),
      ),
      (
        "rdo_tx_decision",
        self.speed_settings.transform.rdo_tx_decision.to_string(),
      ),
      (
        "prediction_modes",
        self.speed_settings.prediction.prediction_modes.to_string(),
      ),
      (
        "fine_directional_intra",
        self.speed_settings.prediction.fine_directional_intra.to_string(),
      ),
      (
        "include_near_mvs",
        self.speed_settings.motion.include_near_mvs.to_string(),
      ),
      (
        "use_satd_subpel",
        self.speed_settings.motion.use_satd_subpel.to_string(),
      ),
    ];
    write!(
      f,
      "{}",
      pairs.iter().map(|pair| format!("{}={}", pair.0, pair.1)).join(" ")
    )
  }
}
