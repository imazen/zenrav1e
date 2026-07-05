// Copyright (c) 2020-2023, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use arg_enum_proc_macro::ArgEnum;
use num_derive::*;

use crate::partition::BlockSize;
use crate::serialize::{Deserialize, Serialize};

use std::fmt;

// NOTE: Add Structures at the end.
/// Contains the speed settings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpeedSettings {
  /// Enables inter-frames to have multiple reference frames.
  ///
  /// Enabled is slower.
  pub multiref: bool,

  /// Enables fast deblocking filter.
  pub fast_deblock: bool,

  /// The number of lookahead frames to be used for temporal RDO.
  ///
  /// Higher is slower.
  pub rdo_lookahead_frames: usize,

  /// Which scene detection mode to use. Standard is slower, but best.
  pub scene_detection_mode: SceneDetectionSpeed,

  /// Enables CDEF.
  pub cdef: bool,

  /// Enables LRF.
  pub lrf: bool,

  /// Enable searching loop restoration units when no transforms have been coded
  /// restoration unit.
  pub lru_on_skip: bool,

  /// The amount of search done for self guided restoration.
  pub sgr_complexity: SGRComplexityLevel,

  /// Search level for segmentation.
  ///
  /// Full search is at least twice as slow.
  pub segmentation: SegmentationLevel,

  // NOTE: put enums and basic type fields above
  /// Speed settings related to partition decision
  pub partition: PartitionSpeedSettings,

  /// Speed settings related to transform size and type decision
  pub transform: TransformSpeedSettings,

  /// Speed settings related to intra prediction mode selection
  pub prediction: PredictionSpeedSettings,

  /// Speed settings related to motion estimation and motion vector selection
  pub motion: MotionSpeedSettings,
}

impl Default for SpeedSettings {
  /// The default settings are equivalent to speed 0
  fn default() -> Self {
    SpeedSettings {
      multiref: true,
      fast_deblock: false,
      rdo_lookahead_frames: 40,
      scene_detection_mode: SceneDetectionSpeed::Standard,
      cdef: true,
      lrf: true,
      lru_on_skip: true,
      sgr_complexity: SGRComplexityLevel::Full,
      segmentation: SegmentationLevel::Complex,
      partition: PartitionSpeedSettings {
        encode_bottomup: true,
        non_square_partition_max_threshold: BlockSize::BLOCK_64X64,
        partition_range: PartitionRange::new(
          BlockSize::BLOCK_4X4,
          BlockSize::BLOCK_64X64,
        ),
        mixed_3way_partitions: false,
        split_trial_depth: 1,
        topdown_prune: None,
      },
      transform: TransformSpeedSettings {
        reduced_tx_set: false,
        // TX domain distortion is always faster, with no significant quality change,
        // although it will be ignored when Tune == Psychovisual.
        tx_domain_distortion: true,
        tx_domain_rate: false,
        rdo_tx_decision: true,
        rdo_tx_size_override: None,
        rdo_tx_type_override: None,
        rdo_tx_size_depth: None,
        enable_inter_tx_split: false,
      },
      prediction: PredictionSpeedSettings {
        prediction_modes: PredictionModesSetting::ComplexAll,
        fine_directional_intra: true,
        // Default off pending RD measurement; enable per-encode for screen
        // content.
        palette: PaletteMode::Off,
        intrabc: false,
        intrabc_hash: true,
        filter_intra: None,
        num_modes_rdo_override: None,
      },
      motion: MotionSpeedSettings {
        include_near_mvs: true,
        use_satd_subpel: true,
        me_allow_full_search: true,
      },
    }
  }
}

impl SpeedSettings {
  /// Set the speed setting according to a numeric speed preset.
  pub fn from_preset(speed: u8) -> Self {
    // The default settings are equivalent to speed 0
    let mut settings = SpeedSettings::default();

    if speed >= 1 {
      settings.lru_on_skip = false;
      settings.segmentation = SegmentationLevel::Simple;
    }

    if speed >= 2 {
      settings.partition.non_square_partition_max_threshold =
        BlockSize::BLOCK_8X8;

      settings.prediction.prediction_modes =
        PredictionModesSetting::ComplexKeyframes;
    }

    if speed >= 3 {
      settings.rdo_lookahead_frames = 30;

      settings.partition.partition_range =
        PartitionRange::new(BlockSize::BLOCK_8X8, BlockSize::BLOCK_64X64);
    }

    if speed >= 4 {
      settings.partition.encode_bottomup = false;
    }

    if speed >= 5 {
      settings.sgr_complexity = SGRComplexityLevel::Reduced;
      settings.motion.include_near_mvs = false;
    }

    if speed >= 6 {
      settings.rdo_lookahead_frames = 20;

      settings.transform.rdo_tx_decision = false;
      settings.transform.reduced_tx_set = true;

      settings.motion.me_allow_full_search = false;
    }

    if speed >= 7 {
      settings.prediction.prediction_modes = PredictionModesSetting::Simple;
      // Multiref is enabled automatically if low_latency is false.
      //
      // If low_latency is true, enabling multiref allows using multiple
      // backwards references. low_latency false enables both forward and
      // backwards references.
      settings.multiref = false;
      settings.fast_deblock = true;
    }

    if speed >= 8 {
      settings.rdo_lookahead_frames = 10;
      settings.lrf = false;
    }

    if speed >= 9 {
      // 8x8 is fast enough to use until very high speed levels,
      // because 8x8 with reduced TX set is faster but with equivalent
      // or better quality compared to 16x16 (to which reduced TX set does not apply).
      settings.partition.partition_range =
        PartitionRange::new(BlockSize::BLOCK_16X16, BlockSize::BLOCK_32X32);

      // FIXME: With unknown reasons, inter_tx_split does not work if reduced_tx_set is false
      settings.transform.enable_inter_tx_split = true;
    }

    if speed >= 10 {
      settings.scene_detection_mode = SceneDetectionSpeed::Fast;

      settings.partition.partition_range =
        PartitionRange::new(BlockSize::BLOCK_32X32, BlockSize::BLOCK_32X32);

      settings.motion.use_satd_subpel = false;
    }

    settings
  }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Default))]
/// Speed settings related to transform size and type decision
pub struct TransformSpeedSettings {
  /// Enables reduced transform set.
  ///
  /// Enabled is faster.
  pub reduced_tx_set: bool,

  /// Enables using transform-domain distortion instead of pixel-domain.
  ///
  /// Enabled is faster.
  pub tx_domain_distortion: bool,

  /// Enables using transform-domain rate estimation.
  ///
  /// Enabled is faster.
  pub tx_domain_rate: bool,

  /// Enables searching transform size and type with RDO.
  ///
  /// Enabled is slower.
  pub rdo_tx_decision: bool,

  /// Decouples the intra transform-SIZE half of
  /// [`rdo_tx_decision`](Self::rdo_tx_decision).
  ///
  /// `None` (the default at every speed preset) follows `rdo_tx_decision`
  /// exactly — byte-identical to builds without this knob. `Some(true)`
  /// searches intra tx sizes with RDO (the frame codes `TX_MODE_SELECT`)
  /// even when `rdo_tx_decision` is off; `Some(false)` pins the largest
  /// legal tx size (`TX_MODE_LARGEST`) even when it is on.
  pub rdo_tx_size_override: Option<bool>,

  /// Decouples the intra transform-TYPE half of
  /// [`rdo_tx_decision`](Self::rdo_tx_decision).
  ///
  /// `None` (the default at every speed preset) follows `rdo_tx_decision`.
  /// `Some(true)` RDO-searches the legal tx-type set for whichever tx size
  /// is being coded — this works under `TX_MODE_LARGEST` too, since
  /// tx-type signaling is independent of the frame tx mode. `Some(false)`
  /// codes `DCT_DCT` only.
  pub rdo_tx_type_override: Option<bool>,

  /// Caps the intra tx-size RDO walk depth (split levels evaluated below
  /// the largest legal tx size).
  ///
  /// `None` (the default at every speed preset) runs the full walk
  /// (`MAX_TX_DEPTH` = 2 levels below the largest). `Some(1)` evaluates
  /// the largest size plus one split level; `Some(0)` evaluates only the
  /// largest (the frame still codes `TX_MODE_SELECT`, so this isolates
  /// the per-block depth-signaling overhead). Ignored when intra tx-size
  /// RDO is off.
  pub rdo_tx_size_depth: Option<u8>,

  /// Enable tx split for inter mode block.
  pub enable_inter_tx_split: bool,
}

impl TransformSpeedSettings {
  /// Whether intra tx-SIZE RDO is enabled:
  /// [`rdo_tx_size_override`](Self::rdo_tx_size_override) falling back to
  /// [`rdo_tx_decision`](Self::rdo_tx_decision).
  pub(crate) const fn tx_size_rdo(&self) -> bool {
    match self.rdo_tx_size_override {
      Some(v) => v,
      None => self.rdo_tx_decision,
    }
  }

  /// Whether intra tx-TYPE RDO is enabled:
  /// [`rdo_tx_type_override`](Self::rdo_tx_type_override) falling back to
  /// [`rdo_tx_decision`](Self::rdo_tx_decision).
  pub(crate) const fn tx_type_rdo(&self) -> bool {
    match self.rdo_tx_type_override {
      Some(v) => v,
      None => self.rdo_tx_decision,
    }
  }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Default))]
/// Speed settings related to partition decision
pub struct PartitionSpeedSettings {
  /// Enables bottom-up encoding, rather than top-down.
  ///
  /// Enabled is slower.
  pub encode_bottomup: bool,

  /// Allow non-square partition type outside of frame borders
  /// on any blocks at or below this size.
  pub non_square_partition_max_threshold: BlockSize,

  /// Range of partition sizes that can be used. Larger ranges are slower.
  ///
  /// Must be based on square block sizes, so e.g. 8×4 isn't allowed here.
  pub partition_range: PartitionRange,

  /// Offer AV1's mixed-granularity 3-way partition types
  /// (`PARTITION_HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`) as candidates in the
  /// top-down partition RDO search.
  ///
  /// Enabled is slower (~1.5x on still images) with a small compression
  /// improvement. Disabled (the default at every speed preset) keeps the
  /// search byte-identical to builds without this feature — it is an
  /// explicit opt-in for beyond-matched-speed operating points.
  pub mixed_3way_partitions: bool,

  /// Recursion depth of the top-down SPLIT-trial cost refinement
  /// (`min(NONE leaf, one-level-deeper SPLIT)` per child).
  ///
  /// `1` (the default at every preset, and the minimum — `0` is treated as
  /// `1`) is the shipped one-level estimate; higher values refine each
  /// quarter's cost recursively, sharpening SPLIT-vs-large-block ranking
  /// where large partition ranges are searched, at extra encode cost.
  /// Only meaningful above the minimum partition size; an explicit opt-in
  /// for beyond-matched-speed operating points.
  pub split_trial_depth: u8,

  /// Early-exit pruning schedule for the top-down partition search
  /// (libaom `partition_search_breakout`-family analog, adapted to the
  /// native NONE-vs-SPLIT-trial cost model).
  ///
  /// `None` (the default at every speed preset) keeps the search
  /// byte-identical to builds without this feature. `Some` re-orders the
  /// candidate walk NONE-first — so the existing per-child early exit
  /// abandons expensive candidates against the NONE incumbent — and
  /// applies whichever gates below are set. Designed to keep
  /// HORZ/VERT (and 16-parent 4-way) candidates affordable at fast
  /// presets, where the historical alternative was deleting them from
  /// the candidate list outright.
  #[serde(default)]
  pub topdown_prune: Option<TopdownPartitionPrune>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
/// Gate thresholds for the pruned top-down partition candidate walk
/// (`PartitionSpeedSettings::topdown_prune`). Every field is an
/// independent opt-in; an all-`None` value only re-orders the walk
/// NONE-first (which changes RD tie-breaks, hence bitstreams — the
/// whole struct lives behind an `Option` for byte-identity when off).
pub struct TopdownPartitionPrune {
  /// Terminate the candidate walk at PARTITION_NONE — skipping the
  /// SPLIT trial and every non-square candidate — when the NONE
  /// incumbent coded as skip (no residual) and its rd cost is below
  /// `none_breakout × lambda × block_pixels`. The `lambda` factor keys
  /// the threshold to the active quantizer and the `block_pixels`
  /// factor to the block size, mirroring libaom's
  /// `partition_search_breakout_dist_thr >> (sb_log2s − bsize_log2s)`
  /// + `rate_thr × num_pels_log2` scaling of the same decision.
  pub none_breakout: Option<f32>,
  /// Skip HORZ/VERT evaluation when the NONE incumbent beats the
  /// SPLIT-trial estimate by more than this relative margin
  /// (`(split − none) / none > margin`; a SPLIT trial abandoned by the
  /// early exit counts as unbounded NONE dominance). One-sided by
  /// measurement: directional candidates earn most of their value on
  /// SPLIT-dominant content, where the per-child early exit already
  /// bounds their cost — only clear NONE dominance justifies skipping
  /// them outright. Inert until the SPLIT trial has run in the walk.
  pub rect_margin: Option<f32>,
  /// Same-shaped NONE-dominance gate for the extended candidates
  /// (HORZ_4/VERT_4 and, when `mixed_3way_partitions` offers them,
  /// HORZ_A/B + VERT_A/B); typically tighter than `rect_margin`.
  pub four_way_margin: Option<f32>,
  /// Skip every non-square candidate when the deviation
  /// `max − min` of `ln(1 + var)` over the block's 4×4 luma sub-blocks
  /// (bit-depth-normalized source pixels) is below this value —
  /// directional partitions cannot pay for their signaling on
  /// homogeneous content. Port of libaom's allintra
  /// `prune_rect_part_using_4x4_var_deviation` (threshold 3.0 there;
  /// `log_sub_block_var`, partition_search.c).
  pub homogeneity_gate: Option<f32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Default))]
/// Speed settings related to motion estimation and motion vector selection
pub struct MotionSpeedSettings {
  /// Use SATD instead of SAD for subpixel search.
  ///
  /// Enabled is slower.
  pub use_satd_subpel: bool,

  /// Enables searching near motion vectors during RDO.
  ///
  /// Enabled is slower.
  pub include_near_mvs: bool,

  /// Enable full search in some parts of motion estimation. Allowing full
  /// search is slower.
  pub me_allow_full_search: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Default))]
/// Speed settings related to intra prediction mode selection
pub struct PredictionSpeedSettings {
  /// Prediction modes to search.
  ///
  /// Complex settings are slower.
  pub prediction_modes: PredictionModesSetting,

  /// Use fine directional intra prediction
  pub fine_directional_intra: bool,

  /// Search palette mode for intra blocks (AV1 screen content tool).
  ///
  /// Only takes effect when the sequence allows screen content tools
  /// (always the case for still pictures). Helps screen content (text,
  /// plots, UI); wasted work on photographic content — `Auto` runs a
  /// per-frame detection pass to decide.
  pub palette: PaletteMode,

  /// Search intra block copy (intraBC) on intra frames (AV1 screen
  /// content tool): blocks copy from the already-reconstructed area of
  /// the same frame, which captures the exact repeats screen content is
  /// full of (text glyphs, grid lines, UI chrome).
  ///
  /// Requires `palette != Off` (the screen-content signaling machinery):
  /// with `PaletteMode::Auto` the same per-frame detection decides
  /// `allow_intrabc` (its stricter variant); with `Always` it is always
  /// allowed. Note that `allow_intrabc` disables all in-loop filters
  /// (deblocking, CDEF, LRF) for the frame per the AV1 spec — the
  /// detection exists precisely to limit that trade to frames where the
  /// copy tool pays for it.
  pub intrabc: bool,

  /// Use the hash-based exact-match candidate table in the intraBC search
  /// (chunk B of the intraBC program): the tile's source luma is
  /// block-hashed once per tile encode, and exact source matches anywhere
  /// in the valid area become displacement-vector candidates alongside
  /// the local (predictor-seeded diamond) search. This is what finds
  /// long-range repeats — repeated glyphs, dashed grid lines, tiled UI —
  /// that a local search never reaches. Inert unless
  /// [`intrabc`](Self::intrabc) is on (and the frame's screen-content
  /// gates arm it).
  pub intrabc_hash: bool,

  /// Override the sequence-level filter-intra enable (`None` derives it
  /// from `prediction_modes >= ComplexKeyframes`, the historical
  /// behavior).
  pub filter_intra: Option<bool>,

  /// Override the number of intra prediction modes that reach full RDO on
  /// intra frames (the SATD-prescreened shortlist length).
  ///
  /// `None` (the default at every speed preset) keeps the historical
  /// budget — 7 when [`prediction_modes`](Self::prediction_modes) is at
  /// least `ComplexKeyframes` on a keyframe (or `ComplexAll` on an inter
  /// frame), else 3 — byte-identical to builds without this knob.
  /// `Some(n)` RDOs the top `n` modes instead, wherever that decision
  /// runs (clamped to 1..=13, the full DC/directional/smooth/Paeth set);
  /// the first `n / 2` come from the CDF-probability ranking and the rest
  /// are re-ranked by SATD, exactly as the historical budgets are. For
  /// still images every frame is a keyframe, so this is the still-image
  /// intra-mode dial (e.g. `Some(5)` = the top-5 midpoint between the
  /// historical 3 and 7).
  pub num_modes_rdo_override: Option<u8>,
}

/// Palette mode search policy (AV1 screen content tool).
#[derive(
  ArgEnum, Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize,
)]
pub enum PaletteMode {
  /// Never search palette mode.
  #[default]
  Off,
  /// Decide per frame with the anti-aliasing-aware screen-content
  /// detection (libaom's `estimate_screen_content_antialiasing_aware`):
  /// photographic frames skip the search (and don't signal screen content
  /// tools), screen-like frames get it.
  Auto,
  /// Always search palette mode on intra frames.
  Always,
}

/// Range of block sizes to use.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PartitionRange {
  pub(crate) min: BlockSize,
  pub(crate) max: BlockSize,
}

impl PartitionRange {
  /// Creates a new partition range with min and max partition sizes.
  ///
  /// # Panics
  ///
  /// - Panics if `max` is larger than `min`.
  /// - Panics if either `min` or `max` are not square.
  pub fn new(min: BlockSize, max: BlockSize) -> Self {
    assert!(max >= min);
    // Topdown search checks the min block size for PARTITION_SPLIT only, so
    // the min block size must be square.
    assert!(min.is_sqr());
    // Rectangular max partition sizes have not been tested.
    assert!(max.is_sqr());

    Self { min, max }
  }
}

#[cfg(test)]
impl Default for PartitionRange {
  fn default() -> Self {
    PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_64X64)
  }
}

/// Prediction modes to search.
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  Eq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
pub enum SceneDetectionSpeed {
  /// Fastest scene detection using pixel-wise comparison
  Fast,
  /// Scene detection using motion vectors and cost estimates
  Standard,
  /// Completely disable scene detection and only place keyframes
  /// at fixed intervals.
  None,
}

impl fmt::Display for SceneDetectionSpeed {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SceneDetectionSpeed::Fast => "Fast",
        SceneDetectionSpeed::Standard => "Standard",
        SceneDetectionSpeed::None => "None",
      }
    )
  }
}

/// Prediction modes to search.
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  Eq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
#[cfg_attr(test, derive(Default))]
pub enum PredictionModesSetting {
  /// Only simple prediction modes.
  #[cfg_attr(test, default)]
  Simple,
  /// Search all prediction modes on key frames and simple modes on other
  /// frames.
  ComplexKeyframes,
  /// Search all prediction modes on all frames.
  ComplexAll,
}

impl fmt::Display for PredictionModesSetting {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        PredictionModesSetting::Simple => "Simple",
        PredictionModesSetting::ComplexKeyframes => "Complex-KFs",
        PredictionModesSetting::ComplexAll => "Complex-All",
      }
    )
  }
}

/// Search level for self guided restoration
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  Eq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
pub enum SGRComplexityLevel {
  /// Search all sgr parameters
  Full,
  /// Search a reduced set of sgr parameters
  Reduced,
}

impl fmt::Display for SGRComplexityLevel {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SGRComplexityLevel::Full => "Full",
        SGRComplexityLevel::Reduced => "Reduced",
      }
    )
  }
}

/// Search level for segmentation
#[derive(
  Clone,
  Copy,
  Debug,
  PartialOrd,
  PartialEq,
  Eq,
  FromPrimitive,
  Serialize,
  Deserialize,
)]
pub enum SegmentationLevel {
  /// No segmentation is signalled.
  Disabled,
  /// Segmentation index is derived from source statistics.
  Simple,
  /// Segmentation index range is derived from source statistics.
  Complex,
  /// Search all segmentation indices.
  Full,
}

impl fmt::Display for SegmentationLevel {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    write!(
      f,
      "{}",
      match self {
        SegmentationLevel::Disabled => "Disabled",
        SegmentationLevel::Simple => "Simple",
        SegmentationLevel::Complex => "Complex",
        SegmentationLevel::Full => "Full",
      }
    )
  }
}
