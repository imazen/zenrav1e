// Copyright (c) 2020-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use thiserror::Error;

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;

use crate::api::{ChromaSampling, Context, ContextInner, PixelRange};
use crate::util::Pixel;

mod encoder;
pub use encoder::*;

pub use av1_grain::*;

use crate::levels::*;

mod rate;
pub use rate::Error as RateControlError;
pub use rate::{RateControlConfig, RateControlSummary};

mod speedsettings;
pub use speedsettings::*;

pub use crate::tiling::TilingInfo;

/// Enumeration of possible invalid configuration errors.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum InvalidConfig {
  /// The width is invalid.
  #[error("invalid width {0} (expected >= 16, <= 65535)")]
  InvalidWidth(usize),
  /// The height is invalid.
  #[error("invalid height {0} (expected >= 16, <= 65535)")]
  InvalidHeight(usize),
  /// Aspect ratio numerator is invalid.
  #[error("invalid aspect ratio numerator {0} (expected > 0)")]
  InvalidAspectRatioNum(usize),
  /// Aspect ratio denominator is invalid.
  #[error("invalid aspect ratio denominator {0} (expected > 0)")]
  InvalidAspectRatioDen(usize),
  /// The render width (width adjusted based on the aspect ratio) is invalid.
  #[error("invalid render width {0} (expected >= 1, <= 65535")]
  InvalidRenderWidth(usize),
  /// The render height (height adjusted based on the aspect ratio) is invalid.
  #[error("invalid render height {0} (expected >= 1, <= 65535")]
  InvalidRenderHeight(usize),
  /// RDO lookahead frame count is invalid.
  #[error(
    "invalid rdo lookahead frames {actual} (expected <= {max} and >= {min})"
  )]
  InvalidRdoLookaheadFrames {
    /// The actual value.
    actual: usize,
    /// The maximal supported value.
    max: usize,
    /// The minimal supported value.
    min: usize,
  },
  /// Maximal keyframe interval is invalid.
  #[error("invalid max keyframe interval {actual} (expected <= {max})")]
  InvalidMaxKeyFrameInterval {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Tile columns is invalid.
  #[error("invalid tile cols {0} (expected power of 2)")]
  InvalidTileCols(usize),
  /// Tile rows is invalid.
  #[error("invalid tile rows {0} (expected power of 2)")]
  InvalidTileRows(usize),
  /// Framerate numerator is invalid.
  #[error("invalid framerate numerator {actual} (expected > 0, <= {max})")]
  InvalidFrameRateNum {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Framerate denominator is invalid.
  #[error("invalid framerate denominator {actual} (expected > 0, <= {max})")]
  InvalidFrameRateDen {
    /// The actual value.
    actual: u64,
    /// The maximal supported value.
    max: u64,
  },
  /// Reservoir frame delay is invalid.
  #[error("invalid reservoir frame delay {0} (expected >= 12, <= 131072)")]
  InvalidReservoirFrameDelay(i32),
  /// Reservoir frame delay is invalid.
  #[error(
    "invalid switch frame interval {0} (must only be used with low latency mode)"
  )]
  InvalidSwitchFrameInterval(u64),

  /// An option unsupported in still picture mode was enabled along with it.
  #[error("invalid option {0} specified with still picture mode")]
  InvalidOptionWithStillPicture(&'static str),

  /// The rate control needs a target bitrate in order to produce results
  #[error("The rate control requires a target bitrate")]
  TargetBitrateNeeded,

  /// The configuration
  #[error("Mismatch in the rate control configuration")]
  RateControlConfigurationMismatch,

  /// The color configuration mismatches AV1 constraints.
  #[error("Mismatch in the color configuration")]
  ColorConfigurationMismatch,

  /// The specified level is undefined in the current version of AV1.
  #[error("Specified level is undefined")]
  LevelUndefined,

  /// The configuration exceeded the specified level constraints.
  #[error("Constraints exceeded for specified level")]
  LevelConstraintsExceeded,

  /// The pixel count (width * height) exceeds the configured maximum.
  #[error("pixel count {actual} exceeds maximum {max}")]
  PixelCountExceeded {
    /// The actual pixel count.
    actual: u64,
    /// The configured maximum pixel count.
    max: u64,
  },

  /// The bit depth is not one of the supported AV1 values.
  #[error("invalid bit depth {0} (expected 8, 10, or 12)")]
  InvalidBitDepth(usize),

  /// `vaq_strength` is not finite or outside the supported range.
  ///
  /// The payload is the rejected value's `f64::to_bits()` representation so
  /// `InvalidConfig` can stay `Eq` (NaN bit patterns compare structurally).
  #[error(
    "invalid vaq_strength bits=0x{0:016x} (expected finite, >= 0.0, <= 4.0)"
  )]
  InvalidVaqStrength(u64),

  /// `seg_boost` is not finite or outside the supported range.
  ///
  /// The payload is the rejected value's `f64::to_bits()` representation so
  /// `InvalidConfig` can stay `Eq` (NaN bit patterns compare structurally).
  #[error(
    "invalid seg_boost bits=0x{0:016x} (expected finite, >= 0.5, <= 4.0)"
  )]
  InvalidSegBoost(u64),
}

/// Contains the encoder configuration.
#[derive(Clone, Debug, Default)]
pub struct Config {
  /// Settings which impact the produced bitstream.
  pub(crate) enc: EncoderConfig,
  /// Rate control configuration
  pub(crate) rate_control: RateControlConfig,
  /// The number of threads in the threadpool.
  pub(crate) threads: usize,
  /// Shared thread pool
  pub(crate) pool: Option<Arc<ThreadPool>>,
  #[cfg(feature = "unstable")]
  /// Number of parallel encoding slots
  pub(crate) slots: usize,
}

impl Config {
  /// Create a default configuration
  ///
  /// same as `Default::default()`
  pub fn new() -> Self {
    Config::default()
  }

  /// Set the encoder configuration
  ///
  /// `EncoderConfig` contains the settings impacting the
  /// codec features used in the produced bitstream.
  pub fn with_encoder_config(mut self, enc: EncoderConfig) -> Self {
    self.enc = enc;
    self
  }

  /// Set the number of workers in the threadpool
  ///
  /// The threadpool is shared across all the different parallel
  /// components in the encoder.
  ///
  /// If it is left unset, the encoder will use the default global
  /// threadpool provided by Rayon instead.
  pub const fn with_threads(mut self, threads: usize) -> Self {
    self.threads = threads;
    self
  }

  /// Set the rate control configuration
  ///
  /// The default configuration is single pass
  pub const fn with_rate_control(
    mut self, rate_control: RateControlConfig,
  ) -> Self {
    self.rate_control = rate_control;
    self
  }

  #[cfg(feature = "unstable")]
  /// Use the provided threadpool
  ///
  /// It takes priority over `with_threads()`
  pub fn with_thread_pool(mut self, pool: Arc<ThreadPool>) -> Self {
    self.pool = Some(pool);
    self
  }

  #[cfg(feature = "unstable")]
  /// Set the maximum number of GOPs to encode in parallel
  pub const fn with_parallel_gops(mut self, slots: usize) -> Self {
    self.slots = slots;
    self
  }
}

fn check_tile_log2(n: usize) -> bool {
  let tile_log2 = TilingInfo::tile_log2(1, n);
  if tile_log2.is_none() {
    return false;
  }
  let tile_log2 = tile_log2.unwrap();

  ((1 << tile_log2) - n) == 0 || n == 0
}

impl Config {
  pub(crate) fn new_inner<T: Pixel>(
    &self,
  ) -> Result<ContextInner<T>, InvalidConfig> {
    assert!(
      8 * std::mem::size_of::<T>() >= self.enc.bit_depth,
      "The Pixel u{} does not match the Config bit_depth {}",
      8 * std::mem::size_of::<T>(),
      self.enc.bit_depth
    );

    self.validate()?;

    let mut config = self.enc.clone();
    config.set_key_frame_interval(
      config.min_key_frame_interval,
      config.max_key_frame_interval,
    );

    // FIXME: inter unsupported with 4:2:2 and 4:4:4 chroma sampling
    let chroma_sampling = config.chroma_sampling;

    // FIXME: tx partition for intra not supported for chroma 422
    if chroma_sampling == ChromaSampling::Cs422 {
      config.speed_settings.transform.rdo_tx_decision = false;
    }

    let mut inner = ContextInner::new(&config);

    if let Some(ref s) = self.rate_control.summary {
      inner.rc_state.init_second_pass();
      inner.rc_state.setup_second_pass(s);
    }

    // First-pass parameters depend on whether second-pass is in effect.
    // So `init_first_pass` must follow `init_second_pass`.
    if self.rate_control.emit_pass_data {
      let maybe_pass1_log_base_q = (self.rate_control.summary.is_none())
        .then(|| inner.rc_state.select_pass1_log_base_q(&inner, 0));
      inner.rc_state.init_first_pass(maybe_pass1_log_base_q);
    }

    Ok(inner)
  }

  /// Create a new threadpool with this configuration if set,
  /// or return `None` if global threadpool should be used instead.
  pub(crate) fn new_thread_pool(&self) -> Option<Arc<ThreadPool>> {
    if let Some(ref p) = self.pool {
      Some(p.clone())
    } else if self.threads != 0 {
      let pool =
        ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap();
      Some(Arc::new(pool))
    } else {
      None
    }
  }

  /// Creates a [`Context`] with this configuration.
  ///
  /// # Errors
  ///
  /// Returns `InvalidConfig` if the config is invalid.
  ///
  /// # Examples
  ///
  /// ```
  /// use zenrav1e::prelude::*;
  ///
  /// # fn main() -> Result<(), InvalidConfig> {
  /// let cfg = Config::default();
  /// let ctx: Context<u8> = cfg.new_context()?;
  /// # Ok(())
  /// # }
  /// ```
  ///
  /// [`Context`]: struct.Context.html
  pub fn new_context<T: Pixel>(&self) -> Result<Context<T>, InvalidConfig> {
    let inner = self.new_inner()?;
    let config = (*inner.config).clone();
    let pool = self.new_thread_pool();

    Ok(Context { is_flushing: false, inner, pool, config })
  }

  /// Validates the configuration.
  ///
  /// # Errors
  ///
  /// - Returns `InvalidConfig` if the tiling config is invalid.
  pub fn validate(&self) -> Result<(), InvalidConfig> {
    use InvalidConfig::*;

    let config = &self.enc;

    // AV1 only supports 8/10/12 bit profiles. Many internal call sites compute
    // `bit_depth - 8`; rejecting other values here prevents usize underflow in
    // release builds and panics in debug builds. The C API enforces the same
    // constraint in `rav1e_config_set_pixel_format`; this mirrors it for the
    // pure-Rust API.
    if !matches!(config.bit_depth, 8 | 10 | 12) {
      return Err(InvalidBitDepth(config.bit_depth));
    }

    if (config.still_picture && config.width < 1)
      || (!config.still_picture && config.width < 16)
      || config.width > u16::MAX as usize
    {
      return Err(InvalidWidth(config.width));
    }
    if (config.still_picture && config.height < 1)
      || (!config.still_picture && config.height < 16)
      || config.height > u16::MAX as usize
    {
      return Err(InvalidHeight(config.height));
    }

    if config.sample_aspect_ratio.num == 0 {
      return Err(InvalidAspectRatioNum(
        config.sample_aspect_ratio.num as usize,
      ));
    }
    if config.sample_aspect_ratio.den == 0 {
      return Err(InvalidAspectRatioDen(
        config.sample_aspect_ratio.den as usize,
      ));
    }

    let (render_width, render_height) = config.render_size();
    if render_width == 0 || render_width > u16::MAX as usize {
      return Err(InvalidRenderWidth(render_width));
    }
    if render_height == 0 || render_height > u16::MAX as usize {
      return Err(InvalidRenderHeight(render_height));
    }

    // Use u64 arithmetic for pixel count to avoid overflow on 32-bit platforms
    let pixel_count = (config.width as u64) * (config.height as u64);
    if config.max_pixel_count > 0 && pixel_count > config.max_pixel_count {
      return Err(PixelCountExceeded {
        actual: pixel_count,
        max: config.max_pixel_count,
      });
    }

    if config.speed_settings.rdo_lookahead_frames > MAX_RDO_LOOKAHEAD_FRAMES
      || config.speed_settings.rdo_lookahead_frames < 1
    {
      return Err(InvalidRdoLookaheadFrames {
        actual: config.speed_settings.rdo_lookahead_frames,
        max: MAX_RDO_LOOKAHEAD_FRAMES,
        min: 1,
      });
    }
    if config.max_key_frame_interval > MAX_MAX_KEY_FRAME_INTERVAL {
      return Err(InvalidMaxKeyFrameInterval {
        actual: config.max_key_frame_interval,
        max: MAX_MAX_KEY_FRAME_INTERVAL,
      });
    }

    if !check_tile_log2(config.tile_cols) {
      return Err(InvalidTileCols(config.tile_cols));
    }
    if !check_tile_log2(config.tile_rows) {
      return Err(InvalidTileRows(config.tile_rows));
    }

    if config.time_base.num == 0 || config.time_base.num > u32::MAX as u64 {
      return Err(InvalidFrameRateNum {
        actual: config.time_base.num,
        max: u32::MAX as u64,
      });
    }
    if config.time_base.den == 0 || config.time_base.den > u32::MAX as u64 {
      return Err(InvalidFrameRateDen {
        actual: config.time_base.den,
        max: u32::MAX as u64,
      });
    }

    if let Some(delay) = config.reservoir_frame_delay
      && !(12..=131_072).contains(&delay)
    {
      return Err(InvalidReservoirFrameDelay(delay));
    }

    if config.switch_frame_interval > 0 && !config.low_latency {
      return Err(InvalidSwitchFrameInterval(config.switch_frame_interval));
    }

    if config.enable_timing_info && config.still_picture {
      return Err(InvalidOptionWithStillPicture("enable_timing_info"));
    }

    // <https://aomediacodec.github.io/av1-spec/#color-config-syntax>
    if let Some(color_description) = config.color_description
      && config.chroma_sampling != ChromaSampling::Cs400
      && color_description.is_srgb_triple()
    {
      if config.pixel_range != PixelRange::Full {
        return Err(ColorConfigurationMismatch);
      }
      if config.chroma_sampling != ChromaSampling::Cs444 {
        return Err(ColorConfigurationMismatch);
      }
    }

    if let Some(level_idx) = config.level_idx {
      if level_idx > 31 {
        return Err(LevelUndefined);
      }
      if level_idx < 31 {
        if !AV1_LEVEL_DEFINED[level_idx as usize] {
          return Err(LevelUndefined);
        }
        if pixel_count > AV1_LEVEL_MAX_PIC_SIZE[level_idx as usize] as u64 {
          return Err(LevelConstraintsExceeded);
        }
        if config.width > AV1_LEVEL_MAX_H_SIZE[level_idx as usize] {
          return Err(LevelConstraintsExceeded);
        }
        if config.height > AV1_LEVEL_MAX_V_SIZE[level_idx as usize] {
          return Err(LevelConstraintsExceeded);
        }
        if (pixel_count * config.time_base.num).div_ceil(config.time_base.den)
          > AV1_LEVEL_MAX_DISPLAY_RATE[level_idx as usize] as u64
        {
          return Err(LevelConstraintsExceeded);
        }
      }
    }

    // Reject NaN / Inf / out-of-range tuning knobs. Without this, NaN
    // propagates into VAQ / segmentation arithmetic and produces undefined
    // segmentation mappings or zero/negative QP offsets that confuse RDO.
    // Range bounds are conservative: 0.0 disables VAQ, > 4.0 produces
    // pathological bit redistribution; seg_boost < 0.5 inverts the boost.
    if !config.vaq_strength.is_finite()
      || !(0.0..=4.0).contains(&config.vaq_strength)
    {
      return Err(InvalidVaqStrength(config.vaq_strength.to_bits()));
    }
    if !config.seg_boost.is_finite()
      || !(0.5..=4.0).contains(&config.seg_boost)
    {
      return Err(InvalidSegBoost(config.seg_boost.to_bits()));
    }

    // TODO: add more validation
    let rc = &self.rate_control;

    if (rc.emit_pass_data || rc.summary.is_some()) && config.bitrate == 0 {
      return Err(TargetBitrateNeeded);
    }

    Ok(())
  }

  /// Provide the tiling information for the current Config
  ///
  /// Useful for reporting and debugging.
  ///
  /// # Errors
  ///
  /// - Returns `InvalidConfig` if the tiling config is invalid.
  pub fn tiling_info(&self) -> Result<TilingInfo, InvalidConfig> {
    self.validate()?;

    let seq = crate::encoder::Sequence::new(&self.enc);

    Ok(seq.tiling)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn base_config() -> Config {
    let enc = EncoderConfig {
      width: 64,
      height: 64,
      bit_depth: 8,
      ..EncoderConfig::default()
    };
    Config::new().with_encoder_config(enc)
  }

  #[test]
  fn rejects_invalid_bit_depth() {
    for bd in [0usize, 4, 6, 7, 9, 11, 13, 16, 32] {
      let mut cfg = base_config();
      cfg.enc.bit_depth = bd;
      match cfg.validate() {
        Err(InvalidConfig::InvalidBitDepth(got)) => assert_eq!(got, bd),
        other => panic!("bit_depth {bd} should be rejected, got {other:?}"),
      }
    }
  }

  #[test]
  fn accepts_supported_bit_depths() {
    for bd in [8usize, 10, 12] {
      let mut cfg = base_config();
      cfg.enc.bit_depth = bd;
      assert!(
        cfg.validate().is_ok(),
        "bit_depth {bd} should be accepted: {:?}",
        cfg.validate()
      );
    }
  }

  #[test]
  fn rejects_non_finite_vaq_strength() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let mut cfg = base_config();
      cfg.enc.vaq_strength = v;
      assert!(
        matches!(cfg.validate(), Err(InvalidConfig::InvalidVaqStrength(_))),
        "vaq_strength={v} should be rejected"
      );
    }
  }

  #[test]
  fn rejects_out_of_range_vaq_strength() {
    for v in [-0.1f64, -1.0, 4.01, 100.0] {
      let mut cfg = base_config();
      cfg.enc.vaq_strength = v;
      assert!(
        matches!(cfg.validate(), Err(InvalidConfig::InvalidVaqStrength(_))),
        "vaq_strength={v} should be rejected"
      );
    }
  }

  #[test]
  fn rejects_non_finite_seg_boost() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
      let mut cfg = base_config();
      cfg.enc.seg_boost = v;
      assert!(
        matches!(cfg.validate(), Err(InvalidConfig::InvalidSegBoost(_))),
        "seg_boost={v} should be rejected"
      );
    }
  }

  #[test]
  fn rejects_out_of_range_seg_boost() {
    for v in [-0.1f64, 0.0, 0.49, 4.01, 100.0] {
      let mut cfg = base_config();
      cfg.enc.seg_boost = v;
      assert!(
        matches!(cfg.validate(), Err(InvalidConfig::InvalidSegBoost(_))),
        "seg_boost={v} should be rejected"
      );
    }
  }

  #[test]
  fn accepts_default_float_knobs() {
    let cfg = base_config();
    assert!(cfg.validate().is_ok());
  }
}
