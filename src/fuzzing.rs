// Copyright (c) 2019-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::marker::PhantomData;
use std::sync::Arc;

use libfuzzer_sys::arbitrary::{Arbitrary, Error, Unstructured};

use crate::prelude::*;

// Adding new fuzz targets
//
// 1. Add a function to this file which looks like this:
//
//    pub fn fuzz_something(data: Data) {
//      // Invoke everything you need.
//      //
//      // Your function may accept a value of any type that implements
//      // Arbitrary [1]. This is how fuzzer affects the execution—by
//      // feeding in different bytes, which result in different
//      // arbitrary values being generated.
//      // [1]: https://docs.rs/arbitrary/0.3.3/arbitrary/trait.Arbitrary.html
//      //
//      // Derive Debug for the structures you create with arbitrary data.
//    }
//
// 2. cargo fuzz add something
// 3. Copy the contents of any other .rs file from fuzz/fuzz_targets/ into the
//    newly created fuzz/fuzz_targets/something.rs and change the function
//    being called to fuzz_something.
//
// Now you can fuzz the new target with cargo fuzz.

#[derive(Debug)]
pub struct ArbitraryConfig {
  config: Config,
}

#[inline]
fn arbitrary_rational(u: &mut Unstructured<'_>) -> Result<Rational, Error> {
  // Constrain to ranges that survive every downstream signed-integer cast:
  // - av-scenechange's SceneChangeDetector takes `Rational32`, and the
  //   encoder casts time_base via `enc.time_base.den as i32`. Values
  //   >= 2^31 wrap to negative i32 (2^31 -> i32::MIN), and
  //   `num_rational::Ratio<i32>::reduce` then overflows on `0 - i32::MIN`.
  // - Other paths cast to `i64`; the i32 bound is a strict subset.
  // Also require den >= 1 so no path divides by zero. Real frame rates and
  // SARs sit far below i32::MAX — typical (1, 30), (1001, 24000), (16, 9).
  let num: u64 = u.int_in_range(0..=i32::MAX as u64)?;
  let den: u64 = u.int_in_range(1..=i32::MAX as u64)?;
  Ok(Rational::new(num, den))
}

#[inline]
fn arbitrary_color_description(
  u: &mut Unstructured<'_>,
) -> Result<Option<ColorDescription>, Error> {
  if Arbitrary::arbitrary(u)? {
    return Ok(None);
  }
  Ok(Some(ColorDescription {
    color_primaries: *u.choose(&[
      ColorPrimaries::BT709,
      ColorPrimaries::Unspecified,
      ColorPrimaries::BT470M,
      ColorPrimaries::BT470BG,
      ColorPrimaries::BT601,
      ColorPrimaries::SMPTE240,
      ColorPrimaries::GenericFilm,
      ColorPrimaries::BT2020,
      ColorPrimaries::XYZ,
      ColorPrimaries::SMPTE431,
      ColorPrimaries::SMPTE432,
      ColorPrimaries::EBU3213,
    ])?,
    transfer_characteristics: *u.choose(&[
      TransferCharacteristics::BT709,
      TransferCharacteristics::Unspecified,
      TransferCharacteristics::BT470M,
      TransferCharacteristics::BT470BG,
      TransferCharacteristics::BT601,
      TransferCharacteristics::SMPTE240,
      TransferCharacteristics::Linear,
      TransferCharacteristics::Log100,
      TransferCharacteristics::Log100Sqrt10,
      TransferCharacteristics::IEC61966,
      TransferCharacteristics::BT1361,
      TransferCharacteristics::SRGB,
      TransferCharacteristics::BT2020_10Bit,
      TransferCharacteristics::BT2020_12Bit,
      TransferCharacteristics::SMPTE2084,
      TransferCharacteristics::SMPTE428,
      TransferCharacteristics::HLG,
    ])?,
    matrix_coefficients: *u.choose(&[
      MatrixCoefficients::Identity,
      MatrixCoefficients::BT709,
      MatrixCoefficients::Unspecified,
      MatrixCoefficients::FCC,
      MatrixCoefficients::BT470BG,
      MatrixCoefficients::BT601,
      MatrixCoefficients::SMPTE240,
      MatrixCoefficients::YCgCo,
      MatrixCoefficients::BT2020NCL,
      MatrixCoefficients::BT2020CL,
      MatrixCoefficients::SMPTE2085,
      MatrixCoefficients::ChromatNCL,
      MatrixCoefficients::ChromatCL,
      MatrixCoefficients::ICtCp,
    ])?,
  }))
}

#[inline]
fn arbitrary_chromacity_point(
  u: &mut Unstructured<'_>,
) -> Result<ChromaticityPoint, Error> {
  Ok(ChromaticityPoint {
    x: Arbitrary::arbitrary(u)?,
    y: Arbitrary::arbitrary(u)?,
  })
}

#[inline]
fn arbitrary_mastering_display(
  u: &mut Unstructured<'_>,
) -> Result<Option<MasteringDisplay>, Error> {
  if Arbitrary::arbitrary(u)? {
    return Ok(None);
  }
  Ok(Some(MasteringDisplay {
    primaries: [
      arbitrary_chromacity_point(u)?,
      arbitrary_chromacity_point(u)?,
      arbitrary_chromacity_point(u)?,
    ],
    white_point: arbitrary_chromacity_point(u)?,
    max_luminance: Arbitrary::arbitrary(u)?,
    min_luminance: Arbitrary::arbitrary(u)?,
  }))
}

#[inline]
fn arbitrary_content_light(
  u: &mut Unstructured<'_>,
) -> Result<Option<ContentLight>, Error> {
  if Arbitrary::arbitrary(u)? {
    return Ok(None);
  }
  Ok(Some(ContentLight {
    max_content_light_level: Arbitrary::arbitrary(u)?,
    max_frame_average_light_level: Arbitrary::arbitrary(u)?,
  }))
}

impl Arbitrary<'_> for ArbitraryConfig {
  fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self, Error> {
    let mut enc = EncoderConfig::with_speed_preset(Arbitrary::arbitrary(u)?);
    enc.width = Arbitrary::arbitrary(u)?;
    enc.height = Arbitrary::arbitrary(u)?;
    enc.bit_depth = u.int_in_range(0..=16)?;
    enc.still_picture = Arbitrary::arbitrary(u)?;
    enc.time_base = arbitrary_rational(u)?;
    enc.min_key_frame_interval = Arbitrary::arbitrary(u)?;
    enc.max_key_frame_interval = Arbitrary::arbitrary(u)?;
    enc.reservoir_frame_delay = Arbitrary::arbitrary(u)?;
    enc.low_latency = Arbitrary::arbitrary(u)?;
    enc.quantizer = Arbitrary::arbitrary(u)?;
    enc.min_quantizer = Arbitrary::arbitrary(u)?;
    enc.bitrate = Arbitrary::arbitrary(u)?;
    enc.tile_cols = Arbitrary::arbitrary(u)?;
    enc.tile_rows = Arbitrary::arbitrary(u)?;
    enc.tiles = Arbitrary::arbitrary(u)?;
    enc.speed_settings.rdo_lookahead_frames = Arbitrary::arbitrary(u)?;
    let config = Config::new().with_encoder_config(enc).with_threads(1);
    Ok(Self { config })
  }
}

pub fn fuzz_construct_context(arbitrary: ArbitraryConfig) {
  let _: Result<Context<u16>, _> = arbitrary.config.new_context();
}

fn encode_frames(
  ctx: &mut Context<u8>, mut frames: impl Iterator<Item = Frame<u8>>,
) -> Result<(), EncoderStatus> {
  loop {
    let rv = ctx.receive_packet();
    log::debug!("ctx.receive_packet() = {:#?}", rv);

    match rv {
      Ok(_packet) => {}
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => {
        break;
      }
      Err(EncoderStatus::NeedMoreData) => {
        ctx.send_frame(frames.next().map(Arc::new))?;
      }
      Err(EncoderStatus::EnoughData) => {
        unreachable!();
      }
      Err(EncoderStatus::NotReady) => {
        unreachable!();
      }
      Err(EncoderStatus::Failure) => {
        return Err(EncoderStatus::Failure);
      }
      Err(EncoderStatus::Cancelled) => {
        return Err(EncoderStatus::Cancelled);
      }
    }
  }

  Ok(())
}

#[derive(Debug)]
pub struct ArbitraryEncoder {
  config: Config,
  frame_count: u8,
  pixels: Box<[u8]>,
}

impl Arbitrary<'_> for ArbitraryEncoder {
  fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self, Error> {
    let speed = u.int_in_range(0..=10)?;
    // The most exhaustive RDO presets (0–3) cost several× more per pixel; a
    // 256×256 speed-0 3-frame encode took ~44 s (fuzz timeout / DoS). Cap the
    // slow presets to a small frame so every preset stays within the per-input
    // time budget while still exercising the full partition/RDO search; faster
    // presets get the full 128×128 (tiling + larger partitions).
    let max_dim = if speed <= 3 { 48 } else { 128 };
    let enc = EncoderConfig {
      speed_settings: SpeedSettings::from_preset(speed),
      width: u.int_in_range(1..=max_dim)?,
      height: u.int_in_range(1..=max_dim)?,
      still_picture: Arbitrary::arbitrary(u)?,
      // Bounded to a sane frame-rate range (1–120 both ways): a pathological
      // fps drives av-scenechange's TilingInfo::from_target_tiles into a
      // `clamp(min, max)` with min > max → panic (third-party; tracked
      // separately). Real encoders never see such rates.
      time_base: Rational::new(
        u.int_in_range(1u64..=120)?,
        u.int_in_range(1u64..=120)?,
      ),
      min_key_frame_interval: u.int_in_range(0..=3)?,
      max_key_frame_interval: u.int_in_range(1..=4)?,
      low_latency: Arbitrary::arbitrary(u)?,
      quantizer: Arbitrary::arbitrary(u)?,
      min_quantizer: Arbitrary::arbitrary(u)?,
      bitrate: Arbitrary::arbitrary(u)?,
      tile_cols: u.int_in_range(0..=2)?,
      tile_rows: u.int_in_range(0..=2)?,
      tiles: u.int_in_range(0..=16)?,

      chroma_sampling: *u.choose(&[
        ChromaSampling::Cs420,
        ChromaSampling::Cs422,
        ChromaSampling::Cs444,
        ChromaSampling::Cs400,
      ])?,
      chroma_sample_position: *u.choose(&[
        ChromaSamplePosition::Unknown,
        ChromaSamplePosition::Vertical,
        ChromaSamplePosition::Colocated,
      ])?,
      pixel_range: *u.choose(&[PixelRange::Limited, PixelRange::Full])?,
      error_resilient: Arbitrary::arbitrary(u)?,
      reservoir_frame_delay: Arbitrary::arbitrary(u)?,

      sample_aspect_ratio: arbitrary_rational(u)?,
      bit_depth: 8,
      color_description: arbitrary_color_description(u)?,
      mastering_display: arbitrary_mastering_display(u)?,
      content_light: arbitrary_content_light(u)?,
      level_idx: Some(31),
      enable_timing_info: Arbitrary::arbitrary(u)?,
      switch_frame_interval: u.int_in_range(0..=3)?,
      tune: *u.choose(&[Tune::Psnr, Tune::Psychovisual])?,
      film_grain_params: None,
      ..Default::default()
    };

    let frame_count =
      if enc.still_picture { 1 } else { u.int_in_range(1..=2)? };
    if u.is_empty() {
      return Err(Error::NotEnoughData);
    }
    let pixels = u.bytes(u.len())?.to_vec().into_boxed_slice();
    let config = Config::new().with_encoder_config(enc).with_threads(1);
    Ok(Self { config, frame_count, pixels })
  }
}

pub fn fuzz_encode(arbitrary: ArbitraryEncoder) {
  let res = arbitrary.config.new_context();
  if res.is_err() {
    return;
  }
  let mut context: Context<u8> = res.unwrap();

  let mut pixels = arbitrary.pixels.iter().cycle();
  let mut frame = context.new_frame();
  let frames = (0..arbitrary.frame_count).map(|_| {
    for plane in &mut frame.planes {
      let stride = plane.cfg.stride;
      for row in plane.data_origin_mut().chunks_mut(stride) {
        for pixel in row {
          *pixel = *pixels.next().unwrap();
        }
      }
    }

    frame.clone()
  });

  let _ = encode_frames(&mut context, frames);
}

#[derive(Debug)]
pub struct DecodeTestParameters<T: Pixel> {
  w: usize,
  h: usize,
  speed: u8,
  q: usize,
  limit: usize,
  bit_depth: usize,
  chroma_sampling: ChromaSampling,
  min_keyint: u64,
  max_keyint: u64,
  switch_frame_interval: u64,
  low_latency: bool,
  error_resilient: bool,
  bitrate: i32,
  tile_cols_log2: usize,
  tile_rows_log2: usize,
  still_picture: bool,
  pixel: PhantomData<T>,
}

impl<T: Pixel> Arbitrary<'_> for DecodeTestParameters<T> {
  fn arbitrary(u: &mut Unstructured<'_>) -> Result<Self, Error> {
    let speed = u.int_in_range(0..=10)?;
    // Encode + full dav1d decode roundtrip per frame (and 10/12-bit transforms
    // when U16), so bound tighter than a bare encode: cap the slow RDO presets
    // (0–3) to 64² and the rest to 128², ≤2 frames, to stay within the per-input
    // time budget (was 271²/speed-0/3-frame → slow-unit timeouts in
    // encode_decode / encode_decode_hbd).
    let max_dim = if speed <= 3 { 64 } else { 128 };
    let mut p = Self {
      w: u.int_in_range(16..=max_dim)?,
      h: u.int_in_range(16..=max_dim)?,
      speed,
      q: u8::arbitrary(u)?.into(),
      limit: u.int_in_range(1..=2)?,
      bit_depth: 8,
      chroma_sampling: *u.choose(&[
        ChromaSampling::Cs420,
        ChromaSampling::Cs422,
        ChromaSampling::Cs444,
        ChromaSampling::Cs400,
      ])?,
      min_keyint: u.int_in_range(0..=3)?,
      max_keyint: u.int_in_range(1..=4)?,
      switch_frame_interval: u.int_in_range(0..=3)?,
      low_latency: bool::arbitrary(u)?,
      error_resilient: bool::arbitrary(u)?,
      bitrate: u16::arbitrary(u)?.into(),
      tile_cols_log2: u.int_in_range(0..=2)?,
      tile_rows_log2: u.int_in_range(0..=2)?,
      still_picture: bool::arbitrary(u)?,
      pixel: PhantomData,
    };
    if matches!(T::type_enum(), PixelType::U16) {
      p.bit_depth = *u.choose(&[8, 10, 12])?;
    }
    if !p.low_latency {
      p.switch_frame_interval = 0;
    }
    if p.still_picture {
      p.limit = 1
    }
    Ok(p)
  }
}

#[cfg(feature = "decode_test_dav1d")]
pub fn fuzz_encode_decode<T: Pixel>(p: DecodeTestParameters<T>) {
  use crate::test_encode_decode::*;

  let mut dec = get_decoder::<T>("dav1d", p.w, p.h);
  dec.encode_decode(
    true,
    p.w,
    p.h,
    p.speed,
    p.q,
    p.limit,
    p.bit_depth,
    p.chroma_sampling,
    p.min_keyint,
    p.max_keyint,
    p.switch_frame_interval,
    p.low_latency,
    p.error_resilient,
    p.bitrate,
    p.tile_cols_log2,
    p.tile_rows_log2,
    p.still_picture,
    None,
  );
}
