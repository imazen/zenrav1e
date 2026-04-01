// Minimal stub replacing av-scenechange when the `scenechange` feature is off.
// Every frame is treated as belonging to the same scene (no keyframe insertion
// beyond forced keyframes and max-interval boundaries).

use crate::util::Pixel;
use std::collections::BTreeMap;
use std::sync::Arc;
use v_frame::frame::Frame;

pub struct SceneChangeDetector<T: Pixel> {
  pub intra_costs: Option<BTreeMap<usize, Box<[u32]>>>,
  _marker: std::marker::PhantomData<T>,
}

impl<T: Pixel> SceneChangeDetector<T> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    _dims: (usize, usize), _bit_depth: usize, _time_base: Rational32,
    _chroma_sampling: v_frame::prelude::ChromaSampling,
    _lookahead_distance: usize, _speed: SceneDetectionSpeed,
    _min_kf: usize, _max_kf: usize, _cpu: CpuFeatureLevel,
  ) -> Self {
    Self { intra_costs: None, _marker: std::marker::PhantomData }
  }

  pub fn enable_cache(&mut self) {}

  pub fn analyze_next_frame(
    &mut self, _frame_set: &[&Arc<Frame<T>>], _input_frameno: usize,
    _previous_keyframe: usize,
  ) -> bool {
    false
  }
}

pub struct Rational32;

impl Rational32 {
  pub fn new(_num: i32, _den: i32) -> Self {
    Self
  }
}

pub enum SceneDetectionSpeed {
  Fast,
  Standard,
  None,
}

pub struct CpuFeatureLevel;

impl Default for CpuFeatureLevel {
  fn default() -> Self {
    Self
  }
}
