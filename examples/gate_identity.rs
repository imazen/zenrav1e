// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! `gate-identity`: byte-exactness of the off-state, as an executable gate.
//!
//! This is invariant A1 of the engineering baseline (zenavif
//! `docs/ENGINEERING_BASELINE.md`): every encoder knob whose default is
//! "off" produces bitstreams byte-identical to the knob's absence. The gate
//! has two halves:
//!
//! 1. **Pinned baselines** — the default-config bitstream (and the
//!    `Tune::Ssimulacra2` bitstream) for every cell of a pinned grid is
//!    hashed and compared against `tests/gate_identity_pins.tsv`. Any drift
//!    means a behavioral change: either an unintended regression (fix it)
//!    or an intended change (re-pin with `--pin` in the same commit, so the
//!    diff documents the change). This is the runnable form of the
//!    program's old-binary-vs-new-binary 27/27-md5 checks.
//! 2. **Neutral arms** — knobs that document an explicit neutral value
//!    ("`Some(x)` is byte-identical to `None`", "inert without its
//!    enabling precondition") are set to that value and the bytes must
//!    equal the same-run baseline exactly. A failing arm is a contract
//!    violation in the knob itself, not pin drift.
//!
//! Pins are platform-tagged (`os-arch`): RDO uses libm functions whose
//! last-ulp behavior may differ across platforms, so a pin is only
//! comparable on the platform that produced it. The committed pins are
//! `linux-x86_64` (the dev box + CI). Running on an unpinned platform
//! fails loudly; the caller (justfile / CI) decides where the gate runs.
//!
//! Usage (see justfile `gate-identity`):
//!   cargo run --release --example gate_identity            # full grid
//!   cargo run --release --example gate_identity -- --ci    # reduced grid
//!   cargo run --release --example gate_identity -- --pin   # re-pin
//!   cargo run --release --example gate_identity -- --emit-y4m DIR

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use zenrav1e::config::SpeedSettings;
use zenrav1e::prelude::*;

// ---------------------------------------------------------------------------
// Pinned deterministic content (integer-only generation: no libm anywhere in
// the inputs, so the *inputs* are identical on every platform).
// ---------------------------------------------------------------------------

/// Minimal deterministic PRNG (64-bit LCG, top-bits output). Stable by
/// construction — never replace with `DefaultHasher`/`rand`, whose output
/// may change across Rust/crate releases and would silently re-roll the
/// pinned content.
struct Lcg(u64);

impl Lcg {
  fn new(seed: u64) -> Self {
    Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
  }

  fn next_u32(&mut self) -> u32 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    (self.0 >> 33) as u32
  }
}

/// One pinned 8-bit 4:2:0 image (chroma planes at ceil dimensions).
struct PinnedImage {
  name: &'static str,
  w: usize,
  h: usize,
  y: Vec<u8>,
  u: Vec<u8>,
  v: Vec<u8>,
}

fn chroma_dims(w: usize, h: usize) -> (usize, usize) {
  (w.div_ceil(2), h.div_ceil(2))
}

/// Photo-like content: smooth gradients + low-frequency bumps + fine noise.
/// Must NOT look like screen content (the palette/intraBC `Auto` detection
/// arms rely on this image reading as photographic).
fn gen_photo() -> PinnedImage {
  let (w, h) = (128usize, 128usize);
  let (cw, ch) = chroma_dims(w, h);
  let mut rng = Lcg::new(0x0001);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      // Smooth (continuous) low-frequency structure only — no block-aligned
      // discontinuities, which the AA-aware screen-content detection would
      // legitimately classify as screen content. Dense per-pixel noise
      // keeps the statistics photographic.
      let base = 48 + (i + j) / 2;
      let bump = (i * i) / 600 + (j * j) / 540 + (i * j) / 800;
      let noise = (rng.next_u32() % 21) as usize;
      y[j * w + i] = (base + bump + noise).min(235) as u8;
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      u[j * cw + i] =
        (104 + (i + 2 * j) / 3 + (rng.next_u32() % 9) as usize).min(240) as u8;
      v[j * cw + i] =
        (116 + i / 2 + (i * j) / 340 + (rng.next_u32() % 9) as usize).min(240)
          as u8;
    }
  }
  PinnedImage { name: "photo", w, h, y, u, v }
}

/// Screen-like content: flat 16x16 patches from a small palette, 1-px
/// separator lines, and a repeated 8x8 glyph (intraBC/hash-friendly).
fn gen_screen() -> PinnedImage {
  let (w, h) = (128usize, 128usize);
  let (cw, ch) = chroma_dims(w, h);
  let palette = [24u8, 235, 80, 160, 48, 200, 112, 16];
  let mut rng = Lcg::new(0x0002);
  let mut patch = vec![0u8; (w / 16) * (h / 16)];
  for p in patch.iter_mut() {
    *p = palette[(rng.next_u32() % 8) as usize];
  }
  let glyph = |i: usize, j: usize| -> bool {
    let (gi, gj) = (i % 8, j % 8);
    gi == 1 || gj == 6 || (gi == gj && gi < 5)
  };
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      let mut px = patch[(j / 16) * (w / 16) + i / 16];
      if i % 16 == 0 || j % 16 == 0 {
        px = 0; // hard separator lines
      } else if (64..96).contains(&i) && glyph(i, j) {
        px = 255; // repeated glyph band
      }
      y[j * w + i] = px;
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      let p = patch[(j * 2 / 16) * (w / 16) + (i * 2) / 16];
      u[j * cw + i] = if p > 128 { 96 } else { 160 };
      v[j * cw + i] = if p > 128 { 176 } else { 72 };
    }
  }
  PinnedImage { name: "screen", w, h, y, u, v }
}

/// Mixed content at odd dimensions (edge-superblock + sliver coverage):
/// gradient/noise left half, sharp checkerboard right half.
fn gen_mixed() -> PinnedImage {
  let (w, h) = (131usize, 97usize);
  let (cw, ch) = chroma_dims(w, h);
  let mut rng = Lcg::new(0x0003);
  let mut y = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      y[j * w + i] = if i < w / 2 {
        (40 + i + j / 2 + (rng.next_u32() % 9) as usize).min(230) as u8
      } else if ((i / 4) + (j / 4)) % 2 == 0 {
        220
      } else {
        35
      };
    }
  }
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      u[j * cw + i] = if i * 2 < w / 2 { 132 } else { 88 };
      v[j * cw + i] = (100 + j).min(200) as u8;
    }
  }
  PinnedImage { name: "mixed", w, h, y, u, v }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encode one still picture, single-threaded, returning the concatenated
/// packet bytes. `mutate` edits the fully-constructed config (preset already
/// applied); `neutral_hints` optionally attaches an all-1.0 `FrameHints` map.
fn encode_cell(
  img: &PinnedImage, speed: u8, quantizer: usize, tune: Tune,
  mutate: &dyn Fn(&mut EncoderConfig), neutral_hints: bool,
) -> Vec<u8> {
  let mut enc = EncoderConfig {
    width: img.w,
    height: img.h,
    speed_settings: SpeedSettings::from_preset(speed),
    quantizer,
    min_quantizer: quantizer as u8,
    still_picture: true,
    chroma_sampling: ChromaSampling::Cs420,
    tune,
    ..Default::default()
  };
  mutate(&mut enc);
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> =
    cfg.new_context().expect("gate_identity: context creation failed");

  let mut frame = ctx.new_frame();
  let (cw, _ch) = chroma_dims(img.w, img.h);
  frame.planes[0].copy_from_raw_u8(&img.y, img.w, 1);
  frame.planes[1].copy_from_raw_u8(&img.u, cw, 1);
  frame.planes[2].copy_from_raw_u8(&img.v, cw, 1);

  if neutral_hints {
    let sb_cols = img.w.div_ceil(8).div_ceil(8);
    let sb_rows = img.h.div_ceil(8).div_ceil(8);
    let hints = FrameHints::new()
      .with_sb_q_scale(vec![1.0f32; sb_cols * sb_rows].into_boxed_slice());
    let params = FrameParameters {
      frame_type_override: FrameTypeOverride::No,
      opaque: None,
      t35_metadata: Box::new([]),
      frame_hints: Some(Arc::new(hints)),
    };
    ctx.send_frame((frame, params)).expect("send_frame(hints) failed");
  } else {
    ctx.send_frame(frame).expect("send_frame failed");
  }
  ctx.flush();

  let mut out = Vec::new();
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => out.extend_from_slice(&pkt.data),
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!("gate_identity: encode error: {e:?}"),
    }
  }
  assert!(!out.is_empty(), "gate_identity: empty bitstream");
  out
}

/// Decode a cell's concatenated packet bytes with rav1d-safe and require a
/// frame of the expected size. Returns the failure reason on any miss.
fn decodes_to_frame(bytes: &[u8], w: usize, h: usize) -> Result<(), String> {
  // A decoder panic is a gate failure attributed to the cell, not an abort
  // of the whole run (rav1d-safe 0.5.7 index-panics in its ARM loop-
  // restoration SIMD on some cells — the gate must still report WHICH).
  match std::panic::catch_unwind(|| decode_inner(bytes, w, h)) {
    Ok(r) => r,
    Err(payload) => {
      let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "<non-string panic>".into());
      Err(format!("decoder panicked: {msg}"))
    }
  }
}

fn decode_inner(bytes: &[u8], w: usize, h: usize) -> Result<(), String> {
  let mut dec = rav1d_safe::Decoder::new().map_err(|e| format!("{e:?}"))?;
  let mut fr = dec.decode(bytes).map_err(|e| format!("decode: {e:?}"))?;
  if fr.is_none() {
    fr = dec.flush().map_err(|e| format!("flush: {e:?}"))?.drain(..).next();
  }
  let frame = fr.ok_or_else(|| "no frame produced".to_string())?;
  let got = (frame.width() as usize, frame.height() as usize);
  if got != (w, h) {
    return Err(format!("decoded {got:?}, expected {:?}", (w, h)));
  }
  Ok(())
}

fn fnv1a64(data: &[u8]) -> u64 {
  let mut h = 0xcbf2_9ce4_8422_2325u64;
  for &b in data {
    h ^= u64::from(b);
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
  }
  h
}

// ---------------------------------------------------------------------------
// Job plan
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum TuneTag {
  /// Default tune (the preset's `Tune::Psychovisual`).
  P,
  /// `Tune::Ssimulacra2` (the flagship shipping tune).
  T,
}

impl TuneTag {
  fn tune(self) -> Tune {
    match self {
      TuneTag::P => Tune::Psychovisual,
      TuneTag::T => Tune::Ssimulacra2,
    }
  }
  fn label(self) -> &'static str {
    match self {
      TuneTag::P => "P",
      TuneTag::T => "T",
    }
  }
}

/// What an arm's bytes are contracted to do relative to other encodes of
/// the same cell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cmp {
  /// Byte-identical to the same-tune baseline (a documented-neutral knob
  /// state).
  EqBaseline,
  /// Byte-identical to another arm of the same cell (e.g. a sub-knob that
  /// is inert while its enabling gate did not fire).
  EqArm(&'static str),
  /// Must DIFFER from the baseline (liveness: the armed path really codes
  /// differently) — and the bytes are pinned so the armed path is also
  /// drift-gated. Note: for still pictures the palette-Off baseline
  /// signals `allow_screen_content_tools=1` (stock-rav1e inherited:
  /// `force_screen_content_tools=2` and `allow` defaults to `force`), so
  /// `PaletteMode::Auto` on photographic content DIFFERS from Off by
  /// dropping that signaling — that is the documented design, not a bug.
  NeBaseline,
}

/// A neutral-arm definition: a documented byte-inert config mutation, or a
/// pinned+liveness-checked armed path (see [`Cmp`]).
struct Arm {
  name: &'static str,
  tune: TuneTag,
  /// Restrict to one pinned image (detection-gated arms must run on
  /// content with the right detector outcome).
  only_img: Option<&'static str>,
  /// Attach the all-1.0 FrameHints map instead of a config mutation.
  hints: bool,
  cmp: Cmp,
  mutate: fn(u8, &mut EncoderConfig),
}

fn no_mutate(_s: u8, _e: &mut EncoderConfig) {}

fn arms() -> Vec<Arm> {
  vec![
    Arm {
      // intrabc_hash is documented inert while `intrabc` is off.
      name: "hash_off_inert",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.speed_settings.prediction.intrabc_hash = false,
    },
    Arm {
      // variance_boost_* are documented effective only under
      // Tune::Ssimulacra2 — under the default tune they must be inert.
      name: "vb_offtune_inert",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.variance_boost_strength = Some(3.0),
    },
    Arm {
      name: "vbdeep_offtune_inert",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.variance_boost_deep = Some((3.0, 6)),
    },
    Arm {
      // ssim_rdmult_strength is documented effective only under ss2.
      name: "ssimrd_offtune_inert",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.ssim_rdmult_strength = Some(1.0),
    },
    Arm {
      // Some(preset value) must equal None (documented fallback).
      name: "txsize_fallback",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| {
        let d = e.speed_settings.transform.rdo_tx_decision;
        e.speed_settings.transform.rdo_tx_size_override = Some(d);
      },
    },
    Arm {
      name: "txtype_fallback",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| {
        let d = e.speed_settings.transform.rdo_tx_decision;
        e.speed_settings.transform.rdo_tx_type_override = Some(d);
      },
    },
    Arm {
      // None runs the full walk (MAX_TX_DEPTH = 2); Some(2) must match.
      name: "txdepth_full",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| {
        e.speed_settings.transform.rdo_tx_size_depth = Some(2);
      },
    },
    Arm {
      // Some(historical budget) must equal None: 7 under
      // ComplexKeyframes (s2..s6 presets), 3 under Simple (s7+).
      name: "modes_hist",
      tune: TuneTag::P,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |s, e| {
        let hist = if s >= 7 { 3 } else { 7 };
        e.speed_settings.prediction.num_modes_rdo_override = Some(hist);
      },
    },
    Arm {
      // PaletteMode::Auto on photographic content: the detection must
      // classify it photo, skip the palette search, and DROP the
      // screen-content signaling the Off-still baseline carries — so the
      // bytes must differ from baseline (see Cmp::NeBaseline) and the
      // Auto-photo path is pinned in its own right.
      name: "palette_auto_photo",
      tune: TuneTag::P,
      only_img: Some("photo"),
      hints: false,
      cmp: Cmp::NeBaseline,
      mutate: |_s, e| e.speed_settings.prediction.palette = PaletteMode::Auto,
    },
    Arm {
      // intraBC behind the Auto detection gate on photo content: the
      // stricter intraBC criterion must not fire, so arming intrabc adds
      // NOTHING on top of palette=Auto (the "never ship intrabc without
      // the Auto gate" contract).
      name: "ibc_autogate_photo",
      tune: TuneTag::P,
      only_img: Some("photo"),
      hints: false,
      cmp: Cmp::EqArm("palette_auto_photo"),
      mutate: |_s, e| {
        e.speed_settings.prediction.palette = PaletteMode::Auto;
        e.speed_settings.prediction.intrabc = true;
      },
    },
    Arm {
      // Liveness + drift pin of the ARMED palette path: on screen
      // content the detection fires and the palette search must actually
      // change the bitstream.
      name: "palette_auto_screen",
      tune: TuneTag::P,
      only_img: Some("screen"),
      hints: false,
      cmp: Cmp::NeBaseline,
      mutate: |_s, e| e.speed_settings.prediction.palette = PaletteMode::Auto,
    },
    Arm {
      // Liveness + drift pin of the ARMED intraBC(+hash) path on screen
      // content (rides the same Auto detection).
      name: "ibc_auto_screen",
      tune: TuneTag::P,
      only_img: Some("screen"),
      hints: false,
      cmp: Cmp::NeBaseline,
      mutate: |_s, e| {
        e.speed_settings.prediction.palette = PaletteMode::Auto;
        e.speed_settings.prediction.intrabc = true;
      },
    },
    Arm {
      // FrameHints with an all-1.0 sb_q_scale map == no hints.
      name: "hints_neutral",
      tune: TuneTag::P,
      only_img: None,
      hints: true,
      cmp: Cmp::EqBaseline,
      mutate: no_mutate,
    },
    Arm {
      // Under ss2: strength 0.0 is documented "off" == None.
      name: "t_ssimrd_zero",
      tune: TuneTag::T,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.ssim_rdmult_strength = Some(0.0),
    },
    Arm {
      // Under ss2: Some(1.0) == None (the fitted constant is 1.0).
      name: "t_vb_fitted",
      tune: TuneTag::T,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.variance_boost_strength = Some(1.0),
    },
    Arm {
      // Under ss2: a deep ramp whose deep strength equals the base
      // strength is a flat ramp == None.
      name: "t_vbdeep_flat",
      tune: TuneTag::T,
      only_img: None,
      hints: false,
      cmp: Cmp::EqBaseline,
      mutate: |_s, e| e.variance_boost_deep = Some((1.0, 6)),
    },
  ]
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Job {
  cell: String,
  img: usize,
  speed: u8,
  q: usize,
  tune: TuneTag,
  arm: Option<usize>,
}

fn platform() -> String {
  format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

fn pins_path() -> std::path::PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gate_identity_pins.tsv")
}

fn emit_y4m(dir: &Path, images: &[PinnedImage]) {
  fs::create_dir_all(dir).expect("create y4m dir");
  for img in images {
    let path = dir.join(format!("{}.y4m", img.name));
    let mut f = fs::File::create(&path).expect("create y4m");
    writeln!(f, "YUV4MPEG2 W{} H{} F25:1 Ip A1:1 C420jpeg", img.w, img.h)
      .unwrap();
    writeln!(f, "FRAME").unwrap();
    f.write_all(&img.y).unwrap();
    f.write_all(&img.u).unwrap();
    f.write_all(&img.v).unwrap();
    println!("wrote {}", path.display());
  }
}

fn main() {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let pin = args.iter().any(|a| a == "--pin");
  let ci = args.iter().any(|a| a == "--ci");
  let images = [gen_photo(), gen_screen(), gen_mixed()];

  if let Some(i) = args.iter().position(|a| a == "--emit-y4m") {
    let dir = args.get(i + 1).expect("--emit-y4m needs a directory");
    emit_y4m(Path::new(dir), &images);
    return;
  }

  // The pinned grid. --ci trims speeds/quantizers but keeps all three
  // images so every arm stays exercised (photo for the detection-gated
  // arms, screen for the armed-path liveness pins, mixed for
  // odd-dimension coverage); s2 and s8 straddle the ComplexKeyframes/
  // Simple and rdo_tx_decision preset boundaries.
  let (img_idx, speeds, quants): (Vec<usize>, Vec<u8>, Vec<usize>) = if ci {
    (vec![0, 1, 2], vec![2, 8], vec![140])
  } else {
    (vec![0, 1, 2], vec![2, 6, 8], vec![60, 140, 220])
  };

  let arm_defs = arms();
  let mut jobs: Vec<Job> = Vec::new();
  for &ii in &img_idx {
    for &s in &speeds {
      for &q in &quants {
        for tune in [TuneTag::P, TuneTag::T] {
          jobs.push(Job {
            cell: format!(
              "{}/{}/s{}/q{}",
              tune.label(),
              images[ii].name,
              s,
              q
            ),
            img: ii,
            speed: s,
            q,
            tune,
            arm: None,
          });
        }
        for (ai, arm) in arm_defs.iter().enumerate() {
          if let Some(only) = arm.only_img
            && images[ii].name != only
          {
            continue;
          }
          jobs.push(Job {
            cell: format!(
              "{}/{}/s{}/q{}",
              arm.tune.label(),
              images[ii].name,
              s,
              q
            ),
            img: ii,
            speed: s,
            q,
            tune: arm.tune,
            arm: Some(ai),
          });
        }
      }
    }
  }

  // Encode all jobs on a small worker pool (each encode is itself
  // single-threaded and fully deterministic).
  let results: Vec<OnceLock<Vec<u8>>> =
    jobs.iter().map(|_| OnceLock::new()).collect();
  let next = AtomicUsize::new(0);
  let workers = std::thread::available_parallelism()
    .map(std::num::NonZeroUsize::get)
    .unwrap_or(4)
    .min(8);
  let t0 = std::time::Instant::now();
  std::thread::scope(|scope| {
    for _ in 0..workers {
      scope.spawn(|| {
        loop {
          let i = next.fetch_add(1, Ordering::Relaxed);
          if i >= jobs.len() {
            break;
          }
          let job = &jobs[i];
          let img = &images[job.img];
          let bytes = match job.arm {
            None => encode_cell(
              img,
              job.speed,
              job.q,
              job.tune.tune(),
              &no_op,
              false,
            ),
            Some(ai) => {
              let arm = &arm_defs[ai];
              let s = job.speed;
              let f = arm.mutate;
              encode_cell(
                img,
                job.speed,
                job.q,
                job.tune.tune(),
                &move |e: &mut EncoderConfig| f(s, e),
                arm.hints,
              )
            }
          };
          results[i].set(bytes).expect("job encoded twice");
        }
      });
    }
  });
  let enc_secs = t0.elapsed().as_secs_f32();

  // Decode gate (zenrav1e#41): a byte pin blesses whatever the encoder
  // emitted — re-pinning after a desync would lock corrupt bytes forever
  // (the zenjpeg#196 hash-lock failure). Every cell must at least decode to
  // a frame of the right size before it is compared or pinned. This catches
  // hard parse failures only; recon divergence (a decoder accepting a
  // desynced stream) is gate-recon's job and stays the deeper local check.
  let t1 = std::time::Instant::now();
  let mut undecodable = 0usize;
  for (i, job) in jobs.iter().enumerate() {
    let img = &images[job.img];
    if let Err(msg) = decodes_to_frame(results[i].get().unwrap(), img.w, img.h)
    {
      undecodable += 1;
      println!("UNDECODABLE {}  {msg}", job.cell);
      // GATE_IDENTITY_DUMP=<dir>: keep the offending bytes (raw OBU
      // stream) for triage with an external decoder.
      if let Some(dir) = std::env::var_os("GATE_IDENTITY_DUMP") {
        let dir = Path::new(&dir);
        fs::create_dir_all(dir).expect("create dump dir");
        let arm = job.arm.map(|ai| arm_defs[ai].name).unwrap_or("base");
        let name = format!("{}_{arm}.obu", job.cell.replace('/', "_"));
        fs::write(dir.join(&name), results[i].get().unwrap())
          .expect("write dump");
      }
    }
  }
  assert!(
    undecodable == 0,
    "gate_identity: {undecodable} cell(s) produced undecodable bitstreams — \
     refusing to compare or pin them"
  );
  let dec_secs = t1.elapsed().as_secs_f32();

  // Index baseline and arm results, then check every arm's contract.
  let mut baselines: BTreeMap<&str, &Vec<u8>> = BTreeMap::new();
  let mut arm_bytes: BTreeMap<(usize, &str), &Vec<u8>> = BTreeMap::new();
  for (i, job) in jobs.iter().enumerate() {
    match job.arm {
      None => {
        baselines.insert(&job.cell, results[i].get().unwrap());
      }
      Some(ai) => {
        arm_bytes.insert((ai, &job.cell), results[i].get().unwrap());
      }
    }
  }
  let arm_index = |name: &str| -> usize {
    arm_defs.iter().position(|a| a.name == name).expect("unknown arm name")
  };

  let mut arm_fail = 0usize;
  let mut arm_ok = 0usize;
  for (i, job) in jobs.iter().enumerate() {
    let Some(ai) = job.arm else { continue };
    let arm = &arm_defs[ai];
    let got = results[i].get().unwrap();
    let (pass, expect) = match arm.cmp {
      Cmp::EqBaseline => {
        let base = baselines[job.cell.as_str()];
        (got == base, "== baseline")
      }
      Cmp::EqArm(other) => {
        let partner = arm_bytes[&(arm_index(other), job.cell.as_str())];
        (got == partner, "== partner arm")
      }
      Cmp::NeBaseline => {
        let base = baselines[job.cell.as_str()];
        (got != base, "!= baseline (liveness)")
      }
    };
    if pass {
      arm_ok += 1;
    } else {
      arm_fail += 1;
      println!(
        "ARM FAIL  {:<22} {}  expected {expect} ({} bytes)",
        arm.name,
        job.cell,
        got.len(),
      );
    }
  }
  // Informational (not a gated contract): whether intraBC actually coded
  // anything beyond the palette arm on the screen image.
  {
    let (pi, bi) =
      (arm_index("palette_auto_screen"), arm_index("ibc_auto_screen"));
    let mut ibc_live = 0usize;
    let mut ibc_cells = 0usize;
    for (&(ai, cell), bytes) in &arm_bytes {
      if ai == bi {
        ibc_cells += 1;
        if arm_bytes.get(&(pi, cell)).is_none_or(|p| p != bytes) {
          ibc_live += 1;
        }
      }
    }
    println!(
      "info: intraBC changed bytes vs palette-only on {ibc_live}/{ibc_cells} \
       screen cells"
    );
  }

  // Pin handling.
  let plat = platform();
  let path = pins_path();
  let mut kept: Vec<String> = Vec::new(); // other-platform rows on --pin
  let mut pins: BTreeMap<String, (usize, u64)> = BTreeMap::new();
  if path.exists() {
    for line in fs::read_to_string(&path).unwrap().lines() {
      if line.starts_with('#') || line.trim().is_empty() {
        continue;
      }
      let f: Vec<&str> = line.split('\t').collect();
      if f.len() != 4 {
        continue;
      }
      if f[0] == plat {
        pins.insert(
          f[1].to_string(),
          (f[2].parse().unwrap(), u64::from_str_radix(f[3], 16).unwrap()),
        );
      } else {
        kept.push(line.to_string());
      }
    }
  }

  let mut drift = 0usize;
  let mut pinned_ok = 0usize;
  let mut unpinned = 0usize;
  let mut new_rows: Vec<String> = Vec::new();
  for (i, job) in jobs.iter().enumerate() {
    // Pin baselines AND the NeBaseline (armed-path) arms; Eq* arms are
    // already byte-covered by their equality contract.
    let key = match job.arm {
      None => job.cell.clone(),
      Some(ai) if arm_defs[ai].cmp == Cmp::NeBaseline => {
        format!("{}+{}", arm_defs[ai].name, job.cell)
      }
      Some(_) => continue,
    };
    let bytes = results[i].get().unwrap();
    let (len, hash) = (bytes.len(), fnv1a64(bytes));
    new_rows.push(format!("{plat}\t{key}\t{len}\t{hash:016x}"));
    match pins.get(&key) {
      Some(&(plen, phash)) if plen == len && phash == hash => pinned_ok += 1,
      Some(&(plen, phash)) => {
        drift += 1;
        println!(
          "PIN DRIFT {key}  pinned len={} fnv={:016x}  got len={} fnv={:016x}",
          plen, phash, len, hash
        );
      }
      None => unpinned += 1,
    }
  }

  if pin {
    let mut out = String::new();
    out.push_str(
      "# gate_identity baseline pins: default-config (P/...) and \
       Tune::Ssimulacra2 (T/...) bitstream fingerprints per cell.\n\
       # Regenerate with: just gate-identity-pin (intentional behavioral \
       changes only — review the diff in the same commit).\n\
       # platform\tcell\tlen\tfnv1a64\n",
    );
    for l in kept {
      let _ = writeln!(out, "{l}");
    }
    for l in &new_rows {
      let _ = writeln!(out, "{l}");
    }
    fs::write(&path, out).expect("write pins");
    println!(
      "pinned {} cells for {plat} -> {} ({:.1}s encode, {:.1}s decode-gate)",
      new_rows.len(),
      path.display(),
      enc_secs,
      dec_secs
    );
    // Arm failures still fail a --pin run: pinning must not paper over a
    // broken neutral contract.
    if arm_fail > 0 {
      println!("gate-identity: FAIL ({arm_fail} arm failures during --pin)");
      std::process::exit(1);
    }
    return;
  }

  println!(
    "gate-identity [{}{plat}]: {} baseline cells ({} pinned-ok, {} drift, \
     {} unpinned), {} arm cells ({} ok, {} FAIL), all decodable, in {:.1}s \
     (+{:.1}s decode-gate)",
    if ci { "ci, " } else { "" },
    new_rows.len(),
    pinned_ok,
    drift,
    unpinned,
    arm_ok + arm_fail,
    arm_ok,
    arm_fail,
    enc_secs,
    dec_secs
  );

  if pins.is_empty() {
    println!(
      "gate-identity: FAIL — no pins for platform {plat} in {}.\n\
       Run `just gate-identity-pin` on this platform and commit the rows \
       (the committed pins are linux-x86_64; run the gate there, or pin \
       this platform deliberately).",
      path.display()
    );
    std::process::exit(1);
  }
  if drift > 0 || arm_fail > 0 || unpinned > 0 {
    if unpinned > 0 {
      println!(
        "gate-identity: {unpinned} grid cells have no pin (grid changed?) — \
         re-pin intentionally with `just gate-identity-pin`."
      );
    }
    println!("gate-identity: FAIL");
    std::process::exit(1);
  }
  println!("gate-identity: PASS");
}

fn no_op(_e: &mut EncoderConfig) {}
