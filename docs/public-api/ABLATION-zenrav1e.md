# zenrav1e Public-API Ablation Report

**Date:** 2026-06-10
**Snapshot commit:** `574c1084` (feat: versioned public-API surface snapshots)
**Snapshot:** `docs/public-api/zenrav1e.txt` — 2,771 items, default features only
**Grep template:** `ugrep -r "<SYMBOL>" /home/lilith/work/ --exclude-dir={target,.jj,docs/public-api}`
**Mode:** REPORT ONLY — no code changes in this commit.

---

## Summary

| Category | Items | % of 2771 |
|---|---|---|
| Total items in snapshot | 2771 | 100% |
| **Flagged (A or B)** | **134** | **4.8%** |
| Action A (#[doc(hidden)]) | 134 | 4.8% |
| Action B (pub(crate)/remove, queued breaking) | 0 | 0% |
| Keep (confirmed consumers or necessary surface) | 2637 | 95.2% |

3 type clusters flagged, all Action A. No B-class items found at this time — the types are still
technically reachable through `prelude::*` and removing them would be a breaking change for any
code that pattern-matches or names them explicitly. `#[doc(hidden)]` is the conservative first
step; a coordinated removal belongs in a future 0.x breaking batch.

---

## External consumer scan (as of this scan)

Known consumers checked: `zenavif`, `ravif` (in `~/work/zen/ravif/`), `zenmetrics`, `zencodecs`
(in `zenpipe`), the in-repo `ivf` member crate, and the in-repo `rav1e`/`rav1e-ch` binaries.
Search excluded `target/`, `.jj/`, `docs/public-api/`.

**Confirmed external users of the zenrav1e `prelude`:**

| Consumer | Items used |
|---|---|
| `ravif/src/av1encoder.rs` | `Config`, `EncoderConfig`, `Context`, `SpeedSettings`, `BlockSize::{BLOCK_*}`, `SegmentationLevel::{Complex,Simple}`, `Tune::{StillImage,Psychovisual}`, `PredictionModesSetting::{ComplexAll,Simple}`, `SceneDetectionSpeed::None`, `MotionSpeedSettings`, `TransformSpeedSettings`, `SGRComplexityLevel` |
| `ravif/src/animated.rs` | `Config`, `EncoderConfig`, `Context`, `Tune::Psychovisual` |
| `ravif/src/lib.rs` (re-export) | `ChromaticityPoint`, `ColorPrimaries`, `ContentLight`, `MasteringDisplay`, `MatrixCoefficients`, `PixelRange`, `TransferCharacteristics` |
| `ravif/src/error.rs` | `zenrav1e::InvalidConfig`, `zenrav1e::EncoderStatus` (top-level) |
| `zenavif/src/expert.rs` | doc-only reference to `SpeedSettings::from_preset` (no direct import) |

**Zero external consumers found for:**
`Sequence`, `PredictionMode` (the codec enum, not `PredictionModesSetting`), `TxType`.

---

## Flagged items

### [A] `prelude::Sequence` — 62 items

**Source:** `src/encoder.rs`, re-exported in `src/lib.rs` as
`pub use crate::encoder::{Sequence, Tune};` inside `pub mod prelude`.

**What it is:** The AV1 sequence header construction struct. Contains ~30 codec-internal fields
(`delta_frame_id_length`, `decoder_model_info_present_flag`, `operating_point_idc`,
`enable_delayed_loopfilter_rdo`, etc.) and two methods whose signatures reference encoder
internals that are not themselves publicly constructible:

```
pub fn get_skip_mode_allowed<T: Pixel>(
    &self,
    &zenrav1e::encoder::FrameInvariants<T>,   // encoder-internal
    &zenrav1e::api::internal::InterConfig,     // pub(crate) module
    bool,
) -> bool
```

**Consumer scan:** 0 external references to `Sequence` in zenavif, ravif, zenmetrics.
In-repo bins (`src/bin/rav1e.rs`) import via `use zenrav1e::prelude::*` but do not name
`Sequence` directly in those files; the bin stats module (`src/bin/stats.rs`) does use it for
frame statistics reporting — that is an internal binary, not a library consumer.

**Note on `Tune`:** `Tune` is paired with `Sequence` in the same re-export line but IS a
confirmed ravif consumer (`Tune::StillImage`, `Tune::Psychovisual`). Keep `Tune`; apply
`#[doc(hidden)]` only to `Sequence`.

**Proposal:** A — add `#[doc(hidden)]` to `Sequence` in `src/encoder.rs`.

Top offenders within this cluster (fields exposing codec internals):

| Item | Note |
|---|---|
| `Sequence::decoder_model_info_present_flag` | AV1 bitstream flag |
| `Sequence::operating_point_idc: [u16; 32]` | Operating point bitmasks |
| `Sequence::delta_frame_id_length` / `frame_id_length` | Frame ID coding |
| `Sequence::get_skip_mode_allowed(FrameInvariants, InterConfig)` | Takes two non-pub types |
| `Sequence::enable_delayed_loopfilter_rdo` | Internal RDO flag |

---

### [A] `prelude::PredictionMode` — 52 items

**Source:** `src/predict.rs`, re-exported in `src/lib.rs` as
`pub use crate::predict::PredictionMode;` inside `pub mod prelude`.

**What it is:** The full AV1 prediction mode enum — 35 variants covering both intra
(`DC_PRED`, `H_PRED`, `V_PRED`, directional angles) and inter (`NEARESTMV`, `NEAR0MV`,
`GLOBALMV`, `GLOBAL_GLOBALMV`, compound modes, etc.). Exposes four methods that take
encoder-internal types none of which are constructible outside the crate:

```
pub fn predict_intra<T: Pixel>(
    self, TileRect, &mut PlaneRegionMut<'_, T>,
    zenrav1e::transform::TxSize,          // private module
    usize, &[i16],
    zenrav1e::predict::IntraParam,         // private module
    Option<zenrav1e::predict::IntraEdgeFilterParameters>,  // private
    &zenrav1e::partition::IntraEdge<'_, T>,  // private module
    zenrav1e::config::CpuFeatureLevel,
)

pub fn predict_inter*(FrameInvariants<T>, TileRect, ...,
    [partition::RefType; 2], [mc::MotionVector; 2],
    &mut predict::InterCompoundBuffers)
```

**Consumer scan:** 0 external references to `PredictionMode` (the codec enum) in zenavif,
ravif, zenmetrics. Confirmed that ravif uses `PredictionModesSetting` (a config enum from
`api::config`), not this enum. The in-repo stats binary does use `PredictionMode` variants
for per-mode statistics reporting.

**Proposal:** A — add `#[doc(hidden)]` to `PredictionMode` in `src/predict.rs`.

Top offenders:

| Item | Note |
|---|---|
| `predict_intra(TxSize, IntraParam, IntraEdgeFilterParameters, IntraEdge)` | All args from private modules |
| `predict_inter*(FrameInvariants, RefType, MotionVector, InterCompoundBuffers)` | 4 private-module types |
| Entire inter-prediction variant cluster | `NEARESTMV`…`GLOBAL_GLOBALMV` — 15 variants of encoder inter-frame logic |

---

### [A] `prelude::TxType` — 23 items

**Source:** `src/transform/` module, re-exported in `src/lib.rs` as
`pub use crate::transform::TxType;` inside `pub mod prelude`.

**What it is:** The AV1 transform type enum (17 variants: `DCT_DCT`, `ADST_DCT`, `FLIPADST_*`,
`IDTX`, `WHT_WHT`, etc.) plus one method:

```
pub fn uv_inter(self, zenrav1e::transform::TxSize) -> Self
```

`TxSize` is from a private module; the method is only callable from within the crate.

**Consumer scan:** 0 external references to `TxType` in zenavif, ravif, zenmetrics.
`aom-decoder-rs` has its own `TxType` (different crate, different type, no relation).
The in-repo stats binary uses `TxType` variants for per-transform statistics.

**Proposal:** A — add `#[doc(hidden)]` to `TxType` in `src/transform/mod.rs` (or wherever
`TxType` is defined with `pub`).

Top offenders:

| Item | Note |
|---|---|
| `TxType::uv_inter(TxSize)` | Argument from private `transform` module |
| 17 codec-internal variants | `WHT_WHT`, `FLIPADST_FLIPADST`, `H_ADST`, `V_FLIPADST`… |

---

## Items explicitly confirmed KEEP

These items were checked and have confirmed external consumers or are necessary for public API
usability. They should not be flagged.

| Item | Reason |
|---|---|
| `prelude::BlockSize` (64 items) | ravif uses `BLOCK_{4..128}X{4..128}` and `from_width_and_height_opt` |
| `prelude::Tune` | ravif uses `Tune::StillImage` and `Tune::Psychovisual` |
| `prelude::SpeedSettings` + sub-structs | ravif constructs and mutates all fields |
| `prelude::SegmentationLevel`, `SGRComplexityLevel`, `SceneDetectionSpeed` | ravif uses variants directly |
| `prelude::PredictionModesSetting` | ravif uses `ComplexAll` / `Simple` |
| `config::GrainTableSegment`, `NoiseGenArgs`, `NUM_*`, `TransferFunction` | Required to construct `EncoderConfig::film_grain_params`; from `av1_grain` crate, re-exported so callers don't need a direct av1-grain dependency |
| `config::CpuFeatureLevel` | In `SpeedSettings` + `predict_intra` signature; useful for diagnostic display |
| `prelude::Sequence::tiling: TilingInfo` | `TilingInfo` is needed; arises because `Sequence` is pub |
| `prelude::context::RcData` | Multi-pass rate control API (`rc_receive_pass_data`) |
| All `color::*`, `data::*`, `version::*` | Legitimate public API surface |
| `prelude::{Config, EncoderConfig, Context, InvalidConfig, EncoderStatus}` | Core encode API |

---

## Notes on the rav1e foundation

`Sequence`, `PredictionMode`, and `TxType` appear in the public surface because they were public
in upstream [rav1e](https://github.com/xiph/rav1e) — required for the `bench` feature module,
which re-exports everything via `pub mod bench { pub mod predict { pub use crate::predict::*; } }`.
The `bench` feature intentionally breaks the public/private boundary for benchmarking. The fix
proposed here does not affect `bench`-feature users (who already expect the full internal surface).

Applying `#[doc(hidden)]` is the correct conservative action: it removes the items from
rendered docs and signals "do not depend on this" without a breaking API change. A future
0.x minor release can promote them to `pub(crate)` once downstream is confirmed clear.

---

## Implementation sketch (for reference, not applied here)

```rust
// src/encoder.rs
#[doc(hidden)]  // no external consumers; bench feature already bypasses visibility
pub struct Sequence { ... }

// src/predict.rs
#[doc(hidden)]  // no external consumers; bins use for stats only
pub enum PredictionMode { ... }

// src/transform/mod.rs (or wherever TxType is defined)
#[doc(hidden)]  // no external consumers; bins use for stats only
pub enum TxType { ... }
```

The `prelude` re-exports do not need changes; `#[doc(hidden)]` on the definition propagates
through re-exports in rustdoc.
