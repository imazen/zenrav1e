# Changelog

## [Unreleased]

### Changed (BREAKING)
- **Config validation now returns `At<InvalidConfig>` for server-side stack
  traces.** `Config::validate`, `Config::new_context`, `Config::tiling_info`,
  and the `new_channel` / `new_firstpass_channel` / `new_secondpass_channel` /
  `new_multipass_channel` / `new_by_gop_channel` constructors now return
  `Result<_, whereat::At<InvalidConfig>>` (aliased `ConfigResult<T>`) instead of
  bare `InvalidConfig`. The trace points at the exact validation site that
  rejected the configuration. `At` is re-exported at the crate root and
  `InvalidConfig` itself is unchanged. These paths are cold (run once per encode
  session). `EncoderStatus` — the hot per-frame status from
  `Context::receive_packet` / `Context::send_frame` — **intentionally stays
  bare**: `whereat` is deliberately kept off the encode hot path (trace
  allocation / register-spill avoidance, and it is frequently ordinary control
  flow such as `NeedMoreData`, not an error). The C API (`src/capi.rs`) is
  unchanged — it stores only `EncoderStatus` and discards the construction
  error to a null pointer. Migration: a caller that matched
  `Err(InvalidConfig::X)` now matches `Err(e)` then inspects `e.error()`
  (borrow) or `e.decompose().0` (owned); propagating with `?` into a
  `Box<dyn Error>` / `anyhow` context still works unchanged. Version bumped
  `0.1.4` → `0.2.0`.

### Investigated
- **#6 bottom-up partition × QM regression** — measured negative result, no fix
  landed. The proving sweep (speeds {1,2,4,6} × q5..=100:5 × QM{off,on} × photo
  + sci-figure, profile-A) shows bottom-up never beats top-down; on synthetic
  content it loses 30–56 zensim *with and without* QM. The issue's `ts.rec`-
  rollback hypothesis is falsified three ways; the cause is bottom-up's
  cost-evaluation path (`rdo_mode_decision` + recursive child-cost summation),
  not the partition set or neighbour-pixel state. The ravif
  `encode_bottomup=false` workaround stays. Evidence + analysis:
  `benchmarks/issue6-bottomup-qm-2026-06-13.md`.

### Fixed
- **Library now rejects a pathological frame rate (#20)** — the
  scene-change-driven encode path (the `scenechange` feature, on by default)
  forwarded the configured frame rate (`time_base.den / time_base.num`)
  unclamped into `av-scenechange`'s `TilingInfo::from_target_tiles`, where an
  extreme rate makes `min_tile_rows_ratelimit_log2` exceed `max_tile_rows_log2`
  and the subsequent `clamp(min, max)` panics with `min > max`
  (`av-scenechange-0.14.1/src/data/tile.rs:314`). The previous #16 fix bounded
  only the **fuzz harness** (`src/fuzzing.rs`); `Config::validate()` still
  admitted fps up to `u32::MAX`, so a default-feature encode with an
  extreme-but-valid config could still reach the panic at encode time.
  `validate()` now rejects any effective rate above `MAX_FRAME_RATE` (65536 fps
  — far above broadcast/web/high-speed-capture rates and well below the
  ~143616 fps panic onset for the smallest frame) as `InvalidFrameRateDen`,
  mirroring the sane-fps bound the harness applies. Purely additive rejection of
  inputs that previously panicked — no legitimate frame rate is affected.
  Regression tests `rejects_pathological_frame_rate` /
  `accepts_realistic_frame_rates` in `src/api/config/mod.rs`.
- **Fuzz harness slow-unit timeouts + av-scenechange panic (#13, #15, #16, #17)** —
  the `encode` / `encode_decode` / `encode_decode_hbd` targets could pick the most
  exhaustive RDO presets (speed 0–3) on up to 271²×3-frame inputs, producing
  multi-second encodes that tripped the fuzzer's per-input timeout. Bounded the
  arbitrary configs in `src/fuzzing.rs`: frame size now scales with the chosen
  speed preset (slow presets 0–3 capped to 48²/64², faster presets to 128²) and
  the decode-roundtrip targets are capped to ≤2 frames — keeping full
  partition/RDO-search coverage on small frames without slow-units. Separately,
  `ArbitraryEncoder::time_base` is now bounded to a sane 1–120 fps range: a
  pathological frame rate drove the third-party `av-scenechange`
  `TilingInfo::from_target_tiles` into a `clamp(min, max)` with `min > max` →
  panic (`av-scenechange-0.14.1/src/data/tile.rs:314`, #16). Verified: 60 s
  `encode` + 45 s each decode target with a 10 s per-input timeout find no
  slow-unit or crash. The underlying av-scenechange clamp is a third-party bug
  (tracked) — harness-bounding stops the fuzz noise; production callers passing
  an extreme fps remain at risk until upstream clamps `min` before `clamp`.
- **`docs(readme)`: complete the truncated encode example** — the README's
  direct-use snippet ended at `// send frames, receive packets...`, so the
  entire encode loop was undocumented and the program could not be written
  (found by an insulated external-developer usability test). Replaced it with a
  complete, copy-pasteable still-image example: the full
  `new_frame` → fill Y/U/V planes (`Plane::copy_from_raw_u8`) → `send_frame` →
  `flush` → `receive_packet` loop over the real `EncoderStatus` variants,
  writing `packet.data`. Also made explicit that input is **planar YCbCr, not
  RGB** (filling planes with RGB encodes cleanly but yields garbage colors),
  that the output is a **raw AV1 bitstream needing a muxer** (zenavif/ravif),
  the **`quantizer` q-index scale + direction** (0..=255, lower = higher
  quality, 0 = lossless), and a pasteable `[dependencies]` line.
- **Fuzz `encode` harness time bound** — `ArbitraryEncoder` allowed a 256×256,
  3-frame encode at speed preset 0 (most exhaustive RDO), ~44 s for a 58-byte
  input (fuzz timeout / DoS). Tightened to 128×128 and ≤2 frames so even the
  slowest preset stays within the per-input budget (~9 s worst case; a 60 s fuzz
  run finds no slow unit). Harness-only — the encoder is unaffected. Seed:
  `fuzz/regression/timeout-encode-speed0-large.bin`.
- **Lossless (`quantizer = 0`) was never actually lossless** — it silently
  coded qi=1 lossy output with ±2 reconstruction error on 7-28% of pixels
  (imazen/zenrav1e#9), which also inverted the size/speed curve
  (imazen/zenavif#8: slower speeds spent bits buying back phantom
  distortion). Root cause chain, all fixed:
  - `QuantizerParameters::new_from_log_q` floors `base_q_idx` at 1 even
    for an explicit `quantizer = 0` request; the constant-quality path now
    routes lossless through a dedicated `new_lossless` constructor (all
    six qi = 0, no delta-q) so `is_lossless()` actually fires.
  - The never-exercised lossless coding path had latent desyncs vs the
    spec/rav1d reader, all corrected: frame header must NOT code
    `delta_q_present` (base_q_idx = 0), `loop_filter_params`, or the
    `tx_mode` bit; `write_tx_type` must not signal for WHT blocks; CFL
    availability under lossless is the decoder's chroma-4x4 rule, not
    `bsize <= 32x32` (different uv_mode CDF alphabets = bitstream
    desync); chroma must use 4x4 WHT like luma (`uv_tx_size`/`uv_tx_type`
    were still the lossy derivations — chroma decoded as garbage);
    `WHT_WHT` (= 16) walked off four `TX_TYPES`-sized tables
    (scan orders ×3, `tx_type_counts`); the delayed-loopfilter-RDO queue
    was never drained for lossless tiles (assertion).
  - Validated end-to-end through zenavif → rav1d-safe: bit-exact
    roundtrip (0 mismatched pixels) on flat/noise/photo/screen content,
    4x4–2048², speeds 1-10, RGB-identity and YCbCr, 4:2:0/4:4:4; the
    size-vs-speed curve is now monotonic (slow ≤ fast bytes).
- **CDEF range assertion on 8-bit content stored in `u16`** — a
  `Context<u16>` with `bit_depth == 8` routed reconstruction through the
  high-bitdepth x86 SIMD kernels, which are only bit-accurate at their native
  depths (≥10). The 10/12bpc inverse transform, 16bpc intra predictor, and
  16bpc inter subpel (`put_8tap`) emitted out-of-range samples (256, 512) for
  8-bit, tripping `p >> coeff_shift <= 255` in CDEF direction search
  (imazen/zenrav1e#10, fuzz target `encode_decode_hbd`). Fix routes 8-bit-in-u16
  through the correctly-clamped Rust kernels in `transform/inverse.rs`,
  `predict.rs`, and `mc.rs` `put_8tap` — the guard the aarch64 paths and x86
  `prep_8tap`/`mc_avg` already had (62df2ec9). Regression test
  `src/test_8bit_u16.rs` (9a72bc3d).

### Added
- Versioned public-API surface snapshot at `docs/public-api/zenrav1e.txt` (default features only — decode_test/dav1d/capi/bench gates documented in the test), regenerated by `tests/public_api_doc.rs` on every `cargo test`; `ZEN_API_DOC=check` gates staleness in the CI clippy job, `=off` skips. Justfile recipes `fmt` / `api-doc` / `api-doc-check`.

## [0.1.4] - 2026-04-27

### Fixed
- QM level mapping: extend `qm_level_for_qindex` to libavif's still-image range `[4, 15]` instead of the old all-intra-video range `[4, 10]`. With the old upper bound, level 15 (= identity / no QM applied) was unreachable, so even at near-lossless qindex the encoder applied substantial QM shaping. On ac_quant 1–4 with QM weights around 80 the integer rounding `(quant * weight + 16) >> 5` multiplied the effective quantizer step 2-3× on high-frequency coefficients, collapsing zensim from ~76 at qindex 18 to ~49 at qindex 0 in zenavif's encode sweep, and degrading the entire q≥60 range by 11–22 zensim points. Fix: linear interpolation across `[4, 15]` so qindex 0 maps to level 15 (no QM applied) and shaping ramps in smoothly. After the fix the q→zensim curve is monotonic across all 5 CID22 test images, and QM-on tracks QM-off within ±0.4 zensim from q=70 onward. Fixes imazen/zenrav1e#7.
- AV1 spec 6.8.11 conformance: `set_quantizers` now clears `using_qmatrix` when the frame is coded-lossless (`base_q_idx == 0` and all delta_q == 0) and also when the selected `qm_level` is 15 for every plane (signaling QM with all-identity levels was rejected by rav1d / libaom in degenerate cases). Without this, decoding zenavif quality=100 with QM=on failed primary-frame decode.

## [0.1.3] - 2026-04-17

### Fixed
- Filter intra: forward `use_filter_intra` / `filter_intra_mode` through `rdo_tx_type_decision` so tx_type cost is estimated against the correctly remapped CDF instead of DC_PRED's (04129b4e). See imazen/zenrav1e#5 for the remaining speed 1 quality regression.
- Filter intra: map `FILTER_PAETH_PRED` to `DC_PRED` in `fimode_to_intradir` (matches AV1 spec and dav1d), and add the skip early-return to `write_tx_blocks` that the inter path already had — both fixed CDF/arithmetic-coder desync on 8-bit content (2d0ae25c).
- Filter intra: index the transform type CDF via `fimode_to_intradir[filter_intra_mode]` instead of `y_mode` per AV1 spec 5.11.40, producing bitstreams that libaom accepts (d696f4d1). Fixes imazen/zenrav1e#4, imazen/zenavif#7.
- Scenechange feature: restore `CpuFeatureLevel::default()` so the default-features build compiles against av-scenechange's multi-variant enum (f6bb314f).
- CLI: enable `--quantizer 0` lossless encoding (the library already supported it) and fix binary imports after the package rename from `rav1e` to `zenrav1e` (d5f2d89b).

### Added
- Fuzz: 197-entry AV1 encoder fuzz dictionary covering speed presets, quantizer values, bit depths, chroma sampling, color description enums, dimensions, tile config, key-frame intervals, rational time bases, HDR metadata, and boundary values, organised per fuzz target (11f7982e).
- CI: nightly fuzz workflow — 60 s on push, 5 min nightly (84affb4d).

### Changed
- Replaced 15 `unimplemented!()` sites in the Y4M decoder, `quantize`, `lrf`, `transform/inverse`, and `me` with `panic!()` / match-on-supported-values so unsupported-but-theoretically-reachable cases produce clear messages instead of generic "not yet implemented" (0143f066).
- Replaced 11 truly unreachable `unimplemented!()` sites in `header`, `frame_header`, `block_unit`, and `rdo` with `unreachable!()` + explanatory comments documenting why the path is impossible in normal encoder usage (51f1f856).
- Added tooling paths (`.superwork/`, `.claude/`, `.zenbench/`, `copter-report/`, profraw/profdata, fuzz logs, Cargo.toml backups) to `.gitignore` (3aa5af25).
- Bumped indirect `git2` dependency from 0.20.2 to 0.20.4 via dependabot (b6a8bc8f).
- Earlier this release cycle: Edition 2024 / MSRV 1.89, `safe_unaligned_simd` minimum 0.2.5, archmage / magetypes 0.9, comprehensive CI (6-platform matrix with ASM on x64, i686, clippy, fmt, MSRV check, Codecov).

## 0.1.0

Initial release. Imazen fork of rav1e, optimized for still and animated AVIF.

### Features over upstream rav1e
- Quantization matrices (~10% BD-rate improvement)
- Filter intra prediction
- Trellis quantization
- Variance adaptive quantization (VAQ)
- `Tune::StillImage` mode
- Lossless encoding mode
- Cooperative cancellation via `enough` crate
- Modernized: Rust 2024 edition, safe_unaligned_simd
