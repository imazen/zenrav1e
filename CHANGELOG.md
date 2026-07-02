# Changelog

## [Unreleased]

### Documentation
- README overhaul to the zen house style: standardized `flat-square` badge row
  (CI/crates.io/lib.rs/docs.rs/MSRV/license, no `branch=`), a `## Quick start`
  section, refreshed feature claims (multi-level trellis RDOQ −0.94% BD-rate
  opt-in; pure-Rust toolchain-free default), `0.1.4`→`0.2.0` dep snippet, a
  skip-wrapped Benchmarks section + `benchmarks/README.md` index, the rendered
  crosslink footer, and a split crates.io README (`README.crates.md`, generated;
  `readme` + `include` updated in `Cargo.toml`).

### Changed (BREAKING)
- **`whereat` traces applied by benefit, not by API boundary.** An earlier
  iteration wrapped the config-validation API in `At<InvalidConfig>`; that was
  reverted after review. `InvalidConfig` is **bare** again — every variant names
  the exact setting it rejected (`InvalidWidth(8)`, `InvalidBitDepth(7)`, …), so
  it is self-describing and a trace would only point back into `validate()`,
  which the variant already implies. `Config::validate` / `new_context` /
  `tiling_info` and the channel constructors return `Result<_, InvalidConfig>`
  (the `ConfigResult<T>` alias is now bare). The per-frame `EncoderStatus` stays
  bare too (hot path / ordinary control flow).
  Instead, the trace is applied where an origin is genuinely **non-obvious**:
  **`RateControlError` (a.k.a. `rate::Error`) is now `At`-wrapped.**
  `RateControlSummary::from_slice` and `RateControlConfig::from_summary_slice`
  return `Result<_, whereat::At<RateControlError>>` — a `CorruptedSummary` comes
  out of the binary deserializer, so the trace points at the parse site that
  rejected the blob (which the flat `String` message can't convey). `At` is
  re-exported at the crate root. Migration: a caller of `from_summary_slice`
  matches `Err(e)` then inspects `e.error()` (borrow); `?` into
  `Box<dyn Error>` / `anyhow` still works. The C API (`src/capi.rs`) is
  unchanged (it uses `.ok()` / discards construction errors). Version bumped
  `0.1.4` → `0.2.0`.
- **Pure-Rust, toolchain-free default features.** `default` is now
  `["threading"]` (was `["asm", "threading", "signal_support", "scenechange"]`).
  Three concerns moved to the `binaries` feature so the `rav1e` CLI stays fast +
  full-featured while library consumers carry none of them by default:
  - `asm` — NASM SIMD (via `nasm-rs` + `cc`); a **build-toolchain** dependency.
    A plain default build is now pure Rust and needs no NASM/C toolchain. Add
    `features = ["asm"]` for the SIMD speedups.
  - `scenechange` — pulls `av-scenechange`; only used for video/by-GOP keyframe
    placement. Still-image encoders fall back to the existing no-op stub. Add
    `features = ["scenechange"]` for scene-cut keyframe placement.
  - `signal_support` — `signal-hook`, a CLI-only Ctrl-C concern used solely in
    `src/bin/*`; never belonged in a library default.

  The default dependency tree drops to just `maybe-rayon`. Verified: pure-Rust
  default builds + 129 lib tests + clippy `-D warnings` pass; the asm path and
  the CLI (built with `binaries`) are unchanged. **The primary downstream —
  ravif/zenavif/zencodecs — already builds `default-features = false` and opts
  into `asm`/`threading` explicitly, so it is unaffected.**

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
- **`angle_delta`/palette-mode gates diverged from libaom's ordinal semantics for
  `BLOCK_4X16`/`BLOCK_16X4` (#26)** — `bsize >= BlockSize::BLOCK_8X8` looks like
  libaom's ordinal `av1_use_angle_delta`/`av1_allow_palette` check but isn't:
  `BlockSize` has a custom width/height-based `PartialOrd`, under which
  `BLOCK_4X16`/`BLOCK_16X4` are *incomparable* with `BLOCK_8X8` (one dimension
  smaller, one larger), so `>=` silently evaluated `false` where libaom's
  ordinal C-enum comparison is `true`. The encoder skipped writing a required
  `angle_delta` syntax element for directional-mode blocks of those two sizes —
  a missing-symbol bitstream desync that any spec-conformant decoder rejects.
  Added `BlockSize::ge_8x8_ordinal()` and swapped it in at the 4 affected call
  sites (2866397e). Found while re-investigating the HORZ_4/VERT_4 conformance
  bug from the previous session's attempt at #26.
- **No panic on a coded-lossless inter frame (#24)** — `encode` fuzzing tripped
  `debug_assert!(depth <= MAX_TX_DEPTH)` in `tx_size_to_depth`
  (`src/context/transform_unit.rs:639`). A lossless frame forces `TX_4X4` on
  every block, but the inter-frame path set `tx_mode_select` from
  `enable_inter_txfm_split` with no lossless check, so an intra block inside a
  lossless inter frame still emitted tx-size syntax via `write_tx_size_intra`.
  For a `BLOCK_32X32` block the `TX_32X32 → TX_16X16 → TX_8X8 → TX_4X4` descent
  is depth 3, past `MAX_TX_DEPTH` (2). The frame became lossless via
  bitrate-mode rate control driving `base_q_idx` to 0, so the config-quantizer
  proxy disagreed with the runtime state. AV1 (spec 5.9.21) infers
  `tx_mode = ONLY_4X4` for lossless and writes no tx-size syntax — matching
  `header.rs`, the tx-size syntax in `encode_block_post_cdef` is now suppressed
  whenever `fi.is_lossless()` (the authoritative, rate-control-aware predicate),
  fixing both the inter-frame path and the bitrate-mode key-frame disagreement.
  Regression: a dav1d encode→decode roundtrip of a multi-frame lossless stream
  at speeds 9/10 (`multiframe_lossless_*`), a decoder-free public-API guard
  (`lossless_inter_frame_tx_size_no_panic`), and the seed replay
  `fuzz/regression/txsize-depth-lossless-inter-encode.bin` via the new
  `tests/fuzz_regression.rs` harness (`_fuzz_replay` feature).
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
- **One-level-deeper SPLIT child estimate in the topdown partition trial
  (#27)** — `rdo_partition_simple` historically scored each SPLIT child as a
  single NONE-leaf while the final encode re-searches every SPLIT child
  recursively and usually does better, making SPLIT's trial cost
  systematically pessimistic vs the exactly-evaluated NONE/HORZ/VERT/
  HORZ_4/VERT_4 candidates. Each SPLIT child's trial cost is now
  `min(NONE-leaf, tell-metered child-SPLIT symbol + 4 quarter NONE-leaves)`
  (`rdo_split_child_deeper_cost`, b073182c) — exactly the first comparison
  the child's own future search will make — with winning deeper state kept
  for sibling estimation and losing state fully rolled back. Measured
  (22-image photo corpus × 12-Q grid, cavif -s2): BD-rate vs libaom-slow
  cpu-used=2 median +0.0695% → **−0.6487%** (mean +2.1734% → +0.2373%),
  improved on 16/19 images, encode time 1.057× median — RD parity crossed
  at matched speed. Full data in the `zenavif` sibling repo's
  `docs/RD_GAP_VS_LIBAOM.md` "Fixed 2026-07-02".
- **`PARTITION_HORZ_4`/`PARTITION_VERT_4` in the RDO search (#26, Phase 1 of
  extended AV1 partition types)** — `encode_partition_topdown` can now choose
  the two uniform 4-way splits for `BLOCK_16X16`/`32X32`/`64X64` blocks fully
  contained within the frame, gated by `non_square_partition_max_threshold`.
  Previously 0 of the 6 "extended" AV1 partition types were ever attempted by
  the RDO search at any speed; this closes 2 of the 6 (7d254289). Measured
  extended-block-size area share 1.8-56% per cell across a 22-image photo
  corpus x 5 quality levels (110 cells), `aomdec`-clean (verified against the
  ordinal-comparison fix above, without which the same feature produced a
  bitstream `aomdec` rejects as corrupt).
  `PARTITION_HORZ_A/B`/`VERT_A/B` (the remaining 4) are still not implemented.
  RD impact measured in the `zenavif` sibling repo's `docs/RD_GAP_VS_LIBAOM.md`.
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
