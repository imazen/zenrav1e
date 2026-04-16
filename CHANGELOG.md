# Changelog

## [Unreleased]

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
