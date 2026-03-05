# Changelog

## Unreleased (since v0.1.0)

### Changed
- Edition 2024, MSRV 1.89
- Bumped safe_unaligned_simd minimum to 0.2.5
- Comprehensive CI: 6-platform matrix (ASM on x64), i686, clippy, fmt, MSRV check, Codecov
- Updated archmage/magetypes to 0.9

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
