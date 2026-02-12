# rav1e (Imazen Fork) - Claude Code Instructions

## Project Overview

Fork of xiph/rav1e focused on best-in-class still image AVIF encoding.
Hard fork — free to make breaking changes for still image quality.

## License

BSD-2-Clause + AOM Patent License (inherited from upstream)

## Build

```bash
# Pure Rust (no asm) — primary development target
cargo check --no-default-features --features threading
cargo test --no-default-features --features threading

# With asm (requires nasm)
cargo check --features threading
```

## Key Files for Still Image Work

- `src/quantize/mod.rs` — quantization/dequantization core
- `src/quantize/tables.rs` — Q-index lookup tables
- `src/header.rs` — bitstream writing (QM at :754, delta_q at :760)
- `src/encoder.rs` — FrameInvariants (:603), Tune enum (:108), SB loop (:3482)
- `src/rdo.rs` — rate-distortion optimization, intra mode search (:1394)
- `src/predict.rs` — prediction modes, FilterIntraMode enum (:512)
- `src/api/config/encoder.rs` — EncoderConfig
- `src/api/config/speedsettings.rs` — SpeedSettings
- `src/context/block_unit.rs` — write_use_filter_intra (:760), code_deltas (:230)
- `src/deblock.rs` — deblocking filter
- `src/cdef.rs` — CDEF
- `src/lrf.rs` — loop restoration

## Implementation Progress

### Completed
- [x] Phase 0: Fork from xiph/rav1e, Edition 2024, MSRV 1.85
- [x] Phase 1A: Quantization Matrices (~10% BD-rate) — `enable_qm: bool`
- [x] Phase 1B: Variance Adaptive Quantization (~5-8% BD-rate) — `enable_vaq: bool`, `vaq_strength: f64`
- [x] Phase 1C: Still-Image Tuning (~3-5% BD-rate) — `Tune::StillImage`
- [x] Phase 2: Filter Intra Prediction (~3-5% BD-rate) — 5 recursive filter modes, auto-enabled at speed ≤ 6
- [x] Phase 3: Lossless Mode — `quantizer: 0` for mathematically lossless output

### Not Yet Started
- [ ] Phase 4: SSIMULACRA2 Target-Quality Convergence (ravif layer)
- [ ] Phase 5: Integration (ravif/zenavif/zencodecs)

## Known Bugs (Fixed)

### QM eob calculation (fixed: 358d4f51)
Deadzone-based eob prediction used global base quantizer, but QM gives each
coefficient position a different effective quantizer. This caused eob overshoot,
leading to segfaults in release builds from incorrect entropy coding.
Fix: recompute eob from actual quantized coefficients when QM is active.

### QM offset scaling (fixed: 734bd79e)
Integer division truncation in offset scaling: `weighted_q * (offset / base_q)`
truncated to 0 (since offset ≈ 42% of base_q), then `.max(1)` made offset = 100%
of weighted_q. This eliminated the quantization deadzone, distorting the
rate-distortion tradeoff. Fix: use u64 proportional scaling.
Before fix: QM caused +20% BD-Rate regression (worse).
After fix: QM provides -5.5% BD-Rate improvement (better).

## Benchmark Results (2026-02-12, 67-image corpus, after all fixes)

Feature ablation (SSIMULACRA2 BD-Rate vs upstream rav1e baseline):
- **QM only**: ~-5.5% (saves 5.5% bitrate at same quality) — best single feature
- **QM+VAQ+StillImage**: ~-3.5% (VAQ overhead partially offsets QM gains)
- **VAQ only**: ~+2.8% (consistently worse, needs investigation)
- **StillImage only**: ~+0.3% (no measurable effect — CDEF disabled at high quality in ravif)

Note: StillImage's CDEF/deblock adjustments have no effect because ravif disables
CDEF for high-quality encodes (`cdef: Some(low_quality && speed <= 9)` where
`low_quality = quantizer > 150`).
