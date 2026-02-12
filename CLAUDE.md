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

## Hardcoded Feature Disables (to enable)

- `src/header.rs:754` — QM disabled: `write_bit(false) // no qm`
- `src/header.rs:760` — delta_q disabled: `write_bit(false) // delta_q_present_flag`
- `src/encoder.rs:302` — filter_intra: `enable_filter_intra: false`

## Implementation Progress

### Completed
- [x] Fork from xiph/rav1e
- [x] Edition 2024, MSRV 1.85

### In Progress
- [ ] Phase 1A: Quantization Matrices (~10% BD-rate)
- [ ] Phase 1B: Variance Adaptive Quantization (~5-8% BD-rate)
- [ ] Phase 1C: Still-Image Tuning (~3-5% BD-rate)
- [ ] Phase 2: Filter Intra Prediction (~3-5% BD-rate)
- [ ] Phase 3: Lossless Mode
