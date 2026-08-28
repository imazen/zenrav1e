# rav1e (Imazen Fork) - Claude Code Instructions

## Project Overview

Fork of xiph/rav1e, extending it for still and animated AVIF encoding.
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

**Executable gates** (zenavif `docs/ENGINEERING_BASELINE.md` section A): run
`just gate-identity` (A1 off-state byte-exactness, pinned fingerprints in
`tests/gate_identity_pins.tsv`; CI runs the `--ci` subset) and
`just gate-recon` (A5 encoder-recon vs conforming decoders) before AND after
any change touching coding paths. `just gate-sliver64` is the A5 subset CI
runs on every push ("Gate A5"): the 64-dim sliver and 4:1/1:4 inter
partition roundtrips against rav1d-safe in-process AND aomdec, recon
byte-exact. `SLIVER64_DUMP_IVF=<dir>` on that test writes all 28 streams
for an external dav1d/aomdec sweep. Re-pin with `just gate-identity-pin` only
for intentional byte movement, committing the TSV diff in the same commit.
zenavif's halves (`gate-determinism`/`gate-conformance`/`gate-ladder`) live
in ../zenavif's justfile.

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

## Known Bugs

- **rav1d-safe 0.5.7 (registry) index-panics in `safe_simd/looprestoration_arm.rs`
  on conformant streams (aarch64 only).** Surfaced 2026-08-26 by
  `gate_identity`'s decode gate: every `s2/q140` cell on Apple Silicon, all
  images/tunes/arms; `aomdec` and `dav1d` decode the same bytes. Fixed on
  rav1d-safe main (0.6.0-unreleased) — the dev-dep is git-pinned to
  `91bf0e30d346a9236ac4a0013f2a8a713452d37b`. Return to a registry dep at the
  rav1d-safe 0.6.0 publish. Local repro: `GATE_IDENTITY_DUMP=~/tmp/gate-dump
  cargo run --release --example gate_identity -- --ci` with the dev-dep back
  on `"0.5.7"`.

## Known Bugs (Fixed)

### Inter 4:1 sliver top-right MV candidate desync (fixed: see master)
Inter frames with HORZ_4/VERT_4 partitions (32x8 / 8x32 / 64x16 / 16x64 /
16x4 / 4x16) could emit streams rav1d-safe, dav1d AND aomdec reject
(rav1d-safe InvalidData, dav1d "Invalid argument", aomdec "Corrupted
segment_ids" -- the segment_id read is just where the garbage surfaced).
Root cause: `has_tr` (`src/partition.rs`, the spatial top-right MV
candidate's availability in `setup_mvref_list`) encoded the VERT/HORZ
rectangle rules as `(x & w) == 0` / `(y & h) != 0`, which agree with
libaom's `is_last_vertical_rect = !((mi_col + w) & (h - 1))` /
`is_first_horizontal_rect = !(mi_row & (w - 1))` only for the 2:1 shapes.
For HORZ_4 the 3rd 4:1 sliver kept a top-right candidate no decoder adds
(and for VERT_4 the 2nd 1:4 sliver of a bottom-right-quadrant parent lost
one it has), shifting the mv stack and the NEWMV/REFMV/GLOBALMV contexts,
so the tile desynced at that sliver's inter mode symbol. Located by
aligning the encoder's per-symbol range trace against an instrumented
dav1d (`DEBUG_BLOCK_INFO`): frame 2 of the `Bands::Both` case diverged
exactly at the inter mode of the 32x8 at mi (16, 60), the 3rd HORZ_4
sliver of the 32x32 parent at (16, 48). Not reachable from the stock
presets (speed >= 2 caps `non_square_partition_max_threshold` at
BLOCK_8X8; presets <= 1 are bottom-up), reachable from zenavif/ravif
widened thresholds on animated encodes. Gates:
`partition::tests::has_tr_matches_libaom_for_every_aligned_position` (all
19 sizes x every aligned mi position over 2x2 superblocks vs a transcribed
libaom rule) and `tests/sliver_64_tx_roundtrip.rs` (`Bands::Both` inter,
both tx-split arms; rav1d-safe in-process and aomdec via
`SLIVER64_AOMDEC` -- `just gate-sliver64`, CI "Gate A5"). Both
mutation-verified against the pre-fix tests. Re-verified independently
2026-08-28: restoring both pre-fix rules fails
`has_tr_matches_libaom_for_every_aligned_position` at BLOCK_4X16 mi (5, 4)
and fails the named case (`Cs420, q 100, tx_select false, frames 3,
inter_tx_split true, Bands::Both`) at packet 2 with rav1d-safe
`InvalidData`; with the fix that case passes and all 28 IVFs dumped by
`SLIVER64_DUMP_IVF` decode under aomdec and dav1d 1.5.4.

### TX_64X16/TX_16X64 eob CDF desync (issue #28, fixed: see master)
Coding a BLOCK_64X16/16X64 sliver (HORZ_4/VERT_4 at a 64x64 parent) with
its max rect transform desynced every conforming decoder. `encode_eob`
picked the eob_pt CDF family from the nominal area (`tx_size.area_log2()`:
1024 → the 11-symbol 1024 CDF) while decoders key it on the coded
coefficient count (512 → the 10-symbol 512 CDF; dav1d `min(lw,3)+min(lh,3)`,
libaom `txsize_log2_minus4`). TX_64X64/64X32/32X64 were unaffected only
because their nominal area ≥ 2048 hits the same 1024 arm. Fix:
`ContextWriter::eob_multi_size` keys on `av1_get_coded_tx_size`. The
3fa735dc intra cap, the palette-trial caps, and the 64x64-parent 4-way
partition gate (#34) are removed. Gates: `eob_multi_size_matches_spec_for_
every_tx_size` (all 19 sizes vs the spec table) and
`tests/sliver_64_tx_roundtrip.rs` (encoder recon == rav1d-safe output,
intra LARGEST/SELECT × 4:2:0/4:4:4, inter ± tx split; the intra sliver
streams also verified with aomdec + dav1d). Mutation-verified.

### CDEF dir-search debug_assert on 8-bit-in-u16 (issue #10, fixed: see master)
`encode_decode_hbd` fuzz target crashed at `src/cdef.rs:95:9`
(`assertion failed: p >> coeff_shift <= 255`) in `cdef_find_dir`. Root cause:
**8-bit content stored in `u16`** (`Context<u16>` with `bit_depth == 8`) was
routed to the high-bitdepth (HBD) x86 SIMD kernels. dav1d's HBD inverse
transform (10bpc *and* 12bpc), 16bpc intra predictor, and 16bpc subpel
(`put_8tap`) kernels are only bit-accurate at their native depths (>= 10) —
8-bit always uses the dedicated 8bpc (u8) kernels — so there is no asm path
valid for 8-bit-in-u16. Fed `bitdepth_max=255` they emit out-of-range
reconstructed samples (256 from the itx, 512 from the DC predictor), which the
CDEF direction search then reads. Three x86 dispatches lacked the `bd==8 ->
Rust` guard the rest of the codebase already had (aarch64 predict/itx/mc, x86
`prep_8tap`/`mc_avg`, x86 `quantize`): `transform/inverse.rs` (12bpc was the
catch-all), `predict.rs`, and `mc.rs` `put_8tap`. Fix routes bd==8 to the
correctly-clamped Rust kernels. Diagnosed by confirming both itx variants and
the DC predictor emit OOR for bd=8 and that forcing the Rust path eliminates
it; the dav1d roundtrip then matches. Regression guard:
`src/test_8bit_u16.rs` (encode-only, runs in the default + no-asm CI test
jobs). Crash seed: `fuzz/regression/cdef-range-8bit-in-u16-encode_decode_hbd.bin`.

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

## Benchmark Results (2026-02-12, 63-image corpus, speed 6)

Per-image BD-Rate vs upstream rav1e (SSIMULACRA2, negative = better):

| Configuration | Mean BD-Rate | Median | Range | Improved |
|---|---|---|---|---|
| **QM only** | -10.1% | -10.0% | [-15.2%, -5.7%] | 67/67 |
| **QM + RdoTx** | -10.3% | -9.6% | [-31.2%, -2.7%] | 63/63 |
| **QM + CDEF + RdoTx** | -10.7% | -9.8% | [-31.6%, -3.5%] | 63/63 |

RdoTx (rdo_tx_decision) adds -5.4% BD-Rate on top of QM alone (58/63 improved)
but at **2.5-3.4x encode time** cost (speed 6: 101→259ms at Q50, 271→927ms at Q95).
CDEF adds -0.3% on top of that (marginal) with additional ~15% encode time.

### Features Tested and Abandoned
- **VAQ (SSIM boost)**: +2.8% mean — consistently worse. Psychovisual tune
  already activates SSIM boost; VAQ with strength < 1.0 reduces masking.
- **StillImage tuning**: ~0% — no effect. ravif disables CDEF at high quality.
- **Variance Boost (SVT-AV1-PSY style)**: Inflates bitrate 8-65% because
  rav1e's RDO allocates more total bits when distortion tolerances vary widely.
- **Separated Segmentation Boost (seg_boost)**: Trades BPP for quality at
  constant ratio. At boost 2.0: -7.6% BPP but -1.40 SS2. Not improving
  compression efficiency — just shifting the operating point.
- **Per-SB delta-q**: Already implemented via segmentation (3-8 segments with
  QP offsets). Additional delta-q mechanism would have same RDO limitation.
- **SGR full complexity**: Zero effect at speed 6. Loop restoration parameters
  don't change with 16 vs 8 SGR parameter sets on small/medium images.
- **LRU on skip (loop restoration on skip blocks)**: Zero effect at speed 6.
- **Complex segmentation**: Shifts operating point (BPP and quality both drop
  ~3%), not an efficiency gain. +35% encode time. Same issue as seg_boost.
- **Bottom-up partition search**: Zero effect at speed 6. Top-down search
  already finds good partitions for still images.
- **Trellis quantization (EOB shrinkage + level round-down)**: CDF-based rate
  model using actual AV1 entropy coder CDFs. Quality-adaptive dampening
  (80/ac_quant) prevents low-Q regression; early exit at ac_quant >= 200.
  Result: ~0.3% BPP savings at Q90-Q95 with no quality loss, but +34%
  encode time. Not worth it — rav1e's adaptive rounding biases are already
  well-tuned, leaving little room for trellis to improve.

Recommended config:
- **Speed priority**: `enable_qm: true` only (-10.1% BD-Rate, ~1x encode time)
- **Quality priority**: `enable_qm: true` + force `rdo_tx_decision: true`
  (-10.3% BD-Rate, ~3x encode time)
- Everything else default/off.
