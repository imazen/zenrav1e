# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
- `EncoderConfig` gained three public fields (`variance_boost_strength`,
  `variance_boost_deep`, `quant_rounding_bias`) — breaking for exhaustive
  struct construction (ravif constructs `EncoderConfig` exhaustively and
  adds the fields at its dep bump). Rides the already-open 0.2.0 window
  (Cargo.toml is at 0.2.0-unreleased); no additional version step needed.
- `EncoderConfig::ssim_rdmult_strength: Option<f64>` — a fourth public
  field in the same 0.2.0 window, same exhaustive-construction impact.
- `EncoderConfig::coeff_rd_stack: Option<CoeffRdStack>` — a fifth public
  field (plus the new `CoeffRdStack` config struct) in the same 0.2.0
  window, same exhaustive-construction impact.

### Added
- **Executable gates for the engineering-baseline invariants** (zenavif
  `docs/ENGINEERING_BASELINE.md` section A): `just gate-identity`
  (`examples/gate_identity.rs` + `tests/gate_identity_pins.tsv`) pins the
  default-config and `Tune::Ssimulacra2` bitstream fingerprints on a
  deterministic 3-image × s{2,6,8} × q{60,140,220} grid, checks every
  documented-neutral knob state byte-equals its absence (tx-override
  fallbacks, off-tune variance-boost/ssim-rdmult inertness, neutral
  FrameHints, intrabc_hash-under-off, num_modes historical budgets, ss2
  Some(0.0)/Some(fitted) neutrals), and liveness-pins the armed
  palette/intraBC paths (Auto-on-photo must DROP the still-picture
  screen-content signaling the palette-Off baseline inherits from stock
  rav1e — expected-different by design, so it is pinned, not
  equality-checked). CI runs the `--ci` subset on linux-x86_64;
  `just gate-identity-pin` re-pins after intentional behavioral changes.
  `just gate-recon` (`scripts/gate_recon.sh`) drives
  `examples/recon_probe.rs` over the same pinned content × the cdef/lrf
  toggle corners and byte-compares encoder recon against rav1d-safe
  and/or aomdec (the #32/#33 desync class as a gate; local-only, decoder
  legs env-selected, zero legs = loud failure, 54/54 cells clean on both
  legs at landing).
- **Composed coefficient-level RD valuation stack**
  (`EncoderConfig::coeff_rd_stack: Option<CoeffRdStack>`, default `None` =
  byte-identical — 36/36 cells sha256-identical vs a master-built rav1e
  across 3 contents × q{60,120,180} × s{2,6} × tune{ss2,psy}; validation
  error `InvalidCoeffRdStack`; disarmed on explicit-lossless configs):
  libaom's coupled "FP round-to-nearest quantization + always-on
  per-coefficient RD descent" posture as ONE knob (zenavif
  docs/COEFF_RD_STACK.md; aom 632172a4 `skip_trellis ? B : FP` +
  `av1_optimize_txb`). Armed: flat `rounding_bias`/256 forward-quant
  offsets (128 = aom FP parity; 0 = keep the fitted Valin offsets, the
  un-gate-only decomposition arm; overrides `quant_rounding_bias`), the
  trellis runs on every TU regardless of `enable_trellis` WITHOUT its
  `ac_quant >= 200` disable and `80/ac_quant` dampening at
  `lambda × trellis_lambda_scale` (aom ss2 posture 17/128, aom
  default-tune posture 4.25), optional aom `sharpness != 0` preserve
  guards (never zero level-1s; level > 2 required at scan pos <= 5;
  descent floor 1; EOB pull-in only to >= 5 kept), and an optional per-TU
  zero-out counterweight at block λ (aom tx_search.c:3294-3311 analog).
  The two prior half-stack probes (flat rounding alone; forced trellis
  alone) are measured rejections — this knob measures the composition aom
  actually ships. Liveness + rav1d-safe decodability gated in
  `tests/trellis_roundtrip.rs` including at coarse quantizers where the
  opt-in trellis was previously a hard no-op.
- **Per-16×16 ssim-rdmult λ scaling** (`EncoderConfig::ssim_rdmult_strength`,
  default `None` = byte-identical — scale 1.0 multiplies λ exactly;
  validation error `InvalidSsimRdmultStrength`; only live under
  `Tune::Ssimulacra2`): port of libaom's `av1_set_mb_ssim_rdmult_scaling`
  + `av1_set_ssim_rdmult` (encoder_utils.c / encodeframe_utils.c at rev
  632172a4, shared by aom `--tune={ssim,iq,ssimulacra2}`) — per 16×16
  source cell, mean per-pixel 8×8 variance →
  `67.035434·(1−exp(−0.0021489·var)) + 17.492222`, frame-geomean
  normalized (≈[0.207, 4.832]), raised to the configured strength
  (exponent blend keeps geomean exactly 1: pure spatial rate
  reallocation); every block-RDO cost site (mode / tx-type /
  partition-symbol / split-trial / NONE-breakout / bottomup edge path /
  opt-in trellis λ) scales its rate term by the geomean of covered
  factors. The λ-side counterpart of the distortion-side `ssim_boost`
  activity masking; the two compose. CDEF/LRF search unscaled (aom
  parity); `me_lambda` (intraBC MV search) unscaled in v1. The (a2)
  mechanism of the zenavif tune study (docs/TUNE_SSIMULACRA2_PLAN.md);
  strength fit on the zenavif rd_gap harness before any default flips.
- **Variance Boost strength override + deep-flat ramp + flat quantizer
  rounding bias** (all three default `None` = byte-identical, md5-gated
  local cells vs master incl. `Some(1.0)` == `None`; the zenavif P3
  near-lossless-rescans handoffs, zenavif docs/RD_GAP_VS_LIBAOM.md
  "Near-lossless rescans residual"):
  `EncoderConfig::variance_boost_strength: Option<f64>` overrides the
  fitted `Tune::Ssimulacra2` per-SB boost strength 1.0 (0.0 = boost fully
  off, the historical strength-0 sweep-arm semantics — no delta-q coded,
  segmentation stays); `EncoderConfig::variance_boost_deep:
  Option<(f64, u8)>` ramps the effective strength linearly in `log2(var)`
  from the deep strength at var=1 to the base strength at
  `var >= 2^ceil_log2` (the aom tune=iq {36,64}-style deeper per-SB
  spread on near-flat content, without re-boosting the mid-variance SBs
  the global strength fit measured butteraugli-vetoed on photos);
  `EncoderConfig::quant_rounding_bias: Option<u8>` replaces the fitted
  Valin-method rounding offsets (DC 109 / AC 98,109 / EOB 88 per 256)
  with one flat k/256 offset — `Some(128)` = 0.5-rounding, the libaom
  `sharpness != 0` `av1_build_quantizer` dead-zone-removal path
  (qrounding 48->64 of 128) that both aom tune=iq and tune=ssimulacra2
  quantize with (the 6096 "aom codes 100% of 4x4 cells at baseQ 64 while
  we skip 57.5% at baseQ 54" probe). New validation errors:
  `InvalidVarianceBoostStrength` / `InvalidVarianceBoostDeep` /
  `InvalidQuantRoundingBias`.

- **intraBC hash-based block search (chunk B)** (inert unless `intrabc`
  is armed; zenrav1e#30 item 3, the fam-7 legacy-plot / 8414-screens
  residual owner): port of libaom's `av1_hash_table` exact-match candidate
  machinery (`hash_motion.c` + CRC-32C, pinned rev 632172a4) in
  `src/intrabc_hash.rs` — the tile's source luma is block-hashed once per
  tile encode (2x2 identity/xor-fold base, hierarchical CRC-32C combine,
  8/16/32/64 squares, 2^16 buckets/size, 256-entry caps filled in the
  dispersal order), and square intraBC blocks add up to 64 exact-match
  DVs (nearest-first) to the chunk-A seed/diamond SAD ranking + top-2
  full-rate RD trials. New `PredictionSpeedSettings.intrabc_hash`
  (default `true`, meaningful only with `intrabc`) + `--intrabc-hash`;
  hash-off is byte-identical to pre-chunk-B builds (81/81 gate cells:
  default + `--palette always/auto --intrabc` across 3 images x s{2,6,8}
  x q{60,140,220} vs 0d392334), including the diamond's incidental
  second-candidate updates (the SAD-0 diamond skip is hash-gated).
  In-repo: `tests/intrabc_roundtrip.rs` gains the long-range-repeat
  liveness test (both samplings) and the existing armed roundtrips run
  the hash path by default.

- **Intra mode-RDO budget override** (default-off; zenavif
  FAST_TIER_PARITY_PLAN s4-tier column, the "missing top-5 knob" the P2
  heads report called out):
  `PredictionSpeedSettings.num_modes_rdo_override: Option<u8>` overrides
  the intra-frame mode-decision shortlist length that was hardcoded 7|3
  (rdo.rs `intra_frame_rdo_mode_decision`; 7 required
  `prediction_modes >= ComplexKeyframes` on keyframes). `Some(n)` RDOs the
  top `n` modes (clamped 1..=13; first `n/2` by CDF probability, rest
  re-ranked by SATD — the historical selection shape, only the count
  moves), giving still-image callers the top-5 midpoint between the
  forced-Simple top-3 and ComplexKeyframes top-7. `None` at every preset —
  byte-identical off (6/6-cell md5 vs 39f0ecdd across s{2,6,8} ×
  quantizer{60,140} still-picture).

- **Pruned top-down partition candidate walk** (default-off; zenavif
  FAST_TIER_PARITY_PLAN P1 lever 1, 725f5f71):
  `PartitionSpeedSettings.topdown_prune: Option<TopdownPartitionPrune>`
  re-orders the top-down partition RDO walk NONE-first (the per-child early
  exit then abandons SPLIT/rect/4-way trials against the NONE incumbent) and
  adds four opt-in gates — `none_breakout` (skip everything at a skip-coded
  NONE below τ·λ·pels; libaom `partition_search_breakout` analog),
  `rect_margin`/`four_way_margin` (skip non-square candidates on clear
  NONE dominance over the SPLIT-trial estimate — one-sided since 767c8ff5:
  the original symmetric closeness band forfeited 74% of the rect/4-way
  liveness win on SPLIT-dominant razor-edge content, measured P1PART wave
  1), `homogeneity_gate` (4×4 log-variance deviation port of libaom
  allintra `prune_rect_part_using_4x4_var_deviation`). `None` at every
  preset — byte-identical off (27/27-cell md5 vs d82c16ba across 3 images ×
  s{2,4,6,8,10} × Q{30,85}; 144/144-cell sentinel across the 767c8ff5
  semantics change). Purpose: keep HORZ/VERT (+16-parent 4-way)
  candidates affordable at fast presets whose tables previously amputated
  them (rect threshold 8×8 at s4+). Measured on train26 s6 (zenavif
  `benchmarks/rd_gap_p1part_2026-07-04.tsv`): the margins are a dead end
  in both semantics and the skip-gated breakout is a null at every τ; the
  homogeneity gate is the one gate that pays (94% of the liveness win at
  86% of its cost) — and it is a shape prior, not just a cost gate
  (skipping rect leaves on smooth blocks redirects them into deeper SPLIT
  recursion: better RD at more time than no-gate 4-way-off).

- **Decoupled intra tx-RDO halves + size-depth cap** (default-off; zenavif
  FAST_TIER_PARITY_PLAN P0):
  `TransformSpeedSettings.rdo_tx_size_override` / `rdo_tx_type_override`
  (each `Option<bool>`, `None` = follow `rdo_tx_decision` exactly) split the
  coupled boolean that forced fast presets into `TX_MODE_LARGEST` + DCT-only
  together, and `rdo_tx_size_depth` (`Option<u8>`) caps the intra tx-size
  walk (`Some(1)` = largest + one split level). All three default `None` at
  every preset — byte-identical off (27/27-cell md5 vs ac8c4ef3 across
  s{2,6} × tune-ss2/off × Q{30,60,85}). Measured (zenavif FASTWINS P0,
  train26 s6 tune-ss2): size-only depth-1 recovers 51% of the s6→s4 RD step
  at ~1.5× time; the type half alone fails the butteraugli-max veto. 3,024
  armed cells aomdec+rav1d-safe conformance-clean, including tx-type RDO
  under `TX_MODE_LARGEST` and `TX_MODE_SELECT` at the s6/s8 presets.

### Fixed
- **CLI `--save-config` can no longer abort on unrepresentable config
  shapes** (zenrav1e#2): the last 10 `unimplemented!()` stubs in the
  key-value serializer (`src/bin/kv.rs`) now return a clean
  `Error::Unsupported` — the crate is `unimplemented!()`-free. Regression
  test covers seq/map/tuple/char shapes.

- **Encoder panic when an INTER frame inherits unusable segmentation data**
  (zenrav1e#31, fuzz signature `1e77077d5a3f1d17`): `segmentation_optimize`
  asserted (`min_segment == MAX_SEGMENTS`, segmentation.rs) when the primary
  ref frame carried no ALT_Q segment usable at the current frame's
  `base_q_idx`. Deterministic trigger: `Tune::Ssimulacra2`'s variance boost
  (and the `FrameHints` sb_q_scale path) disables segmentation on KEY/intra
  frames, so the following INTER frame inherited an all-features-false
  `SegmentationState`; also reachable when rate control drops `base_q_idx`
  between frames (the lossless floor rises above every stored delta). The
  encoder now re-signals fresh segment data (`segmentation_update_data = 1`)
  on such frames instead of panicking; rav1d-safe roundtrip-verified
  (tests/segmentation_resignal_roundtrip.rs + fuzz/regression seed).

### Added
- **`FrameHints` — external per-superblock AC-quantizer-scale input**
  (c4047cec): `FrameParameters.frame_hints` carries an optional
  `FrameHints { sb_q_scale: Option<Box<[f32]>> }` per frame (keyframe /
  intra-only scoped); the per-SB scale composes onto any tune-driven map
  (Variance Boost) and is coded through the real per-SB `delta_q` syntax
  with the `(ac_q(base)/ac_q(sb))²` RDO distortion follow. Metric-free by
  design — the first consumer is zenavif's butteraugli-diffmap-guided
  second pass (zenavif `docs/DIFFMAP_TWO_PASS.md`). Absent, all-neutral,
  or grid-mismatched maps are byte-identical to a plain encode
  (api::test contract).
- **Chroma (UV) palette search** (a3b72033): the previously-"off"-coded UV
  palette flag now carries a real joint (U,V) palette — libaom
  `av1_rd_pick_palette_intra_sbuv` 2-D k-means candidates plus a
  dominant-pairs family (2-D analogue of the luma top-colors family;
  k-means-only search measurably misses exact palettes on palette-exact
  content), U colors min-step-0 against the U neighbor cache, V colors
  raw-vs-wraparound-delta by libaom's exact rate arithmetic, one shared
  chroma index map, trialed through the real writers on top of the winning
  luma side. Same `PaletteMode` knob, default Off (byte-identical off).
  Measured vs the luma-palette base: fam-7000 plots −1.95%/−2.59% ssim2-BD
  median (s2/s6), butteraugli agreeing; conformance 200/200 corpus cells
  @420 + 84/84 @444, aomdec + rav1d-safe raw-md5 agreement (zenavif
  `benchmarks/uvpal_ab_2026-07-03.tsv`).
- **Intra block copy (intraBC), chunk A** (bf1f4a13): DV prediction
  (rav1d `decode_b` dual), `av1_is_dv_valid` port (256-px delay +
  wavefront rule), fullpel all-plane copy MC from the tile recon, seeded
  diamond SAD search + top-2 full-rate RD trials; per-block flag + DV
  coding on `allow_intrabc` intra frames, reusing the inter tx/coef path
  (GLOBALMV + INTRA_FRAME ref). Behind
  `SpeedSettings.prediction.intrabc` (default off, byte-identical off,
  80/80 gate cells) and the binary's `--intrabc`; with `PaletteMode::Auto`
  the AA-aware detection's stricter intraBC criterion gates it per frame.
  In-loop filters forced off on allowed frames per spec. Shrinks
  exactly-repeating non-palettizable content to ~0.52x bytes at both
  chroma samplings (tests/intrabc_roundtrip.rs).

- **Size-conditional strength for the Tune::Ssimulacra2 QM-dist ratio**:
  `qm_dist_ratio_m = clamp((log2(long_edge) - 8) / 2, 0.5, 1.0)` — full
  strength at >= 1024 long edge (bit-identical to the previous encoder
  there, md5-gated), log2-linear ramp down to half strength at <= 256.
  The zenavif wedge-#3 size-decay isolation A/B convicted the ratio as
  the only tune mechanism whose win decays on small renditions (leave-
  one-out −3.48% -> −0.96% median ssim2 BD 1024->256, high-q band
  flipping positive at <= 512), and strength trials measured an
  inverted-U: half strength BEATS full at small sizes (train +0.87%
  @512 / +1.03% @256 median vs full, val +1.00 / +1.12, 9-11/12 origins,
  butteraugli agreeing +1.1..+3.3%) while removing the ratio loses.
  Tune-off output byte-identical; conformance-swept (aomdec + rav1d-safe
  raw md5 agreement) across the 36-file x 5-q size ladder. Record:
  zenavif `benchmarks/hyperparam_size_decay_ab_2026-07-03.tsv` +
  `docs/RD_GAP_VS_LIBAOM.md` "Size-decay isolation A/B".

### Fixed
- **CDF undo-log cross-field overspill (latent bitstream-desync class)**
  (e86235b5): the RDO CDF undo log captured/restored fixed 16-word
  snapshots regardless of the CDF's real length, and its small/large
  partitions rolled back sequentially rather than globally LIFO — an
  8-wide `palette_y_color_index_cdf[6][4]` update snapshot spilled over
  `palette_uv_color_index_cdf[0][0]` and rollbacks resurrected stale UV
  CDF state (encoder-only state no decoder reaches ⇒ content-dependent
  desync; silent while the adjacent bytes held constant defaults, i.e.
  for every luma-only palette stream). Exact-length snapshots +
  compile-time bounds; regression test
  `cdf_log_rollback_is_exact_length_across_field_boundaries`.
- **4:2:0 chroma TU grid truncated to zero for 4:1 slivers** (#35): the
  chroma TU-loop bounds in `write_tx_blocks`/`write_tx_tree` shifted each
  mi dimension by the subsampling and patched zeros with a 1x1 fallback --
  correct for the classic 4x4..8x8 pairing shapes, but for BLOCK_16X4 in
  4:2:0 the fallback clobbered the 2x1-mi coded-chroma extent and the
  division by TX_8X4's 2-mi width truncated the loop to ZERO iterations:
  no chroma TUs written (nor predicted into the recon) while conforming
  decoders parse a TX_8X4 TU there (`Subsampled_Size[16X4][1][1] = 8X4`,
  spec 5.11.38). Every 4:2:0 encode choosing HORZ_4/VERT_4 with coded
  chroma desynced (found as zenavif#29: ravif's `--yuv 420` output was
  100% aomdec-rejected; 4:4:4 unaffected, which is why 7d254289's
  110-cell conformance sweep -- run at cavif's default 4:4:4 -- missed
  it). The grid is now derived from `BlockSize::subsampled_size` (the
  spec's paired chroma block size), byte-identical for every
  previously-working shape (36/36 4:4:4 corpus cells md5-equal pre/post).
  Verified 258/258 4:2:0 corpus cells aomdec-clean + aomdec/rav1d-safe
  raw md5 agreement; regression gate `tests/sliver_chroma_roundtrip.rs`
  fails pre-fix (rav1d-safe `InvalidData`) and is liveness-checked.
  Master-only: crates.io 0.1.4 predates the topdown 4-way types.
- **Intra 64x64-parent 4-way slivers now require TX_MODE_SELECT** (1dabba91,
  #34): the 3fa735dc sliver TX cap (TX_64X16/16X64 -> TX_32X16/16X32) is
  decoder-followable only via the written intra tx-size depth -- a
  TX_MODE_SELECT symbol. With `rdo_tx_decision=false` the frame header
  signals TX_MODE_LARGEST, no tx-size symbol exists, and conforming decoders
  (aomdec, rav1d-safe) derived the uncapped sliver transform against capped
  coefficient units -- guaranteed desync ("Corrupted segment_ids" / "Failed
  to decode tile data"). Latent since 7d254289 (HORZ_4/VERT_4 Phase 1);
  reachable via `override_partition_range` max=64 and the stock speed 6-8
  presets (partition max 64 + rdo_tx off) on intra frames.
  `encode_partition_topdown` now offers intra 64-parent 4-way (and nested
  mixed-3-way) candidates only under `tx_mode_select`; both sliver-cap sites
  hard-assert the invariant. Plain 64-dim transforms (TX_64X64/64X32/32X64)
  code consistently under LARGEST and remain offered. Verified: 6/6
  previously-corrupt shapes clean under both decoders, byte-identity at
  shipped default configs, 170 lib tests.
- **Deblock filter + level optimizer honor frame-header `sharpness`**
  (aba01be7): nonzero sharpness (previously only Tune::StillImage's
  schedule) was written to the frame header but ignored by the encoder's
  own loop filter and by `deblock_filter_optimize` — encoder recon
  diverged from every conforming decoder and levels were priced for
  thresholds the decoder would not use. The threshold inversions
  (`limit_to_level`/`blimit_to_level`) now take sharpness, with exact
  const-built inverse tables verified exhaustively against the AV1
  7.14.4 forward map; sharpness is decided once per frame BEFORE tile
  encoding (delayed-loopfilter RDO included) and forced 0 for lossless.
  Sharpness-0 output (every non-StillImage config) is byte-identical
  (18/18-cell md5 gate vs the previous master binary).
- **Filter-intra predictions now byte-match conforming decoders**
  (32477046, zenrav1e#33): `get_intra_edges` prepared DC's edge needs for
  filter-intra blocks (no top-left ever — fed 128 instead of the corner
  pixel; no left column at x==0), and `pred_filter_intra` read the
  bottom-to-top left-edge buffer un-reversed (upside-down left column).
  Every filter-intra block's prediction diverged from the decode,
  compounding to 17-25 luma RMSE at speeds <= 6. Encoder recon now
  byte-agrees with aomdec + rav1d-safe on the repro corpus (RMSE 0.000 at
  s2/s4/s6/s0) and across a 120-cell train26 conformance sweep.
  `--filter-intra false` streams are byte-identical to previous builds;
  feature-on streams change only via RDO/residuals now being computed
  against correct predictions. This was also the "broken cost estimation /
  12 dB PSNR regression" (zenrav1e#5) that made ravif pin
  `complex_prediction_modes: Some(false)` — with the fix, arming
  filter-intra is measurable again (spot: 5048 gray q60 s2+tune, fi-on
  ssim2 59.50 pre-fix -> 75.11 post-fix vs 75.00 shipped).
- **Signaled sgrproj units now always apply to the encoder recon**
  (17cff82f, zenrav1e#32): an inherited 2019 skip left `Sgrproj`
  restoration units unapplied in `lrf_filter_frame` when
  `enable_cdef=false`, while the sgrproj RDO (gated only on
  `enable_restoration`) still selected and signaled them — cdef-off +
  lrf-on configs (API-reachable via pub `SpeedSettings::cdef`; not
  reachable from the CLI) decoded differently than the encoder recon
  (measured luma RMSE 0.387-0.580, now 0.000 vs both decoders).
  Default (cdef-on) configs are byte-identical across the change.
  The *reported* #32 repro (~45 RMSE at s <= 7 on smooth content) was
  the filter-intra bug above: its speed bisect jumped s6->s8, and s7
  (lrf on, filter-intra off) measures 0.000; LRF application itself is
  byte-exact on 27 isolation cells + 120 all-defaults cells.
- **Skipped intra blocks now write their prediction into the recon**
  (b30dd752): `write_tx_blocks` early-returned on skip, so any intra block
  whose residual quantized to zero (coded skip=1) left stale RDO-trial
  pixels in the recon buffer; every later intra prediction chained off
  them — valid bitstreams whose decoded image drifts from the encoder's
  intent (measured luma RMSE 67.7 -> 45.6 end-to-end on a smooth gray
  photo at s2 q100; the remainder was zenrav1e#33). Not byte-identical
  to previous encodes wherever a forced-skip intra block occurred.

### Added
- **`examples/recon_probe.rs`** (17cff82f): encodes a single-frame 4:2:0
  y4m still with direct `cdef`/`lrf` speed-setting control (the CLI does
  not expose cdef) and emits IVF + the encoder's reconstruction, for
  recon-vs-decoder byte-agreement probing.
- **Loop-filter sharpness schedule for `Tune::Ssimulacra2`** (zenrav1e#30
  item 1): the tune now codes frame-header deblock sharpness {7,5,3} at
  base_q_idx {<80, <160, else} — the schedule Tune::StillImage already
  used — after a 4-arm A/B (sharpness 0; aom's constant 7; tune-IQ's
  {7,1,0}@{112,160} qindex clamp; this schedule) on the zenavif rd_gap
  harness with the mandatory butteraugli veto. Measured (train26, cavif
  s2+tune, full 12-pt grid, direct isolation): ssim2 BD −0.43% median /
  −0.47% mean (better 19/24); legacy photos −0.67% median (16/19) with
  butteraugli flat (ba3n +0.00%, bamax −0.12% med). First tune ingredient
  where butteraugli's sign diverges from ssim2 on train26 (+0.11/+0.29%,
  far under the +1.0/+1.5 veto) — sharpness trades a small blocking cost
  for a larger edge-retention win; aom ships constant 7 for SS2/IQ on
  subjective-sharpness grounds. const-7 tied on ssim2 (Δ0.0015%) and lost
  the pre-registered ba3n tie-break; the adaptive clamp missed the −0.3%
  ship bar (−0.23%). 110/110-cell aomdec+rav1d-safe conformance at s2 and
  s1-deep with the schedule armed. Record: zenavif
  benchmarks/rd_gap_lfsharp_2026-07-03.tsv + docs/RD_GAP_VS_LIBAOM.md.
- **QM-weighted RD distortion for `Tune::Ssimulacra2`** (the
  `dist_metric=AOM_DIST_METRIC_QM_PSNR` analog, adapted): the tune's luma
  RDO distortion is now scaled by the per-block QM-weighted / unweighted
  transform-domain error ratio — the frequency-dependent error forgiveness
  QM dequantization actually applies, composed with (not replacing) the
  Psychovisual activity-masked pixel metric. Forward weights derive from
  the ported inverse tables (`QM_FWD_WEIGHT[iwt] = round(1024/iwt)`,
  verified equal to libaom's stored `wt_matrix_ref`); the lookup shares
  `dequantize_with_qm`'s storage-order indexing so the zenrav1e#29
  orientation fix carries over by construction. Measured (zenavif train26,
  cavif s2+tune, 12-pt grid): **ssim2 BD −1.78% median / −1.45% mean
  (better 15/24), butteraugli 3n −1.46% / max −0.37%, ~1.01× encode
  time**; s1 −1.71%/−1.52% with all butteraugli norms agreeing. Legacy
  tier-2 confirm: gap vs aom cpu0 --tune=ssimulacra2 +5.63% → +2.12% (s2)
  and +5.02% → **−1.94% (s1) — the tier-2 median crosses**; o_6629 (the
  residual coefficient-RD outlier) −13.5/−15.3% direct. libaom's literal
  routing — force tx-domain distortion and weight it — measured +4.47%
  median WORSE here (the domain switch alone +6.07%: cdef_dist's activity
  masking is worth more than tx-domain SSE; the weighting itself −2.57%
  inside that frame, which motivated the ratio composition). Running the
  trellis unconditionally under the tune re-measured as a regression
  (+0.3..0.6%, 1.66× time) and stays out; when the user-opt-in trellis
  runs under the tune its coefficient distortion now uses the same
  forward-QM weighting (`get_coeff_dist` analog, ≈0 ssim2 vs unweighted,
  softer butteraugli max). Byte-identity for every non-`Ssimulacra2`
  config and for tune-off verified against the previous master binary.
  Record: zenavif benchmarks/rd_gap_qmdist_2026-07-03.tsv +
  docs/RD_GAP_VS_LIBAOM.md "QM-weighted RD distortion".
- **AV1 palette mode** (68a8d81f, 5f82e2d4, cda831e7): full encoder-side
  implementation of the screen-content palette tool, default OFF behind
  `SpeedSettings.prediction.palette` (`PaletteMode::{Off, Auto, Always}`,
  `--palette` on the CLI). Luma palettes of 2-8 colors: libaom-ported
  top-color + k-means search with neighbor-cache snapping, cache-aware
  color coding (reuse bits + shrinking-width deltas), wavefront index-map
  coding with the spec's neighbor-context color ordering (exhaustively
  unit-tested against the score-based spec formulation), palette recon
  (`pal_pred` dual) including the zero-residual skip path, and RD trials
  through the real bitstream writers (libaom's `discount_color_cost`
  overuse bias deliberately not ported, see their b:421196988).
  `PaletteMode::Auto` ports libaom's anti-aliasing-aware screen-content
  detection (16x16 color counting + dominant-value dilation) and decides
  `allow_screen_content_tools` per key frame. Byte-identical when Off;
  240 palette-on cells aomdec-clean with aomdec==rav1d-safe md5 output
  agreement; measured (train26 24 images x 5 q x 2 speeds, BD-rate at
  matched ssim2): plots -31.7%(s2)/-79.4%(s6) median bytes, screenshots
  -15.3%/-82.5%, scans -14.8%/-36.5%, with butteraugli-p3 agreeing.
  UV palette search and palette-in-inter-frames are not implemented (the
  UV flag is coded "off"; both are conformant omissions).
- **Per-superblock delta-q coding** (d125713f): the encoder can now code real
  `delta_q` syntax — frame-header `delta_q_present`/`delta_q_res` (spec
  5.9.17), the per-SB `delta_q_index` symbol at the first block of each
  superblock (spec-exact skip-SB omission and per-tile qindex predictor),
  the `delta_q_cdf` with dav1d-matching defaults, and per-SB qindex plumbed
  through quantize/dequant/rate via `get_qidx` composing with segmentation
  exactly like `init_quant_tables`. Includes a checkpoint fix for the
  rollback-unprotected `code_deltas` flag (same side-state class as
  zenrav1e#27) that any delta coding would have desynced on. Inert unless a
  frame sets `delta_q_present` — byte-identity vs previous master verified
  across tunes.
- **Variance Boost per-SB delta-q for `Tune::Ssimulacra2`** (66733720): port
  of libaom's `DELTA_Q_VARIANCE_BOOST` (allintra_vis.c rev 632172a4,
  SVT-AV1-PSY lineage) through the real delta-q syntax — flat/fine-gradient
  superblocks get a finer quantizer (octile-5 smoothed 8×8 variance, aom
  boost curve + qindex damping, res 1/2/4/8 by base qindex), with per-SB
  RDO distortion follow `(ac_q(base)/ac_q(sb))²` and segmentation disabled
  while active (the segmentation-channel variant double-boosted flats,
  +1.92% ssim2 BD). Replaces the "Variance Boost measured as a regression"
  status below — that verdict was about the segmentation channel.
  Strength offline-fit (follow-up commit): swept {off,1,2,3,4.5,6} + a
  keep-segmentation arm on the zenavif train26 corpus (24 train-split
  origins × 12 q, cavif s2+tune, BD vs boost-off, butteraugli veto) —
  **strength 1.0 ships**: ssim2 median −2.34% / mean −2.24% (19/24
  better), butteraugli agreeing (3n −1.13%, max −0.76%); 4.5/6.0 and
  keep-segmentation were butteraugli-vetoed (max +4.5..+5.5%). libaom's
  3.0 default over-boosts here because the Psychovisual pipeline already
  activity-masks distortion. 110/110-cell conformance (aomdec +
  rav1d-safe) at both s2+tune and s1-deep+tune, at strength 3.0 AND at
  the shipped 1.0. Record: zenavif benchmarks/rd_gap_deltaq_2026-07-02.tsv.
- **`Tune::Ssimulacra2`** (a37faea8): SSIMULACRA2-tuned still-image mode
  porting the two libaom `--tune=ssimulacra2` mechanisms that measured as
  wins on top of the Psychovisual pipeline — aom-parity chroma delta-q by
  subsampling, and the SSIMULACRA2 QM level curves with QM always on.
  Composed: −4.28% median ssim2 BD-rate vs tune-off at cavif s2 on the
  22-image rd_gap corpus (butteraugli agrees: 3-norm −2.53%); flips the
  gap to aom cpu0-default from +1.47% to −3.43% median. aom's all-intra
  rdmult weight, sharpness-7 trellis, and Variance Boost delta-q measured
  as regressions on this encoder and are deliberately excluded (full A/B
  record: zenavif docs/TUNE_SSIMULACRA2_PLAN.md). Tune-off output is
  byte-identical to previous behavior.

### Fixed
- **`qm_v` omitted whenever a QM frame's u/v delta-qs coincided** (9a8eaf61).
  AV1 5.9.12 codes `qm_v` iff the sequence header's `separate_uv_delta_q`
  (always 1 here); gating it on the frame-level `diff_uv_delta` produced
  streams aomdec rejects and dav1d-lineage decoders silently mis-parse.
  Masked by the Daala chroma offsets (u ≠ v almost always); exposed by
  `Tune::Ssimulacra2`'s aom-style chroma delta-q (u == v). zenrav1e#29.
- **Rectangular transforms quantized with transposed QM weights** (2310c7be).
  rav1e stores coefficients transposed (like dav1d); `qm_table()` didn't swap
  w/h the way rav1d-safe's `dav1d_qm_tbl` mapping deliberately does, so every
  rect TX used the transposed matrix — self-consistent in the encoder, wrong
  on every conforming decoder. Small at the near-flat levels 12–15 the
  existing curve picks, catastrophic at steeper levels (decoded ssim2
  85.7→55.7 at cavif Q85 with the ssimulacra2-tune curve). Adds
  transpose-pair + rav1d-reference spot tests. zenrav1e#29.
- **64×64-parent `HORZ_4`/`VERT_4` slivers emitted corrupt bitstreams** (3fa735dc).
  `BLOCK_64X16`/`BLOCK_16X64` — reachable only via `override_partition_range`
  up to 64 — desynced every conforming decoder through their never-validated
  `TX_64X16`/`TX_16X64` max transforms. Intra slivers now cap to
  `TX_32X16`/`TX_16X32` (spec-valid; decoder follows the written depth), the
  tx-size RDO walk shrinks by the consumed level (an out-of-alphabet depth-3
  symbol otherwise), inter frames without `enable_inter_txfm_split` no longer
  offer 64×64-parent 4-way candidates, and the intra tx-depth bound is a hard
  assert in all builds. Byte-identical at every shipped preset; validating the
  real 64-dim sliver transforms is #28.

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
- **`PartitionSpeedSettings::split_trial_depth`** — the b073182c one-level
  SPLIT-trial cost refinement generalized to a recursion-depth knob (default
  `1` = unchanged, byte-identical; `0` treated as `1`). Depth 2 applies the
  same `min(NONE leaf, deeper SPLIT)` comparison one level further down,
  sharpening SPLIT-vs-large-block ranking where wide partition ranges are
  searched — an opt-in for deep/max-quality modes (cavif `-s1`, #27).
- **`PARTITION_HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B` in the RDO search (#27,
  Phase 2 of extended AV1 partition types)** — the four mixed-granularity
  3-way splits (one half stays a single block, the other half splits again
  into two square quarters), completing all 6 extended types on top of
  Phase 1's uniform 4-way splits. Same gating as Phase 1
  (`non_square_partition_max_threshold` + exact `{16X16,32X32,64X64}` size +
  strict full containment), **plus a new opt-in speed setting
  `PartitionSpeedSettings::mixed_3way_partitions` (default `false` at every
  preset)**: the wider search costs ~1.5× encode time on stills, beyond the
  matched-speed budget of the default operating point, so consumers enable
  it explicitly for deep/max-quality modes (cavif `-s1`). With the knob off
  the search is byte-identical to before this change (verified on 9
  image×quality cells). Implementation surfaced two real bugs, both fixed
  in the same change:
  - Every child of these types is an unconditional **leaf** per spec (libaom's
    `decode_partition` calls `decode_block` for all 3 sub-blocks — none get a
    fresh `read_partition`), but the square "split again" quarters — the only
    square children reachable through the topdown recursion that DON'T carry
    their own partition syntax (SPLIT's do) — were re-entering the partition
    search and writing an extra, illegal partition symbol: a bitstream desync
    `aomdec` rejects as "Corrupted segment_ids". Fixed via forced-leaf
    threading in `encode_partition_topdown` (no fresh decision, no partition
    symbol, no per-child partition-context update; the parent does libaom's
    `update_ext_partition_context`-style two-region update instead).
  - `encode_tx_block` read the block's parent-split tag from
    `cw.bc.blocks[bo].partition`, rollback-unprotected side state that RDO
    trial paths calling `write_tx_blocks`/`write_tx_tree` directly never
    refreshed — a stale read from a previous, rolled-back trial at different
    geometry (repro'd as the impossible pair `(VERT_B, BLOCK_16X8)` indexing
    an empty `has_tr_vert_tables` slot → index-out-of-bounds panic). The
    parent split is now threaded as an explicit parameter down
    `encode_block_post_cdef` → `write_tx_blocks`/`write_tx_tree` →
    `encode_tx_block` (mirroring libaom's always-current `mbmi->partition`),
    and the previously-dormant VERT_A/VERT_B traversal-order tables
    (`has_tr_vert_tables`/`has_bl_vert_tables`, present-but-disabled since
    the original rav1e port) are now live in
    `has_top_right`/`has_bottom_left`.
  Verified: 22-image photo corpus × 5 quality levels = 110 cells, 100%
  `aomdec`-clean (0 corrupt), rav1d-safe roundtrip pixel diff scaling
  normally with quality; all 4 types chosen by the search on 9/9 sampled
  cells (1,282–1,399 chosen instances each across 3 images × 3 qualities).
  RD impact measured in the `zenavif` sibling repo's
  `docs/RD_GAP_VS_LIBAOM.md`.
- **`PARTITION_HORZ_4`/`PARTITION_VERT_4` in the RDO search (#26, Phase 1 of
  extended AV1 partition types)** — `encode_partition_topdown` can now choose
  the two uniform 4-way splits for `BLOCK_16X16`/`32X32`/`64X64` blocks fully
  contained within the frame, gated by `non_square_partition_max_threshold`.
  Previously 0 of the 6 "extended" AV1 partition types were ever attempted by
  the RDO search at any speed; this closed 2 of the 6 (7d254289), and Phase 2
  above closes the remaining 4. Measured
  extended-block-size area share 1.8-56% per cell across a 22-image photo
  corpus x 5 quality levels (110 cells), `aomdec`-clean (verified against the
  ordinal-comparison fix above, without which the same feature produced a
  bitstream `aomdec` rejects as corrupt).
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
