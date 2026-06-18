# Trellis RDOQ + RDO rate-estimator — goal to completion

## North star
Ship two **opt-in, off-by-default** encoder features in zenrav1e, each landing on
`master` **only after a clean, un-confounded rate-distortion gate passes**:

1. **Stage 1 — a real trellis RDOQ.** Turn the trellis from a weak heuristic
   (single-step round-down, ~0.3% BPP, shelved for cost) into a multi-level
   context-accurate RDOQ that, now that its rate model is ~6× cheaper (the LUT
   already on master), is *worth enabling*.
2. **Stage 2 — a closed-form coefficient-rate estimator in mode-decision RDO.**
   Replace the ~100–1000×/superblock "encode-to-counter + rollback" rate
   measurement with a closed-form estimate for a large encoder speedup at a
   small, documented BD-rate cost.

A feature ships **or** is killed with a measured-and-verified reason. Both are
valid ends. Master never takes an unmeasured RD change.

## Definition of done
- [ ] A **trustworthy RD methodology** exists and is committed (constant-QP
      harness + dense sweep + PSNR *and* perceptual + BD-rate).
- [ ] Stage 1: either merged opt-in with a committed BD-rate win and no low-q
      regression, **or** documented-and-dropped (master unchanged).
- [ ] Stage 2: either merged opt-in with a measured speedup at an agreed BD-rate
      cost, **or** documented-and-dropped.
- [ ] The untested-trellis gap is closed (a trellis-on encode→decode roundtrip
      test in CI).
- [ ] CHANGELOG / docs / `benchmarks/` updated; branch removed after merge.

## Current state (2026-06-17)
- **master `24d77fe9`**: LUT'd `cdf_rate` (5.8×, byte-identical). Trellis off.
- **branch `feat/trellis-rdoq`**: multi-level RDOQ + exact monotonic early-break.
  Decode-safe, 16 unit tests pass, clippy clean. **RD UNMEASURED.**
- **Blocker**: the fixed-`quantizer` A/B harness is confounded — rav1e's default
  adaptive quant/segmentation reacts to the trellis, so even the trellis-OFF
  Y-PSNR is non-monotonic in q. Cannot attribute any RD delta to the trellis.
  Reliable signal so far: trellis-ON shaves 0.5–1.9% bytes at +50–80% encode time.

## Milestones (ordered; each is demoable and gated)

### M0 — Clean measurement harness  ← unblocks everything
Force a single constant QP with **no spatial adaptation** (disable VAQ,
segmentation, delta-q, adaptive rounding redistribution) so the *only* variable
between A and B is the trellis's level changes.
- **Gate:** trellis-OFF Y-PSNR is **monotonic in QP** (proves the confound is
  gone). Until this holds, no Stage-1 verdict is valid.
- Then: dense QP sweep over the trellis-active range, multi-content corpus
  (photo / screen / line-art), score PSNR + ssim2/zensim via `zen-metrics`,
  compute BD-rate. Commit harness + methodology to the branch.

### M1 — Stage 1 verdict + calibration
- Run M0 on the multi-level RDOQ. BD-rate (PSNR + perceptual) vs **two**
  baselines: trellis-off, and the old single-step trellis (build from master).
- If positive: sweep-calibrate `lambda_trellis` + dampening across q to maximise
  BD-rate with **no low-q regression**; re-measure.
- **Gate:** beats the old 0.3% meaningfully (target 1–3% BD-rate), zero regression
  at any q, encode-time acceptable. If unreachable → **stop, document, master
  stays put.**

### M2 — Stage 1 hardening + merge
- Add the trellis-on encode→decode roundtrip test (close the CI gap).
- Optional refinement: floor=0 interior-zeroing with incremental context updates
  (the larger RDOQ win); measure separately, ship only if it adds BD-rate.
- Full CI (asm + no-asm, clippy, all platforms), bench results committed to
  `benchmarks/`, user review → merge opt-in.

### M3 — Stage 2: closed-form RDO estimator
- Extract `estimate_block_coeff_rate` from the (closed-form, LUT-fast)
  `coeff_rate` machinery; unit-test it against real `write_coeffs_lv_map` output
  and **bound the estimate error**.
- Wire into `rdo.rs` behind an opt-in flag (`rdo_tx_type_decision` first, then
  `luma_chroma_mode_rdo`), replacing encode-to-`WriterCounter` + `rollback`.
- **Gate:** measured encoder speedup (target multi-×) at a BD-rate cost within an
  agreed budget; opt-in only. Else document-and-stop.

### M4 — Close-out
CHANGELOG, docs for the opt-in flags + when to use them, `benchmarks/INDEX`,
remove the branch after merge, final measured summary.

## Risks / kill-criteria (honest)
- RDOQ may not beat the baseline even calibrated → kill Stage 1; the shipped LUT
  is already a clean win, so that's an acceptable end.
- Estimator BD-rate cost may exceed budget → narrow its scope or kill Stage 2.
- Encode-time may make either not worth enabling even if RD-positive → keep
  opt-in/off and document the tradeoff.

## Invariants (always)
- **Branch-only until a gate passes.** Master never takes an unmeasured RD change.
- **No external-project names** in anything committed (code, messages, tests,
  benches). Techniques are described on their own terms.
- **"Done" is verified on `origin`** (`git show master@origin`), never claimed.
- Heavy builds/sweeps run under `scripts/run-heavy`.
