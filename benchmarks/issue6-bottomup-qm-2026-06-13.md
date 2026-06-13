# Issue #6 — bottom-up partition RDO vs top-down: measured negative result

**Status:** outcome (B) — every approach tried **fails to beat top-down**, so no
fix is landed and the ravif workaround (`encode_bottomup = Some(false)` for all
speeds, cavif-rs `40ddb66`) stays. This doc records what reproduced, what was
falsified, and where the cause actually lives, so the next attempt skips the
dead ends.

**Date:** 2026-06-13 · **zenrav1e:** master `10204699` (clean — no fix landed) ·
**Metric:** zensim **profile-A** (`ZensimProfile::A`, codec-target bake, external
name `zensim-a`; zensim 0.3.0) · **Harness:** `zenavif/examples/encode_sweep.rs`
`--force-bottomup both`, local zenrav1e via `[patch.crates-io]`.

## TL;DR

- Bottom-up **never meaningfully beats top-down** in the full proving sweep
  (4 speeds × q5..=100:5 × QM{off,on} × 2 content classes). Its only "wins" are
  +1.0/+1.4 noise at q10 where both encodes score 3–5 (garbage).
- On **photo** (CID22 1001682), bottom-up+QM loses ~3–8 zensim across q40–q95
  (the originally-reported bug; ~5 mean at s1/s2). QM-off is closer but still
  net-negative on the full grid.
- On **sci-figure** (synthetic/line-art), bottom-up is **catastrophic: −30 to
  −56 zensim at every cell, with QM *and without it*** — so the problem is far
  broader than the "× QM" framing in the issue title.
- The signature everywhere is an **operating-point shift, not efficiency**:
  bottom-up emits *slightly smaller* files at *much worse* quality (e.g. sci-fig
  s1 q60 QM-on: top-down 14467 B / 85.3, bottom-up 13970 B / 52.2).
- Bottom-up is also **~3–4× slower** at speeds 1–2 (the speeds where it is the
  default tier).

## The issue's root-cause hypothesis (`ts.rec` not rolled back) is FALSIFIED

Three independent lines of evidence:

1. **Restore `ts.rec` to pre-NONE (entry) state — regresses hard.** The
   abandoned Option A (dangling commit `849888f3`, never on master) restored the
   footprint to its pre-reconstruction state before each probe; per the issue
   thread it crashed quality to ~2.5 (zeroing the neighbour preview is worse
   than any reconstruction).
2. **Restore `ts.rec` to the post-NONE reconstruction — no improvement.** Tried
   here: checkpoint *after* the PARTITION_NONE probe (a real, useful preview)
   and restore to it before each split probe so all probes see the same
   footprint-interior preview. Net wash vs baseline, still 4–9 below top-down
   (`issue6-fix_v1-postNONE-restore.tsv`): e.g. s1 q60 33.5→59.2 but s2 q60
   59.8→54.6.
3. **Top-down also never rolls back `ts.rec`.** `rdo_partition_decision`
   (`src/rdo.rs:2147`) checkpoints/rolls back **`cw` + the bit writers only** —
   identical to bottom-up (`src/encoder.rs:2955`). Yet top-down+QM does not
   regress. The missing `ts.rec` rollback is therefore shared by both paths and
   cannot be the differentiator.

Also falsified: **disabling early-exit under QM** — no improvement, only slower
(`issue6-probe-earlyexit-disable.tsv`; s2 q40 got *worse*).

## Where the cause actually lives: the cost-evaluation path

Bottom-up and top-down use **different cost machinery**:

| | NONE cost | split/rect cost | partition set |
|---|---|---|---|
| top-down (`encode_partition_topdown`) | `rdo_partition_none` | `rdo_partition_simple` | `{NONE, SPLIT}` |
| bottom-up (`encode_partition_bottomup`) | inline `rdo_mode_decision` | recursive `encode_partition_bottomup` (Σ children) | `{NONE, HORZ, VERT, SPLIT}` |

**Confirmatory experiment** — restrict bottom-up to `{NONE, SPLIT}` (skip the
bottom-up-exclusive rectangular partitions), QM-on, vs top-down
(`issue6-probe-norect-{photo,scifig}.tsv`):

| cell | top-down | bottom-up (full) | bottom-up `{NONE,SPLIT}` |
|---|---:|---:|---:|
| PHOTO s1 q60 | 62.9 | 54.7 (−8.2) | **62.4 (−0.5)** |
| PHOTO s2 q30 | 23.9 | 17.1 (−6.8) | **25.3 (+1.4)** |
| PHOTO s1 q90 | 87.7 | 82.1 (−5.6) | 85.5 (−2.2) |
| SCIFIG s1 q60 | 85.3 | 52.2 (−33.1) | **38.4 (−46.9, worse)** |
| SCIFIG s1 q90 | 81.1 | 41.4 (−39.7) | **30.9 (−50.2, worse)** |

Two conclusions:
- On **photo**, the rectangular HORZ/VERT partitions are a major *secondary*
  contributor: removing them recovers bottom-up to near-parity with top-down.
- On **sci-figure**, `{NONE,SPLIT}` bottom-up stays catastrophic (38.4) while
  top-down with the *same* partition set scores 85.3. Same partitions, ~47-point
  gap ⇒ **the root cause is the cost-evaluation path itself** (bottom-up's
  `rdo_mode_decision` + recursive child-cost summation produces systematically
  worse partition/mode decisions than top-down's `rdo_partition_*`), not the
  partition set, not `ts.rec`, not early-exit. QM and rectangular partitions
  amplify it; neither is necessary.

## Recommendation

Keep the ravif workaround — and note it is *more* justified than #6 implied:
bottom-up is broadly inferior to top-down, not merely on photos with QM. A real
fix (outcome A) requires reworking bottom-up's cost path — most plausibly
routing its split/rect evaluation through the same
`rdo_partition_simple`/`rdo_partition_none` machinery top-down uses — which is a
change to core RDO affecting **every** encode and must clear a full
multi-content RD + speed sweep before landing. The cheap fixes (`ts.rec`
rollback, early-exit, dropping rectangular partitions) are all falsified above:
none beats top-down.

## Data

All numbers measured under `run-heavy` (no extrapolation); profile-A; every row
carries encode-time, size, compression ratio, and zensim.

- **Full proving sweep** (speeds {1,2,4,6} × q 5..=100:5 × QM{off,on} ×
  bottom-up{off,on}): `issue6-photo-clean-2026-06-13.tsv` (CID22 photo
  `1001682`), `issue6-scifig-clean-2026-06-13.tsv` (sci-figure
  `osti-3011990-010`).
- **Falsification probes:** `issue6-fix_v1-postNONE-restore.tsv`,
  `issue6-probe-earlyexit-disable.tsv`,
  `issue6-probe-norect-{photo,scifig}.tsv`.

### Full-sweep bottom-up vs top-down (zensim Δ = bottom-up − top-down)

| image | QM | speed | mean Δ | min | max | beats top-down at any q? |
|---|---|---|---:|---:|---:|---|
| photo | on | 1 | −5.4 | −11.2 | +1.0 | no (only q10 noise) |
| photo | on | 2 | −5.3 | −14.3 | +1.4 | no (only q10 noise) |
| photo | off | 1 | −0.9 | −7.6 | +2.0 | no (net negative) |
| sci-fig | on | 1 | −38.6 | −56.6 | −19.0 | no |
| sci-fig | off | 1 | −32.8 | −54.4 | −14.6 | no |

(Full per-speed breakdown in the TSVs.)
