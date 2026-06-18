# Stage 2 — closed-form RDO rate estimate: measured verdict (2026-06-18)

Branch `feat/rdo-estimate`. Goal: replace RDO's encode-to-counter+rollback coeff-rate
with a closed-form estimate for a LARGE encoder speedup at a small BD-rate cost.
Measured on 38 real photos (trellis off, constant-QP, reconstruction PSNR), Hetzner box.

| rate model | where | enc speedup | BD-rate cost(Y) | verdict |
|---|---|--:|--:|---|
| real coding (default) | — | 1.00× | 0% | baseline |
| table estimate (`tx_domain_rate`, existing) | mode-decision | 1.9× | **+29.8%** | fast, RD-catastrophic |
| per-coeff closed-form | mode-decision direct | 1.06× | +6.1% | accurate, barely faster |
| per-coeff closed-form, **double-count bug** | `rdo_tx_type_decision` | 1.64× | +10.9% | speedup was the bug |
| **per-coeff closed-form, fixed** | `rdo_tx_type_decision` | **~1.05×** | **−0.02%** | RD-neutral, ~5% faster |
| **per-coeff closed-form, fixed** | mode + luma_chroma + tx_type (ALL) | **~1.05×** | **+0.33%** | RD-neutral, full ceiling ~5% |

## The decisive finding (large-speedup premise measured-FALSIFIED)
The new `estimate_block_coeff_rate` (trellis.rs) — same per-coefficient CDF rate model the
trellis uses — is **RD-neutral**: routed into the heavy `rdo_tx_type_decision` lever it picks
essentially the same tx-types as real coding (−0.02% BD-rate, range ±0.3%). The estimator is
*accurate*. But the **encoder speedup is only ~5%**, because coefficient entropy-coding is only
~5% of the RDO cost — the forward/inverse transform + quantize + reconstruct + distortion
(which `write_tx_blocks` does for every candidate regardless) dominate, and the estimate does
not avoid them.

The *apparent* 1.64× speedup (before the fix) was a **double-count bug**: when
`need_recon_pixel` is set, `encode_tx_block` already real-codes the coeffs to `w`, and the
EstRate block added the estimate on top — inflating the rate, which biased RDO toward
cheaper-to-encode (worse, +10.9%) decisions that were also faster. Once the estimate is only
added when real coding is skipped (`encoder.rs:1924`, `&& !need_recon_pixel`), decisions match
real coding and the speedup collapses to the true ~5%.

Routing **all three** RDO rate paths (mode-decision + `luma_chroma_mode_rdo` + `rdo_tx_type_decision`)
through the estimate at once does not move the needle: +0.33% BD-rate, still ~5% faster. The ~5%
speedup ceiling holds regardless of coverage — conclusive across 5 measured configurations.

**Conclusion:** you can have an *accurate* coeff-rate estimate (RD-neutral) or a *fast* one
(table, +30% BD-rate), but not a large speedup with small BD-rate cost — because the entropy
coding the estimate replaces is not the RDO bottleneck (it is ~5% of RDO cost; transform,
quantize, reconstruct, and distortion dominate and are unavoidable in the candidate loop).
This is the measured-and-verified reason the Stage-2 "large speedup" goal cannot ship as
conceived. `estimate_block_coeff_rate` is nonetheless validated as a correct, RD-neutral
block-rate estimator (the same model that earned Stage 1's −0.94%).

## What IS validated / available
- `estimate_block_coeff_rate` is a correct, RD-neutral block-rate estimator (a useful building
  block; the same model earned Stage 1's −0.94%). Env-gated probes:
  `ZENRAV1E_CLOSED_FORM_RATE=1`, `ZENRAV1E_TXTYPE_ESTIMATE=1`.
- A modest (~5%), RD-neutral opt-in speedup is reachable in `rdo_tx_type_decision`. Shipping it
  as a user-facing flag would be a public-API addition (needs approval) for a small gain.

## Remaining lever (untested, predicted small)
`luma_chroma_mode_rdo` (rdo.rs:886-1027) also real-codes per mode candidate. Routing it through
the estimate could add a few % more, but the ~5%-of-cost evidence predicts diminishing returns,
and it is more invasive (monolithic mode+tx+coeff encode). Not pursued.

Data: stage2_{realrate,tableest,closedform}_2026-06-18.tsv + stage2_txtype_fixed (in /mnt/v).
