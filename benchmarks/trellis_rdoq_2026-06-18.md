# Trellis multi-level RDOQ — BD-rate measurement (2026-06-18)

Branch `feat/trellis-rdoq`. Measures the multi-level round-down + interior-zeroing
RDOQ (`src/quantize/trellis.rs`, opt-in `enable_trellis`) vs trellis-off.

## Headline: the test corpus dominates the verdict

| corpus | mean BD-rate %(Y) | per-image range | mean byte% | enc time |
|---|--:|--:|--:|--:|
| **real photos** (10× CID22-512) | **−1.04** | −0.40 … −1.98 (all wins) | −3.06 | 1.75× |
| non-photo (painting/figure/screen/line-art) | −0.13 | −1.06 … **+1.28** (mixed) | −0.5 | 1.7× |

On photographic content — what AVIF targets — the trellis is a **consistent −1.04%
BD-rate win, positive on every image**, in the low end of the 1–3% goal range. On
non-photographic content it is marginal and mixed: it helps smooth paintings but
*regresses* born-digital/high-detail figures (osti_fig +1.28%), because flat regions
and hard edges give RDOQ few small coefficients to profitably optimize and zeroing
detail coefficients hurts. Photos have the broadband HF detail + sensor noise that
RDOQ exploits. **An earlier non-photo-only corpus gave a misleading −0.13% "marginal"
verdict; real photos were missing.**

## Methodology (two confounds caught and fixed)

1. **Constant QP.** Disable segmentation / VAQ / `tune=StillImage`; `tune=Psnr`. Without
   this, rav1e's adaptive per-block QP makes even trellis-OFF PSNR non-monotonic in q.
2. **Reconstruction PSNR.** Measure source vs the encoder's own reconstruction
   (`pkt.source` vs `pkt.rec`), NOT a third-party decode — a decode mismatch on some
   blocks otherwise corrupted PSNR (a localized band of ±100-level luma error → flat
   ~33 dB at all bitrates). With reconstruction PSNR, q10-OFF reads a sane 53 dB and the
   OFF curve is monotonic.

Harness: RGB8 → I420(BT.601 full) → rav1e still encode (off vs on) at constant QP →
Y/U/V PSNR vs reconstruction. q-grid {20,40,60,80,100,120,140}. Speed preset 4.

## Config measured
- Multi-level descent: each coefficient evaluates candidate levels down to its floor,
  keeps the rate-distortion min, with an exact monotonic distortion early-break.
- floor=0 (interior coefficients may zero); EOB coefficient stays ≥1.
- Rate model: the LUT-fast CDF rate (`cdf_rate`); distortion: transform-domain MSE.

## Cost
+75% encode time on these stills (the trellis runs per RDO candidate, not just the
final coefficients).

## Data
- `trellis_rdoq_photos_2026-06-18.tsv` — real-photo RD (CID22-512).
- `trellis_rdoq_nonphoto_floor1_2026-06-18.tsv` / `_floor0_` — non-photo RD.
- Harness + analysis: `/mnt/v/output/trellis-rdoq-measure/` (trellis_bench_harness.rs, bdrate.py).

## Status / next
Stage 1 passes the gate on real photos. Open: pick best config (floor / lambda) on the
photo corpus, add a trellis-on encode→decode roundtrip test, full CI, then merge opt-in.

## Update: 38-photo per-class verdict + lambda sweep (Hetzner, 16-core)

Re-run on 38 real photos (24 CID22-512 + 14 imageflow web classes), 3 lambda scales,
parallel on a dedicated box. Default lambda is the calibrated optimum.

| lambda scale | OVERALL BD-rate%(Y) |
|---|--:|
| 0.5  | −0.39 |
| 0.75 | −0.75 |
| **1.0 (default)** | **−0.94** |
| 1.5  | +0.15 |
| 2.0  | +1.83 |

Per-class @ scale 1.0 (all wins, no regression):
- unsplash-textures −1.89, unsplash-people −1.08, art-objects −1.04, photo-cid22 (n=24) −0.99,
  lilith-food −0.59, lilith-nature −0.57, lilith-photos-general −0.52, lilith-interiors −0.40.

**Stage 1 gate: PASS on photographic content.** −0.94% mean (≈3× the old single-step 0.3%),
regression-free across all 8 photo classes, +72% encode time, lambda calibrated-optimal.
Data: `trellis_rdoq_photos38_s*.tsv`.
