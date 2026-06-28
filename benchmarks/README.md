# zenrav1e benchmarks

Two kinds of measurement live here:

1. **Compression (RD) results** — BD-rate of zenrav1e's still-image features
   versus upstream rav1e, the numbers quoted in the top-level README. These come
   from research harnesses (recorded per file); the raw per-image data is
   committed as `.tsv` so the verdicts are auditable.
2. **Kernel microbenchmarks** — the in-repo `../benches/` Criterion suite, fully
   reproducible from this repo.

All numbers here are **measured, not estimated**. No `-C target-cpu=native` is
used (runtime SIMD dispatch is what ships). Negative results are kept on purpose:
most knobs that were tried did **not** make the default config, and the data
shows why.

## Compression results (BD-rate vs upstream rav1e)

- **Metric:** SSIMULACRA2 / zensim (per file); BD-rate negative = smaller files
  at equal quality.
- **Baseline:** upstream rav1e, same source images, dimensions, pixel format,
  chroma subsampling, and quality/speed target as zenrav1e (apples-to-apples).
- **Threading:** stated per file; A/B always compared in the same mode.
- **I/O:** corpus decoded to pixels before the timed/encoded region.

| File | What it measures | Headline |
|------|------------------|----------|
| (CLAUDE.md, repo root) | QM / RdoTx / CDEF, 63-image corpus, speed 6 | QM only **−10.1%**; +RdoTx −10.3% (~3× time) |
| `trellis_rdoq_2026-06-18.md` | Multi-level trellis RDOQ (`enable_trellis`) | **−0.94%** mean BD-rate(Y) on 38 photos, +72% time, regression-free |
| `trellis_rdoq_photos38_s*.tsv` | Per-image trellis RD across lambda scales 0.5/0.75/1.0 | lambda 1.0 is the calibrated optimum |
| `trellis_rdoq_{photos,nonphoto_floor0,nonphoto_floor1}_2026-06-18.tsv` | Trellis RD, photo vs non-photo | photo wins; non-photo marginal/mixed |
| `issue6-bottomup-qm-2026-06-13.md` | Bottom-up vs top-down partition × QM | **negative** — bottom-up never beats top-down; workaround stays |
| `issue6-{photo,scifig}-clean-2026-06-13.tsv`, `issue6-probe-*.tsv` | Bottom-up proving sweep data | — |
| `stage2_rdo_estimate_2026-06-18.md` | Closed-form RDO coeff-rate estimate | RD-neutral but only ~5% faster — large-speedup premise falsified |

Each `.md` records its own corpus, exact zenrav1e commit, metric build, and the
harness used (the RD harnesses live in `zenavif`'s `encode_sweep` example and in
`/mnt/v/output/...` measurement dirs, pinned via `[patch.crates-io]` to the
local zenrav1e under test). Re-deriving a verdict means re-running that file's
stated harness against the listed commit; the committed `.tsv` is the recorded
output.

## Kernel microbenchmarks (`../benches/`)

Criterion microbenchmarks for hot kernels (motion compensation, transforms,
intra prediction, distortion, plane ops). These are reproducible from this repo:

```sh
git clone https://github.com/imazen/zenrav1e && cd zenrav1e
cargo bench --features bench          # runs the declared `bench` target
```

The harness only times the kernel under test (inputs are built before the
measured closure). Run **without** `-C target-cpu=native`; compare against a
clean checkout of the commit you started from. These measure CPU-kernel
throughput, not compression — use the RD results above for codec-quality
decisions.

## Adding a benchmark

New comparative benchmarks should follow the house fair-benchmark rules
(`~/work/claudehints/topics/benchmarking.md`): real named corpora (no Kodak /
gradient overfit), pinned competitor versions/commits, I/O excluded from the
timed region, threading mode stated, no `-C target-cpu=native`, and the raw
per-cell data committed alongside the summary. New timing benchmarks use
[zenbench](https://github.com/imazen/zenbench).
