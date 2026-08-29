# Strict-decoder conformance sweep for the rav1d-safe 66f58fa6 bump

rav1d-safe `2e0f7e8` made `Settings::default()` — and therefore every
`Decoder::new()` in this repo — `Strictness::Strict`: the AV1 §6.10.8
`segment_id` bound plus dav1d's `strict_std_compliance` checks now return
`Error::InvalidData` where the decoder used to conceal and continue.

rav1d-safe is this encoder's conformance oracle, so the question the bump
raises is not "does the decoder still work" but **"was this suite hiding an
encoder bug that a conformant decoder rejects?"** rav1d-safe#422 was found
exactly that way, through zenrav1e#35. This is the sweep that answers it.

Answer: **no**. Nothing this encoder emits was rejected, and nothing it emits
decodes to different pixels than it reconstructed.

## What ran

| Gate | Cells | Oracles | Result |
|---|---|---|---|
| `cargo test --workspace` | 6 round-trip suites (18 tests) + 197 unit + 3 doctest-shims + 6 doctests | rav1d-safe, Strict | green, counts identical to the pre-bump run |
| `gate_identity` full grid | 441 (3 images x s{2,6,8} x q{60,140,220} x {P,T}, plus every neutral and armed-path arm) | rav1d-safe, Strict | all decodable, 0 undecodable |
| `gate-sliver64` (release, aomdec leg armed) | 5 tests / 28 streams | rav1d-safe Strict + aomdec | 5/5 |
| `scripts/gate_recon.sh`, both legs | 54 (3 images x s{2,6,8} x q{60,140,220} x {cdef1-lrf1, cdef0-lrf1}) | rav1d-safe Strict + aomdec, byte-equal to recon | 54 cells, 0 failures |
| `scripts/sliver64_corpus_decode.sh` | 141 (47 images x q{80,130,205}, `deep`) | rav1d-safe Strict + aomdec + dav1d, byte-equal to recon | 141 ok, 0 failed |

The corpus sweep is the widest leg and the one that matters most: `deep` is the
widened-partition configuration that actually reaches `BLOCK_64X16` /
`BLOCK_16X64`, and **73 of the 141 cells placed at least one 64-dimension
sliver** (`px_64x16` or `px_16x64` non-zero in the TSV). That is the code path
behind zenrav1e#28/#32/#33/#35 — every historical desync in this repo lived
there, and it is where a newly-strict decoder had the best chance of finding
another one.

"Decodes" is not the bar anywhere above: gate-recon and the corpus gate compare
the decoder's raw output byte-for-byte against the encoder's own
reconstruction, so a decoder that accepts a desynced stream still fails them.

## Positive control — Strict is actually armed

"Nothing failed" is only a result if the new checks were live. At `66f58fa6`,
with the feature set this dev-dependency resolves to (`default` = `bitdepth_8`
+ `bitdepth_16`, no `unchecked`), rav1d-safe's committed zenrav1e#35 desync
vector `tests/strictness_vectors/segment_id_desync_zenrav1e35.obu` gives:

```
Decoder::new()            -> Err(InvalidData)
with_settings(Lenient)    -> Ok(Some(frame))
```

So the segment_id bound rejects a real stream on this host, in this
configuration, and the suite's green is a verdict rather than a no-op.

## Encoded bytes did not move

The `gate_identity` fingerprint set (length + fnv1a64 per cell) was captured
from an old-pin (`91bf0e30`) build and compared against a new-pin build:
**81 baseline and armed-path cells, 81 pinned-ok, 0 drift**, with all 360
neutral-arm equality contracts holding, repeated across a forced rebuild. CI's
`Gate A1` job confirms the same against the committed `linux-x86_64` pins.

The `Cargo.lock` delta is exactly the two `source =` lines for `rav1d-safe` and
`rav1d-disjoint-mut`; no transitive version moved.

Comparing `.rlib` bytes would *not* have shown this — rustc's output for this
crate is not reproducible across recompiles even at an identical pin
(`e4393774` vs `4715a889` with no input change). The bitstream fingerprints are
the instrument.

## `flush()` is inert here

rav1d-safe `59eb17b` makes `Decoder::flush()` drain owed frames before it
resets. It changes nothing in this repo, and that is a property of the
configuration rather than luck: the dev-dependency resolves rav1d-safe without
`unchecked`, and `get_num_threads` carries
`#[cfg(not(feature = "unchecked"))] let n_fc = 1`, so frame threading is
unreachable at any thread count and `decode()` returns each single-temporal-unit
packet synchronously. The eight `if fr.is_none() { fr = dec.flush() }` recovery
sites never take the branch; the two collect-everything sites
(`tests/segmentation_resignal_roundtrip.rs`, `tests/sliver_64_tx_roundtrip.rs`)
drain zero frames under both the old and the new semantics.

The paired runs prove it rather than merely arguing it, because the old
semantics were the mutation:

* **The recovery branch was never taken.** Old `flush()` reset first and
  therefore returned an empty `Vec`, so a taken branch would leave `fr` as
  `None` and the following `.expect("no decoded frame")` would panic. The suite
  was green at `91bf0e30`, so the branch was not reached — no experiment
  needed.
* **The drain returns nothing.** Both collect-everything sites assert an exact
  frame count (`raw_frames.len() == frames`, `decoded.len() ==
  enc.packets.len()`). Under the old semantics `flush()` contributed zero, so
  the `decode()` loop already supplied the full count; if the new drain
  produced even one extra frame the assertion would now fail high. It does not.

Counts are byte-identical across the two runs for all fourteen test binaries
(`3/197/0/0/3/0/3/6/1/5/1/2/0/6+6ign`).

Note for anyone reading the earlier brief for this bump: it described nine
`dec.flush()` sites, all in the recovery shape. There are **ten**, in three
shapes — eight recovery, one `raw_frames.extend(dec.flush()…)`, and one
`for frame in dec.flush()…`. The literal string `dec.flush()` appears nine
times only because `tests/sliver_64_tx_roundtrip.rs:252` wraps the call across
two lines. The two non-recovery sites are exactly the ones that could have
moved, so the distinction matters.

## Host and versions

Apple M-series (aarch64), macOS 26.5, rustc 1.98.0, `--release`, default
features (`threading`), no `-C target-cpu=native`. aomdec from libaom 3.14.1,
dav1d 1.5.4, rav1d-safe at `66f58fa6a64c689998721cc5cdb16a4698e26eec`.

**This is an aarch64 result.** Two fixes in `91bf0e30..66f58fa6` are x86_64-only
(`ee07356` 16bpc AVX2 horizontal 8-tap source over-read, `3426ebf` the x86_64
loopfilter H window of rav1d-safe#524) and cannot execute on this host. CI
covers them: this repo runs `cargo test --workspace` on Linux x64, Linux
aarch64, Linux i686 (via `cross`), macOS arm64, macOS x64, Windows x64 and
Windows arm64, plus `clippy --all-targets`, `cargo llvm-cov` and both gate
jobs — every one of which compiles and runs rav1d-safe, because it is an
unconditional dev-dependency.

## Reproducing

```sh
# corpus: the 47-image raw-RGB manifest from benchmarks/sliver64_rd_2026-08-28.md
cargo build --release --example sliver64_rd
SLIVER64_RD_DUMP=~/tmp/sliver64-corpus \
  ./target/release/examples/sliver64_rd manifest.tsv 2 80,130,205 deep \
  > strict_decoder_corpus_2026-08-29.tsv

# three-decoder byte-equality gate. IVF_RAW is any binary with zenavif
# ivf_raw's CLI (<in.ivf> <out.raw>) built against this rav1d-safe rev; it
# must NOT pass Strictness::Lenient, or the leg proves nothing.
IVF_RAW=<ivf_raw> AOMDEC=$(command -v aomdec) DAV1D=$(command -v dav1d) \
  bash scripts/sliver64_corpus_decode.sh ~/tmp/sliver64-corpus

# recon grid, both legs
IVF_RAW=<ivf_raw> AOMDEC=$(command -v aomdec) bash scripts/gate_recon.sh
```

`strict_decoder_corpus_2026-08-29.tsv` is the encode side of the corpus run:
141 rows, one per (image, q) cell, with `px_64x16` / `px_16x64` recording how
many sliver pixels each cell actually coded. The RD columns are a byproduct of
the harness — this run was a conformance sweep, not an RD measurement, and the
RD comparison for these transforms is `benchmarks/sliver64_rd_2026-08-28.md`.
