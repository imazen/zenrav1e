# Lossless inter skip and small sequence frames

Baseline: `1447c200fd083916c69ac72c1335079ba4624b2e`, Linux x86_64.
This is correctness validation, not an encoder performance or quality benchmark.

`tests/small_video_roundtrip.rs` exposed a source mismatch on an already valid
17×17 4:4:4 sequence: 8-bit, quantizer 0, preset 6, reordered presentation 2.
The bottom-right luma sample was 53 instead of 54; another component also
differed. The same class appeared on 1×17 monochrome after relaxing the API's
16-pixel sequence restriction. Testing only preset 10 without full reordering
did not find it.

The encoder evaluated inter skip using weighted rate/distortion cost. A small
error can cost less than an exact residual, and weighted distortion can also
round small errors to zero. Neither criterion establishes losslessness.
Lossless skip now requires exact equality between source and prediction on all
visible owned components. It includes shared chroma, clips partial frame edges,
and ignores allocation padding. A rejected trial rolls back its entropy context
and continues to the residual-coded candidates. Lossy skip selection is unchanged.

After this fix, sequence dimensions 1..65535 use the same validation domain as
still frames. Backend allocation padding remains internal: signaled and decoded
dimensions must equal the submitted dimensions. Zero dimensions remain invalid.

## Coverage

The new regression checks five presentations per configuration:

- Dimensions 1×1, 1×17, 17×1, 3×5, 7×9, 17×13 and 16×16.
- Mono, 4:2:0, 4:2:2 and 4:4:4; 8/10/12-bit.
- Presets 2, 6 and 10; quantizers 0 and 80; default Psychovisual tune.
- Low-latency and full reordered groups (eight-frame keyframe interval).
- Additional lossless reordered 17×17, 17×33 and 33×17 cases, all four layouts
  and three presets at 8-bit.

Total: 1,044 configurations and 5,220 presentation comparisons per run. Every
decoded sample must equal the encoder reconstruction; at quantizer zero every
reconstruction must also equal the original source. The decoder is rav1d-safe
with its strict default. Runs with and without the `scenechange` feature pass.
`SMALL_VIDEO_ARTIFACTS=<directory>` preserves raw OBU sequences, original samples
and encoder reconstructions per configuration; the test never silently skips.

## Gates

Before and after the coding change:

```sh
just gate-identity
just gate-recon
```

Both identity runs: 81/81 pinned baseline cells, zero drift, 360/360 arm cells,
all outputs decodable. Both reconstruction runs: 54 cells, zero failures, native
rav1d-safe decoder leg. No identity pins were changed.

Additional checks:

```sh
cargo test --release --test small_video_roundtrip
cargo test --release --features scenechange --test small_video_roundtrip
cargo test --release --lib
cargo test --workspace
cargo clippy --all-targets -- -D warnings
```

The release library suite passed 198 tests. Clippy passed with warnings denied.
The full debug workspace suite, integration tests, and doctests also passed;
six pre-existing doctests remain ignored.
