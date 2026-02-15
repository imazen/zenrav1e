# zenrav1e

[![crates.io](https://img.shields.io/crates/v/zenrav1e.svg)](https://crates.io/crates/zenrav1e)
[![docs.rs](https://docs.rs/zenrav1e/badge.svg)](https://docs.rs/zenrav1e)
[![license](https://img.shields.io/crates/l/zenrav1e.svg)](LICENSE)

AV1 encoder optimized for still and animated AVIF images. Fork of [rav1e](https://github.com/xiph/rav1e) by Imazen.

## Fork of rav1e

Fully synced with upstream rav1e — no upstream commits are missing. All changes are additive (38 commits on top of upstream HEAD).

### Encoding features

- **Quantization matrices** — frequency-dependent quantization weights, ~10% BD-rate improvement
- **Filter intra prediction** — 5 recursive filter modes, auto-enabled at speed <= 6
- **Trellis quantization** — Viterbi DP with CDF-based rate estimation and quality-adaptive dampening
- **Variance adaptive quantization (VAQ)** — configurable strength parameter
- **Tune::StillImage mode** — tuning preset for photographic content
- **Lossless mode** — mathematically lossless encoding via `quantizer: 0`
- **Cooperative cancellation** — `enough::Stop` support behind the `stop` feature

### Modernization

- Rust 2024 edition (MSRV 1.88)
- `safe_unaligned_simd` for safe SIMD load/store in entropy coding

All upstream rav1e video encoding capabilities are preserved.

## Usage

zenrav1e is a library. If you want to encode AVIF images, use [ravif](https://lib.rs/crates/ravif) or [zenavif](https://github.com/imazen/zenavif), which wrap zenrav1e with a higher-level API.

For direct use:

```rust
use zenrav1e::prelude::*;

let mut enc = EncoderConfig::default();
enc.width = 640;
enc.height = 480;
enc.speed_settings = SpeedSettings::from_preset(6);
enc.still_picture = true;
enc.enable_qm = true;  // quantization matrices

let cfg = Config::new().with_encoder_config(enc);
let mut ctx: Context<u8> = cfg.new_context().unwrap();
// send frames, receive packets...
```

## Building

```bash
# Pure Rust (no asm) — primary development target
cargo check --no-default-features --features threading
cargo test --no-default-features --features threading

# With x86_64 asm (requires nasm)
cargo check --features threading
```

Requires Rust 1.88+. The `asm` feature needs [NASM](https://nasm.us/) 2.14.02+ on x86_64.

## License

BSD-2-Clause, same as upstream rav1e. See [LICENSE](LICENSE) and [PATENTS](PATENTS).
