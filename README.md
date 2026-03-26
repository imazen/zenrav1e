# zenrav1e

[![crates.io](https://img.shields.io/crates/v/zenrav1e.svg)](https://crates.io/crates/zenrav1e)
[![docs.rs](https://docs.rs/zenrav1e/badge.svg)](https://docs.rs/zenrav1e)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg?style=for-the-badge)](LICENSE-AGPL3)

AV1 encoder optimized for still and animated AVIF images. Fork of [rav1e](https://github.com/xiph/rav1e) by Imazen.

## Fork of rav1e

All changes are additive on top of upstream rav1e.

### Encoding features

- **Quantization matrices** — frequency-dependent quantization weights, ~10% BD-rate improvement
- **Filter intra prediction** — 5 recursive filter modes, auto-enabled at speed <= 6
- **Trellis quantization** (experimental, disabled by default — marginal gains at +34% encode time)
- **Variance adaptive quantization (VAQ)** (experimental, disabled by default)
- **Tune::StillImage mode** — tuning preset for photographic content
- **Lossless mode** — mathematically lossless encoding via `quantizer: 0`
- **Cooperative cancellation** — `enough::Stop` support behind the `stop` feature

### Modernization

- Rust 2024 edition (MSRV 1.89)
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

Requires Rust 1.89+. The `asm` feature needs [NASM](https://nasm.us/) 2.14.02+ on x86_64.

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream code from [xiph/rav1e](https://github.com/xiph/rav1e) is licensed under BSD-2-Clause.
Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

### Upstream Contribution

We are willing to release our improvements under the original BSD-2-Clause
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.
