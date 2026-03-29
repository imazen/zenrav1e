# zenrav1e [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenrav1e/ci.yml?branch=master&style=flat-square)](https://github.com/imazen/zenrav1e/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenrav1e?style=flat-square)](https://crates.io/crates/zenrav1e) [![lib.rs](https://img.shields.io/badge/lib.rs-zenrav1e-orange?style=flat-square)](https://lib.rs/crates/zenrav1e) [![docs.rs](https://img.shields.io/docsrs/zenrav1e?style=flat-square)](https://docs.rs/zenrav1e) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](https://github.com/imazen/zenrav1e#license)

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

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · **zenrav1e** · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

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

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
