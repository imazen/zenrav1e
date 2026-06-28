<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenrav1e

zenrav1e is an AV1 encoder tuned for still and animated AVIF images — an Imazen
fork of [rav1e](https://github.com/xiph/rav1e) that adds frequency-dependent
quantization matrices (~10% BD-rate), recursive filter-intra prediction, opt-in
trellis RDOQ, and a mathematically-lossless mode, all additive on top of the
battle-tested upstream encoder. The library default is **pure Rust and
toolchain-free** (no NASM/C needed); the x86_64 SIMD assembly is one feature
flag away. Rust 2024 edition, dual-licensed AGPL-3.0 / commercial.

## Quick start

zenrav1e is a library that emits a raw AV1 bitstream. If you just want to write
AVIF image files, reach for [ravif](https://lib.rs/crates/ravif) or
[zenavif](https://github.com/imazen/zenavif) — they wrap zenrav1e with a
higher-level API that does the RGB→YCbCr conversion and AVIF/HEIF muxing for
you. Use the raw API below only when you need direct control over the encoder.

```toml
[dependencies]
# Pure-Rust default (no NASM/C toolchain). The default feature is `threading`.
zenrav1e = "0.2.0"

# For the x86_64 SIMD assembly kernels (needs NASM 2.14+ at build time):
# zenrav1e = { version = "0.2.0", features = ["asm"] }
```

> **Input is planar YCbCr, not RGB.** rav1e (like AV1 itself) encodes
> Y′CbCr planes — there is no RGB entry point. If you fill the planes with
> interleaved RGB bytes the encode *succeeds* but the colors come out wrong,
> because the encoder reads plane 0 as luma and planes 1/2 as chroma. Convert
> to YCbCr (e.g. BT.601/709) and lay the samples out one plane at a time. The
> defaults below are 8-bit (`Context<u8>`) 4:2:0 (`ChromaSampling::Cs420`), so
> the U and V planes are half-resolution in each dimension. For 10/12-bit use
> `Context<u16>` and pass a `source_bytewidth` of `2` to `copy_from_raw_u8`.

```rust
use zenrav1e::prelude::*;

// `y`, `u`, `v`: tightly packed planar YCbCr 4:2:0, 8-bit (U/V half-size).
fn encode_still(
    y: &[u8], u: &[u8], v: &[u8],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut enc = EncoderConfig::default();
    enc.width = 640;
    enc.height = 480;
    enc.still_picture = true;                      // single-frame AVIF still
    enc.chroma_sampling = ChromaSampling::Cs420;   // YCbCr 4:2:0 (the default)
    enc.bit_depth = 8;                             // 8-bit (the default)
    enc.quantizer = 80;                            // see the quantizer note below
    enc.speed_settings = SpeedSettings::from_preset(6);
    enc.enable_qm = true;                          // quantization matrices (~10% BD-rate)

    let cfg = Config::new().with_encoder_config(enc);
    let mut ctx: Context<u8> = cfg.new_context()?;

    // Allocate a frame sized to the config and copy each YCbCr plane in.
    // `copy_from_raw_u8(src, src_stride_in_bytes, src_bytewidth)`:
    //   - src_bytewidth = 1 for `Context<u8>`, 2 for `Context<u16>` (10/12-bit)
    //   - src_stride is the row stride of *your* buffer; the call handles the
    //     frame's internal (padded) stride. Here each plane is tightly packed,
    //     so the stride is just that plane's width.
    let mut frame = ctx.new_frame();
    let planes = [y, u, v];
    for (plane, src) in frame.planes.iter_mut().zip(planes) {
        // Chroma planes are subsampled by `xdec`/`ydec` for 4:2:0, so derive
        // each plane's row width from its own decimation factor.
        let plane_width = (enc.width + (1 << plane.cfg.xdec) - 1) >> plane.cfg.xdec;
        plane.copy_from_raw_u8(src, plane_width, 1);
    }

    ctx.send_frame(frame)?;
    ctx.flush(); // signal end-of-stream so the still gets emitted

    // Drain packets. For a still picture this yields exactly one packet.
    let mut bitstream = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(packet) => bitstream.extend_from_slice(&packet.data),
            Err(EncoderStatus::Encoded) => {}          // frame consumed, no packet yet
            Err(EncoderStatus::LimitReached) => break, // all frames emitted
            Err(EncoderStatus::NeedMoreData) => break, // nothing left after flush
            Err(e) => return Err(e.into()),
        }
    }

    // `bitstream` is a raw AV1 bitstream (OBUs / temporal units) — NOT a `.avif`
    // file. To get a playable image, mux it into an AVIF/HEIF container with
    // `zenavif`, `ravif`, or `zenavif-serialize`. (Those crates also do the
    // RGB→YCbCr conversion, so prefer them unless you need this level of control.)
    Ok(bitstream)
}
```

### Quantizer scale

`EncoderConfig::quantizer` is the AV1 base **q-index**, a `usize` in `0..=255`
(default `100`). **Lower means higher quality and larger output; higher means
smaller and lower quality** — the opposite direction from a "quality %" dial.
`quantizer = 0` is the special mathematically-lossless mode (see the lossless
feature below); `255` is the most aggressive. For perceptual quality targeting,
the `ravif`/`zenavif` layers map a friendlier quality scale onto this q-index.

### Cooperative cancellation

The `stop` feature (`features = ["stop"]`) lets a watchdog or request-deadline
thread abort an in-progress encode. Pass any [`enough::Stop`][enough] token to
`Context::set_stop`; it is checked once per superblock. The simplest
constructible token is `almost_enough::Stopper` (`cargo add almost-enough`) —
`#[derive(Clone)]`, and all clones share one cancellation flag:

```rust
use std::sync::Arc;
use zenrav1e::prelude::*;

// Continuing from the example above, with the `stop` feature enabled:
let mut ctx: Context<u8> = cfg.new_context().unwrap();

// Hand the encoder a clone; keep `stopper` to trigger cancellation elsewhere.
let stopper = almost_enough::Stopper::new();
ctx.set_stop(Arc::new(stopper.clone()));

// From a deadline/watchdog thread (or on client disconnect):
stopper.cancel();
// `send_frame` / `receive_packet` then return `Err(EncoderStatus::Cancelled)`.

// To clear cancellation again, restore the no-op token:
ctx.set_stop(Arc::new(zenrav1e::Unstoppable));
```

`zenrav1e` re-exports `Stop`, `StopReason`, and `Unstoppable` from [`enough`][enough]
so you can name them without a direct dependency; the concrete `Stopper` token
lives in the companion `almost-enough` crate.

## Fork of rav1e

All changes are additive on top of upstream rav1e — every upstream video
encoding capability is preserved.

### Still-image encoding features

- **Quantization matrices** (`enable_qm`) — frequency-dependent quantization
  weights; ~10% BD-rate improvement on photographic stills. Off by default,
  recommended on.
- **Filter-intra prediction** — 5 recursive filter modes, auto-enabled at
  speed ≤ 6.
- **`Tune::StillImage`** — a tuning preset for photographic content (perceptual
  distortion with activity masking).
- **Lossless mode** — mathematically lossless encoding via `quantizer = 0`.
- **Trellis RDOQ** (`enable_trellis`, opt-in) — multi-level Viterbi coefficient
  optimization; −0.94% mean BD-rate(Y) on a 38-image photo corpus
  (regression-free across photo classes) at ~+72% encode time. Off by default
  because it mainly helps photographic content.
- **Variance-adaptive quantization** (`enable_vaq`, `vaq_strength`) and
  **segment boost** (`seg_boost`) — experimental knobs, off by default (see the
  benchmarks for why they didn't make the default config).
- **Cooperative cancellation** — `enough::Stop` support behind the `stop` feature.

### Modernization

- Rust 2024 edition (MSRV 1.89).
- **Pure-Rust, toolchain-free default.** The default feature set is `threading`
  only; `asm` (NASM SIMD), `scenechange`, and `signal_support` moved to the
  `binaries` feature, so library consumers need no C/NASM toolchain by default.
- `safe_unaligned_simd` for safe SIMD load/store in entropy coding (under `asm`).

## Building

```bash
# Pure Rust (no asm) — the default; primary development target
cargo check
cargo test

# With x86_64 asm SIMD kernels (requires NASM 2.14.02+)
cargo check --features asm
```

Requires Rust 1.89+. The `asm` feature needs [NASM](https://nasm.us/) 2.14.02+
on x86_64.


## License

Dual-licensed: [AGPL-3.0](https://github.com/imazen/zenrav1e/blob/master/LICENSE-AGPL3)
or [commercial](https://github.com/imazen/zenrav1e/blob/master/LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](https://github.com/imazen/zenrav1e/blob/master/LICENSE-COMMERCIAL) for details.

Upstream code from [xiph/rav1e](https://github.com/xiph/rav1e) is licensed under
BSD-2-Clause, with the Alliance for Open Media Patent License 1.0 (see
[PATENTS](https://github.com/imazen/zenrav1e/blob/master/PATENTS)). Our additions
and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

### Upstream contribution

We are willing to release our improvements under the original BSD-2-Clause
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · **zenrav1e** · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
