# zenrav1e

AV1 encoder optimized for still and animated AVIF images. Fork of [rav1e](https://github.com/xiph/rav1e) by Imazen.

## What's different

zenrav1e adds features that matter for photographic AVIF encoding:

- **Quantization matrices** — frequency-dependent quantization weights, ~10% BD-rate improvement on a 67-image corpus (every image improved)
- **Filter intra prediction** — 5 recursive filter modes, auto-enabled at speed ≤ 6
- **Lossless mode** — mathematically lossless via `quantizer: 0`

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

Requires Rust 1.85+. The `asm` feature needs [NASM](https://nasm.us/) 2.14.02+ on x86_64.

## License

BSD-2-Clause, same as upstream rav1e. See [LICENSE](LICENSE) and [PATENTS](PATENTS).
