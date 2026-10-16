# AV1 still-encode paths on macOS ARM64, 2026-10-16

Source `1e2d13cd`, M4 Pro. These were two separate builds without `target-cpu=native`:

- `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-tiers-macos threading` (`~/tmp/zenrav1e-arm-threading-20261016T232747Z.log`)
- `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-tiers-macos threading,asm` (`~/tmp/zenrav1e-arm-threading,asm-20261016T232840Z.log`)

Each recipe ran `cargo bench -p zenrav1e --bench tier_isolation` with its named feature set and `--cpu-policy=unrestricted --format=llm`. The scheduler was unrestricted. Both paths encoded the same seeded 8-bit 4:2:0 still fixtures at speed 8 and q-index 100, one encoder thread. The benchmark decoded each output and matched the encoder reconstruction plane by plane before timing.

| Fixture | Rust fallback mean | Compiled NEON assembly mean | Output bytes |
| --- | ---: | ---: | ---: |
| 256 × 256 | 158.83 ms | 119.55 ms | 57141, identical |
| 512 × 512 | 618.48 ms | 466.70 ms | 227994, identical |

The separate-build comparison confirms the optional ARM assembly path is faster on these two fixtures. It does not measure other speeds, bit depths, sizes, or image content. `zenavif` keeps assembly opt-in through its existing `encode-asm` feature, so no default behavior change follows from this result.

`run-heavy` Rust: peak RSS 1.24 GiB, minimum available memory 5187 MiB, peak load 2.63. Assembly: peak RSS 1.23 GiB, minimum available memory 5244 MiB, peak load 2.72. Generated OBU files are under the matching `~/tmp/zenrav1e-arm-.../` directories.
