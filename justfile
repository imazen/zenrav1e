# rav1e (Imazen Fork) build recipes

# Default: check pure-Rust build
default: check

# Check pure-Rust build (no asm)
check:
    cargo check --no-default-features --features threading

# Check with asm
check-asm:
    cargo check --features threading

# Run tests (pure Rust)
test:
    cargo test --no-default-features --features threading

# Format and lint
lint:
    cargo fmt
    cargo clippy --no-default-features --features threading

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Test feature permutations
feature-check:
    cargo test --workspace --no-default-features --features "threading,serialize"
    cargo check --no-default-features --features "threading,channel-api"
    cargo check --no-default-features --features "threading,stop"

# Full CI check
ci: lint test feature-check

# --- Executable gates (zenavif docs/ENGINEERING_BASELINE.md section A) ---

# Gate A1: byte-exactness of the off-state. Pinned baseline fingerprints
# (tests/gate_identity_pins.tsv, linux-x86_64) + documented-neutral knob
# arms + armed-path liveness pins. CI runs the --ci subset.
gate-identity:
    cargo run --release --example gate_identity

# Re-pin the identity baselines after an INTENTIONAL behavioral change.
# Commit the TSV diff in the same commit as the change that moved the bytes.
gate-identity-pin:
    cargo run --release --example gate_identity -- --pin

# Gate A5: encoder recon byte-agrees with conforming decoders (the #32/#33
# desync class). Local-only. Decoder legs are env-selected; the rav1d-safe
# leg is this repo's own `examples/ivf_raw` (built here, decoding at
# Decoder::new()'s Strict default), and aomdec comes from PATH when present.
# At least one leg must resolve. IVF_RAW= overrides, e.g. to point at
# zenavif's equivalent example.
gate-recon:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --release --example ivf_raw
    IVF_RAW="${IVF_RAW:-{{justfile_directory()}}/target/release/examples/ivf_raw}" \
    AOMDEC="${AOMDEC:-$(command -v aomdec || true)}" \
    bash scripts/gate_recon.sh

# Gate A5, sliver subset: the 64-dim sliver + 4:1 inter partition
# roundtrips (`tests/sliver_64_tx_roundtrip.rs`) against rav1d-safe
# in-process AND aomdec (`aom-tools`; CI installs it). Fails, never skips,
# when aomdec is missing.
gate-sliver64:
    SLIVER64_AOMDEC="${SLIVER64_AOMDEC:-$(command -v aomdec)}" \
    cargo test --release --test sliver_64_tx_roundtrip

# Gate A5, corpus half of zenrav1e#28: encode a real-image corpus in the
# only configuration that reaches BLOCK_64X16/16X64 (top-down, 4..64
# partition range, non-square threshold 64) and require aomdec AND dav1d AND
# rav1d-safe (this repo's `examples/ivf_raw`, the same leg gate-recon uses)
# to decode every stream byte-identically to the encoder's own
# reconstruction.
# Needs a manifest of raw 8-bit RGB stills; see benchmarks/sliver64_rd_*.md
# for the corpus recipe, and benchmarks/strict_decoder_corpus_2026-08-29.md
# for the last full run. Local-only (external decoders + corpus).
gate-sliver64-corpus manifest speed="2" qs="80,130,205":
    #!/usr/bin/env bash
    set -euo pipefail
    dump="${SLIVER64_RD_DUMP:-$HOME/tmp/sliver64-corpus}"
    rm -rf "$dump"
    cargo build --release --example sliver64_rd --example ivf_raw
    SLIVER64_RD_DUMP="$dump" ./target/release/examples/sliver64_rd \
      "{{manifest}}" "{{speed}}" "{{qs}}" deep > "$dump.tsv"
    IVF_RAW="${IVF_RAW:-{{justfile_directory()}}/target/release/examples/ivf_raw}" \
      bash scripts/sliver64_corpus_decode.sh "$dump"

# The CI-safe gate set (gate-recon and perf gates stay explicit local runs).
gates: gate-identity gate-sliver64

# Address sanitizer (requires nightly + clang)
asan:
    RUSTFLAGS="-Zsanitizer=address -Clinker=clang" cargo +nightly test --no-default-features --features threading --target x86_64-unknown-linux-gnu -- --test-threads=1

# Miri (unit tests only — encoder tests too slow)
miri:
    cargo +nightly miri test --no-default-features --features threading -- "quantize::trellis::tests"
