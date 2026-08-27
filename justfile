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
# desync class). Local-only. Decoder legs are env-selected; the default
# wires rav1d-safe via the zenavif sibling's ivf_raw example (build it
# first: `cargo build --release --example ivf_raw` in ../zenavif), and
# aomdec from PATH when present. At least one leg must resolve.
gate-recon:
    IVF_RAW="${IVF_RAW:-{{justfile_directory()}}/../zenavif/target/release/examples/ivf_raw}" \
    AOMDEC="${AOMDEC:-$(command -v aomdec || true)}" \
    bash scripts/gate_recon.sh

# Gate A5, sliver subset: the 64-dim sliver + 4:1 inter partition
# roundtrips (`tests/sliver_64_tx_roundtrip.rs`) against rav1d-safe
# in-process AND aomdec (`aom-tools`; CI installs it). Fails, never skips,
# when aomdec is missing.
gate-sliver64:
    SLIVER64_AOMDEC="${SLIVER64_AOMDEC:-$(command -v aomdec)}" \
    cargo test --release --test sliver_64_tx_roundtrip

# The CI-safe gate set (gate-recon and perf gates stay explicit local runs).
gates: gate-identity gate-sliver64

# Address sanitizer (requires nightly + clang)
asan:
    RUSTFLAGS="-Zsanitizer=address -Clinker=clang" cargo +nightly test --no-default-features --features threading --target x86_64-unknown-linux-gnu -- --test-threads=1

# Miri (unit tests only — encoder tests too slow)
miri:
    cargo +nightly miri test --no-default-features --features threading -- "quantize::trellis::tests"
