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

# Format + regenerate the public-API surface snapshots (docs/public-api/)
fmt:
    cargo fmt --all
    cargo test -p zenrav1e --test public_api_doc

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test -p zenrav1e --test public_api_doc

# Verify the committed snapshots are current (what CI runs)
api-doc-check:
    ZEN_API_DOC=check cargo test -p zenrav1e --test public_api_doc

# Test feature permutations
feature-check:
    cargo test --workspace --no-default-features --features "threading,serialize"
    cargo check --no-default-features --features "threading,channel-api"
    cargo check --no-default-features --features "threading,stop"

# Full CI check
ci: lint test feature-check

# Address sanitizer (requires nightly + clang)
asan:
    RUSTFLAGS="-Zsanitizer=address -Clinker=clang" cargo +nightly test --no-default-features --features threading --target x86_64-unknown-linux-gnu -- --test-threads=1

# Miri (unit tests only — encoder tests too slow)
miri:
    cargo +nightly miri test --no-default-features --features threading -- "quantize::trellis::tests"
