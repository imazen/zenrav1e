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

# Full CI check
ci: lint test
