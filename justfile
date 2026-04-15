# wls-alloc development tasks

# List available recipes
default:
    @just --list

# Run the full test suite
test:
    cargo test

# Run only the C golden-data compliance tests (1000 cases, 5 checks)
golden:
    cargo test --test regression -- --nocapture

# Run a single named C compliance test
# e.g.: just golden-one all_cases_incremental_vs_c_ref
golden-one name:
    cargo test --test regression -- --nocapture {{ name }}

# Run edge-case and unregularised tests (no C reference data)
unit:
    cargo test --test edge_cases --test unreg

# Run doc-tests only
doctest:
    cargo test --doc

# Clippy (matches CI: deny all warnings)
lint:
    cargo clippy --all-targets -- -D warnings

# Format check (matches CI)
fmt:
    cargo fmt --check

# Apply formatting
fmt-fix:
    cargo fmt

# Full CI gate: fmt + lint + test + doc
ci: fmt lint test
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Verify the library builds for embedded (no_std, thumbv7em)
no-std:
    cargo build --target thumbv7em-none-eabihf --lib

# Run benchmarks
bench:
    cargo bench
