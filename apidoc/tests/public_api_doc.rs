//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! Only `zenrav1e` is snapshotted (`ivf` is `publish = false`), with default
//! features only (`no_extra_section` — the full feature set doesn't build
//! cleanly as public surface):
//! - `decode_test` / `decode_test_dav1d` (aom-sys / dav1d-sys) link system
//!   libaom/libdav1d, which CI runners and contributor machines don't
//!   reliably have;
//! - `capi` documents `src/capi.rs`, which the include-list excludes from the
//!   published package (`!/src/capi.rs`) — its surface would be a lie for
//!   crates.io consumers (it ships via cargo-c builds instead);
//! - `bench` exposes ~800 internal benchmark helpers that are not user API.
//!
//! The default feature set includes `asm`, so regeneration needs NASM on
//! PATH (same as any default build of this crate).
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(["zenrav1e"])
        .no_extra_section("zenrav1e")
        .run();
}
