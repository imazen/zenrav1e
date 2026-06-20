// Copyright (c) 2024-2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Fuzz crash regression suite.
//!
//! Replays every minimized seed in `fuzz/regression/` through the same
//! `crate::fuzzing` entry point its originating fuzz target uses — as an
//! ordinary `cargo test`, no nightly / cargo-fuzz toolchain required. Each
//! seed is a previously-found crash that has been fixed; this ensures none of
//! them re-introduces a panic.
//!
//! Build with `--features _fuzz_replay` so `pub mod fuzzing` (and its
//! `arbitrary`-based byte shims) is compiled in; an integration test links the
//! library without `cfg(test)`, so a plain `cfg(test)` gate would not reach it.
//! When that feature is off the whole file is inert.
//!
//! Routing is by filename suffix (mirroring how the seeds are named after the
//! target that found them):
//!
//! - `*-encode.bin`            -> `encode`            (`fuzz_encode`)
//! - `*-construct-context.bin` -> `construct_context` (`fuzz_construct_context`)
//! - `*-encode_decode.bin`     -> `encode_decode`     (`fuzz_encode_decode::<u8>`)
//! - `*-encode_decode_hbd.bin` -> `encode_decode_hbd` (`fuzz_encode_decode::<u16>`)
//!
//! The `encode_decode*` replays need the heavy `crate::test_encode_decode`
//! roundtrip harness, which only compiles under `test` / `fuzzing` (an
//! integration test is neither). So those seeds are skipped here — they are
//! still exercised by the nightly `encode_decode*` fuzz targets and by the
//! decode-roundtrip tests in `crate::test_encode_decode`. The encode-side
//! crashes that this harness exists for are covered by the `*-encode.bin` and
//! `*-construct-context.bin` seeds, which need no decoder.
//!
//! To add a new seed: drop the (preferably minimized) crash file into
//! `fuzz/regression/` named `<short-description>-<target>.bin`. No other action
//! required.

#![cfg(feature = "_fuzz_replay")]

use std::fs;
use std::path::{Path, PathBuf};

fn regression_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// Replay one seed through the entry point named by its filename suffix.
///
/// Any entry point may legitimately drop the input or return early, but it must
/// not panic. A panic propagates and fails the test with the seed name.
fn replay_seed(path: &Path, name: &str) {
  use zenrav1e::fuzzing::{
    fuzz_construct_context_bytes, fuzz_encode_bytes,
  };

  let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {name}: {e}"));

  if name.ends_with("-construct-context.bin") {
    fuzz_construct_context_bytes(&bytes);
    return;
  }

  // `encode_decode*` seeds need the decode-roundtrip harness, which is not
  // reachable from an integration test (see module docs). Skip them here.
  if name.ends_with("-encode_decode_hbd.bin")
    || name.ends_with("-encode_decode.bin")
  {
    return;
  }

  // Default / `*-encode.bin`: the bare encode path.
  fuzz_encode_bytes(&bytes);
}

#[test]
fn fuzz_regression_seeds_do_not_panic() {
  let dir = regression_dir();
  let mut entries: Vec<_> = fs::read_dir(&dir)
    .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
    .filter_map(|e| e.ok())
    .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
    .map(|e| e.path())
    .filter(|p| p.extension().map(|x| x == "bin").unwrap_or(false))
    .collect();
  entries.sort();

  assert!(
    !entries.is_empty(),
    "fuzz/regression/ has no .bin seeds — at least the committed crash seeds \
     should be present"
  );

  for path in entries {
    let name =
      path.file_name().and_then(|n| n.to_str()).unwrap_or("<unnamed>");
    replay_seed(&path, name);
    eprintln!("ok: {name}");
  }
}
