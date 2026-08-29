// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! `ivf_raw`: decode an IVF-contained (or bare-OBU) AV1 stream with
//! rav1d-safe and write the raw planar pixels of its last frame.
//!
//! Output is byte-compatible with `aomdec --rawvideo -o out.raw in.ivf`, so
//! an `md5sum` of the two is a decoder byte-agreement check. That is what
//! `scripts/gate_recon.sh` and `scripts/sliver64_corpus_decode.sh` use it
//! for: their rav1d-safe legs compare this output against the encoder's own
//! reconstruction, and a decoder that merely *accepts* a stream does not
//! clear them.
//!
//! Existing to make those legs self-contained. Both scripts take the binary
//! through `IVF_RAW`, and before this example the only implementation was
//! zenavif's `examples/ivf_raw`, so running the rav1d-safe leg meant
//! building inside a sibling checkout. rav1d-safe is already a dev-dependency
//! here for the round-trip tests; this spends nothing extra to remove that.
//!
//! **Settings are deliberately `Decoder::new()`'s defaults**, which since the
//! rav1d-safe `66f58fa6` pin means `Strictness::Strict`. Do not pass
//! `Strictness::Lenient` here: the whole value of the leg is that the oracle
//! rejects what libaom rejects, and a lenient decode would conceal exactly
//! the desyncs these gates exist to catch (see the repo CLAUDE.md, "Decoder
//! oracle: rav1d-safe is Strict by default").
//!
//! Usage: `ivf_raw <in.ivf|in.obu> <out.raw>`
//!
//! Layout: Y rows at the visible width, then U, then V, at
//! `ceil(w/2) x ceil(h/2)` for I420, `ceil(w/2) x h` for I422, `w x h` for
//! I444, and absent for I400. 8-bit is one byte per sample; 10/12-bit is two
//! bytes per sample little-endian, matching `aomdec --rawvideo` on LE hosts.

use std::io::Write as _;
use std::process::ExitCode;

use rav1d_safe::src::managed::{Decoder, Frame, PixelLayout, Planes};

/// Feeds the whole input and returns the last frame decoded.
///
/// An IVF file is demuxed frame by frame (32-byte file header, then a
/// 12-byte per-frame header carrying a `u32le` size and a `u64le` pts);
/// anything else is handed over as one bare OBU stream. `flush()` runs
/// afterwards so any frame the decoder still owes is collected — it is a
/// drain-then-reset since rav1d-safe `59eb17b`.
fn decode_last(data: &[u8]) -> Result<Frame, String> {
  let mut dec = Decoder::new().map_err(|e| format!("decoder init: {e:?}"))?;
  let mut frames: Vec<Frame> = Vec::new();

  let mut feed =
    |payload: &[u8], out: &mut Vec<Frame>| -> Result<(), String> {
      match dec.decode(payload) {
        Ok(Some(f)) => {
          out.push(f);
          Ok(())
        }
        Ok(None) => Ok(()),
        Err(e) => Err(format!("decode: {e:?}")),
      }
    };

  if data.len() >= 32 && &data[0..4] == b"DKIF" {
    let mut off = 32usize;
    while off + 12 <= data.len() {
      let sz = u32::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
      ]) as usize;
      off += 12;
      if off + sz > data.len() {
        return Err("truncated IVF frame".into());
      }
      feed(&data[off..off + sz], &mut frames)?;
      off += sz;
    }
  } else {
    feed(data, &mut frames)?;
  }

  frames.append(&mut dec.flush().map_err(|e| format!("flush: {e:?}"))?);
  frames.pop().ok_or_else(|| "no frames decoded".to_string())
}

/// Chroma plane dimensions for `layout`, or `(0, 0)` when there are none.
fn chroma_dims(layout: PixelLayout, w: usize, h: usize) -> (usize, usize) {
  match layout {
    PixelLayout::I400 => (0, 0),
    PixelLayout::I420 => (w.div_ceil(2), h.div_ceil(2)),
    PixelLayout::I422 => (w.div_ceil(2), h),
    PixelLayout::I444 => (w, h),
  }
}

fn serialize(frame: &Frame) -> Result<Vec<u8>, String> {
  let (w, h) = (frame.width() as usize, frame.height() as usize);
  let layout = frame.pixel_layout();
  let (cw, ch) = chroma_dims(layout, w, h);
  let mut out = Vec::with_capacity((w * h + 2 * cw * ch) * 2);

  match frame.planes() {
    Planes::Depth8(planes) => {
      for row in planes.y().rows().take(h) {
        out.extend_from_slice(&row[..w]);
      }
      for view in [planes.u(), planes.v()] {
        if cw == 0 {
          break;
        }
        let p =
          view.ok_or_else(|| format!("missing chroma for {layout:?}"))?;
        for row in p.rows().take(ch) {
          out.extend_from_slice(&row[..cw]);
        }
      }
    }
    Planes::Depth16(planes) => {
      for row in planes.y().rows().take(h) {
        for &v in &row[..w] {
          out.extend_from_slice(&v.to_le_bytes());
        }
      }
      for view in [planes.u(), planes.v()] {
        if cw == 0 {
          break;
        }
        let p =
          view.ok_or_else(|| format!("missing chroma for {layout:?}"))?;
        for row in p.rows().take(ch) {
          for &v in &row[..cw] {
            out.extend_from_slice(&v.to_le_bytes());
          }
        }
      }
    }
  }
  Ok(out)
}

fn run(input: &str, output: &str) -> Result<String, String> {
  let data = std::fs::read(input).map_err(|e| format!("read {input}: {e}"))?;
  let frame = decode_last(&data).map_err(|e| format!("{input}: {e}"))?;
  let bytes = serialize(&frame)?;
  let mut f = std::fs::File::create(output)
    .map_err(|e| format!("create {output}: {e}"))?;
  f.write_all(&bytes).map_err(|e| format!("write {output}: {e}"))?;
  Ok(format!(
    "{}x{} {:?} {} bytes",
    frame.width(),
    frame.height(),
    frame.pixel_layout(),
    bytes.len()
  ))
}

fn main() -> ExitCode {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let [input, output] = args.as_slice() else {
    eprintln!("usage: ivf_raw <in.ivf|in.obu> <out.raw>");
    return ExitCode::FAILURE;
  };
  match run(input, output) {
    Ok(summary) => {
      println!("{summary}");
      ExitCode::SUCCESS
    }
    Err(msg) => {
      eprintln!("FAIL {msg}");
      ExitCode::FAILURE
    }
  }
}
