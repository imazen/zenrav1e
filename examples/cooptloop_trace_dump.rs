//! Encode a PPM (P6; `convert img.png out.ppm`) or a built-in synthetic frame
//! single-threaded with the COOPT_LOOP decision trace armed, and dump the
//! trace TSV — the per-encode dataset generator for the Phase-1 λ–D–R fits.
//!
//! Usage:
//!   cargo run --release --features cooptloop_trace --example cooptloop_trace_dump -- \
//!     OUT.tsv [INPUT.ppm] [--speed N] [--quantizer N] [--ivf-out PATH]
//!
//! `--ivf-out` also writes the encode as a decodable IVF (aomdec/dav1d/
//! rav1d) so the trace can be joined to decoded-quality scores.
//!
//! Without INPUT.ppm a deterministic gradient+noise synthetic frame is used
//! (the tests' fixture). The encode itself is discarded — the trace is the
//! product. Run one encode per process (the trace buffer is global).

#[cfg(not(feature = "cooptloop_trace"))]
fn main() {
  eprintln!("rebuild with --features cooptloop_trace");
  std::process::exit(2);
}

/// Minimal binary-PPM (P6, maxval 255) reader: returns (rgb bytes, w, h).
#[cfg(feature = "cooptloop_trace")]
fn parse_ppm_p6(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
  // Header = "P6" + 3 whitespace-separated ints (comments start with '#').
  let mut pos = 0usize;
  let mut tokens: Vec<usize> = Vec::new();
  if &data[..2] != b"P6" {
    return None;
  }
  pos += 2;
  while tokens.len() < 3 {
    while pos < data.len() && data[pos].is_ascii_whitespace() {
      pos += 1;
    }
    if pos < data.len() && data[pos] == b'#' {
      while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
      }
      continue;
    }
    let start = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
      pos += 1;
    }
    tokens.push(std::str::from_utf8(&data[start..pos]).ok()?.parse().ok()?);
  }
  pos += 1; // the single whitespace after maxval
  let (w, h, maxval) = (tokens[0], tokens[1], tokens[2]);
  if maxval != 255 || data.len() < pos + w * h * 3 {
    return None;
  }
  Some((data[pos..pos + w * h * 3].to_vec(), w, h))
}

#[cfg(feature = "cooptloop_trace")]
fn main() {
  use zenrav1e::cooptloop_trace;
  use zenrav1e::prelude::*;

  let mut args = std::env::args().skip(1);
  let out = args.next().unwrap_or_else(|| {
    eprintln!("usage: cooptloop_trace_dump OUT.tsv [INPUT.png] [--speed N] [--quantizer N]");
    std::process::exit(2);
  });
  let mut input: Option<String> = None;
  let mut speed: u8 = 6;
  let mut quantizer: usize = 100;
  let mut ivf_out: Option<String> = None;
  while let Some(a) = args.next() {
    match a.as_str() {
      "--speed" => {
        speed = args.next().expect("--speed N").parse().expect("speed")
      }
      "--quantizer" => {
        quantizer =
          args.next().expect("--quantizer N").parse().expect("quantizer")
      }
      "--ivf-out" => {
        ivf_out = Some(args.next().expect("--ivf-out PATH"));
      }
      p => input = Some(p.to_string()),
    }
  }

  // Load 8-bit RGB from a binary PPM (P6, maxval 255), or synthesize.
  let (w, h, y, u, v) = match &input {
    Some(path) => {
      let data = std::fs::read(path).expect("read input");
      let (rgb, pw, ph) = parse_ppm_p6(&data).expect("parse P6 ppm");
      let (w, h) = (pw & !1, ph & !1);
      // BT.601 full-range RGB->YCbCr 4:2:0 (analysis-grade; the trace cares
      // about RD structure, not colorimetric exactness).
      let px = |x: usize, j: usize| {
        let o = (j * pw + x) * 3;
        (rgb[o] as f32, rgb[o + 1] as f32, rgb[o + 2] as f32)
      };
      let (cw, ch) = (w / 2, h / 2);
      let mut y = vec![0u8; w * h];
      let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
      for j in 0..h {
        for i in 0..w {
          let (r, g, b) = px(i, j);
          y[j * w + i] = (0.299 * r + 0.587 * g + 0.114 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
        }
      }
      for j in 0..ch {
        for i in 0..cw {
          let (r, g, b) = px(2 * i, 2 * j);
          u[j * cw + i] = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
          v[j * cw + i] = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
        }
      }
      (w, h, y, u, v)
    }
    None => {
      let (w, h) = (256usize, 256usize);
      let mut s: u32 = 0x1234_5678;
      let mut rng = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 24) as i32 - 128
      };
      let (cw, ch) = (w / 2, h / 2);
      let mut y = vec![0u8; w * h];
      for j in 0..h {
        for i in 0..w {
          let g = (i * 255 / w + j * 255 / h) / 2;
          y[j * w + i] = (g as i32 + rng() / 4).clamp(0, 255) as u8;
        }
      }
      let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
      for j in 0..ch {
        for i in 0..cw {
          u[j * cw + i] =
            (128 + (i * 64 / cw) as i32 + rng() / 8).clamp(0, 255) as u8;
          v[j * cw + i] =
            (128 + (j * 64 / ch) as i32 + rng() / 8).clamp(0, 255) as u8;
        }
      }
      (w, h, y, u, v)
    }
  };

  let mut ss = SpeedSettings::from_preset(speed);
  ss.segmentation = SegmentationLevel::Disabled;
  let enc = EncoderConfig {
    width: w,
    height: h,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    still_picture: true,
    low_latency: true,
    quantizer,
    tune: Tune::Ssimulacra2,
    speed_settings: ss,
    ..Default::default()
  };
  let cfg =
    zenrav1e::config::Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: zenrav1e::Context<u8> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  f.planes[0].copy_from_raw_u8(&y, w, 1);
  f.planes[1].copy_from_raw_u8(&u, w / 2, 1);
  f.planes[2].copy_from_raw_u8(&v, w / 2, 1);

  cooptloop_trace::clear();
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut bytes = 0usize;
  let mut frames: Vec<Vec<u8>> = Vec::new();
  while let Ok(pkt) = ctx.receive_packet() {
    bytes += pkt.data.len();
    frames.push(pkt.data);
  }

  // Optional IVF wrap (decodable by aomdec/dav1d/rav1d) so the trace's
  // encode can be scored — the D-vs-metric join the Phase-1 fits need.
  if let Some(path) = &ivf_out {
    use std::io::Write;
    let mut fout =
      std::io::BufWriter::new(std::fs::File::create(path).expect("ivf out"));
    // 32-byte IVF header: DKIF v0, hdr len 32, AV01, w, h, timebase 25/1,
    // frame count, unused.
    fout.write_all(b"DKIF").unwrap();
    fout.write_all(&0u16.to_le_bytes()).unwrap();
    fout.write_all(&32u16.to_le_bytes()).unwrap();
    fout.write_all(b"AV01").unwrap();
    fout.write_all(&(w as u16).to_le_bytes()).unwrap();
    fout.write_all(&(h as u16).to_le_bytes()).unwrap();
    fout.write_all(&25u32.to_le_bytes()).unwrap();
    fout.write_all(&1u32.to_le_bytes()).unwrap();
    fout.write_all(&(frames.len() as u32).to_le_bytes()).unwrap();
    fout.write_all(&0u32.to_le_bytes()).unwrap();
    for (i, fr) in frames.iter().enumerate() {
      fout.write_all(&(fr.len() as u32).to_le_bytes()).unwrap();
      fout.write_all(&(i as u64).to_le_bytes()).unwrap();
      fout.write_all(fr).unwrap();
    }
  }

  let n = cooptloop_trace::dump_tsv(&out).expect("dump trace");
  eprintln!(
    "encoded {}x{} s{} q{} -> {} B; trace {} rows ({} dropped) -> {}",
    w,
    h,
    speed,
    quantizer,
    bytes,
    n,
    cooptloop_trace::dropped(),
    out
  );
  if cooptloop_trace::dropped() > 0 {
    eprintln!("WARNING: trace incomplete (COOPTLOOP_TRACE_CAP hit)");
    std::process::exit(3);
  }
}
