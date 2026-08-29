// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Rate/distortion harness for the 64-dimension sliver transforms
//! TX_64X16/TX_16X64 (zenrav1e#28, "measure the RD delta of the real 64-dim
//! sliver transforms vs the capped ones").
//!
//! It encodes a corpus of raw RGB stills over a quantizer sweep and prints
//! one TSV row per (image, speed, quantizer) with the coded byte count, the
//! PSNR of the encoder reconstruction against the source, and the pixel
//! counts of the two sliver block sizes. Run the same binary from two source
//! trees — one with the 3fa735dc `rdo_tx_size_type` cap and one without —
//! and BD-rate the two TSVs; the sliver pixel columns keep the comparison
//! honest by showing whether HORZ_4/VERT_4 at BLOCK_64X64 parents fired at
//! all (they only do at speed presets 0 and 1, where
//! `non_square_partition_max_threshold` is BLOCK_64X64).
//!
//! Set `SLIVER64_RD_DUMP=<dir>` to also write, per cell, the IVF bitstream
//! and the encoder's reconstruction as y4m. `scripts/sliver64_corpus_decode.sh`
//! feeds those to aomdec and dav1d and requires the decoded frame to be
//! byte-equal to the reconstruction, which is the corpus-sweep half of
//! zenrav1e#28's "done means" (the synthetic half is
//! `tests/sliver_64_tx_roundtrip.rs`).
//!
//! Usage:
//!   sliver64_rd <manifest.tsv> <speed> <q,q,...> <stock|deep>
//!
//! `stock` is `SpeedSettings::from_preset(speed)` untouched. `deep` is that
//! preset with the three knobs a 64x64-parent 4-way partition needs:
//! top-down partitioning (`encode_bottomup = false`; only
//! `encode_partition_topdown` offers HORZ_4/VERT_4 at all), a 4..64 partition
//! range, and `non_square_partition_max_threshold = BLOCK_64X64`. No stock
//! preset combines them — 0/1 are bottom-up and 2+ cap the non-square
//! threshold at BLOCK_8X8 — so `stock` rows are the control that must show
//! zero sliver pixels.
//!
//! Manifest rows are `name<TAB>path.rgb<TAB>width<TAB>height`, where the file
//! is packed 8-bit RGB (`magick in.png -depth 8 rgb:out.rgb`). Lines starting
//! with `#` are ignored.

use std::fs;
use std::time::Instant;

use zenrav1e::config::SpeedSettings;
use zenrav1e::prelude::*;

/// BT.709 limited-range RGB -> YUV 4:2:0, 8-bit. Chroma is a box average of
/// the full-resolution chroma planes (odd dimensions replicate the last
/// column/row), which is what the AV1 encoder is fed everywhere else in this
/// repo's test harnesses.
fn rgb_to_i420(rgb: &[u8], w: usize, h: usize) -> [Vec<u8>; 3] {
  assert_eq!(rgb.len(), w * h * 3, "raw RGB size mismatch");
  let mut y = vec![0u8; w * h];
  let mut uf = vec![0f32; w * h];
  let mut vf = vec![0f32; w * h];
  for i in 0..w * h {
    let r = f32::from(rgb[i * 3]);
    let g = f32::from(rgb[i * 3 + 1]);
    let b = f32::from(rgb[i * 3 + 2]);
    let yl = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    y[i] = (16.0 + 219.0 / 255.0 * yl).round().clamp(0.0, 255.0) as u8;
    uf[i] = 128.0 + 224.0 / 255.0 * (b - yl) / 1.8556;
    vf[i] = 128.0 + 224.0 / 255.0 * (r - yl) / 1.5748;
  }
  let cw = w.div_ceil(2);
  let ch = h.div_ceil(2);
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      let x0 = 2 * i;
      let y0 = 2 * j;
      let x1 = (x0 + 1).min(w - 1);
      let y1 = (y0 + 1).min(h - 1);
      let idx = [y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1];
      let su: f32 = idx.iter().map(|&k| uf[k]).sum();
      let sv: f32 = idx.iter().map(|&k| vf[k]).sum();
      u[j * cw + i] = (su / 4.0).round().clamp(0.0, 255.0) as u8;
      v[j * cw + i] = (sv / 4.0).round().clamp(0.0, 255.0) as u8;
    }
  }
  [y, u, v]
}

/// Mean squared error between a reconstructed plane and the source plane.
fn plane_mse(rec: &Plane<u8>, src: &[u8], w: usize, h: usize) -> f64 {
  let mut acc = 0u64;
  for (j, row) in rec.rows_iter().take(h).enumerate() {
    for i in 0..w {
      let d = i64::from(row[i]) - i64::from(src[j * w + i]);
      acc += (d * d) as u64;
    }
  }
  acc as f64 / (w * h) as f64
}

fn psnr(mse: f64) -> f64 {
  if mse <= 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

struct Row {
  bytes: usize,
  psnr_y: f64,
  psnr_u: f64,
  psnr_v: f64,
  ms: u128,
  px_64x16: usize,
  px_16x64: usize,
}

/// Minimal single-frame IVF wrapper, matching `examples/recon_probe.rs`.
fn write_ivf(path: &str, w: usize, h: usize, frames: &[Vec<u8>]) {
  use std::io::Write;
  let mut f = fs::File::create(path).unwrap();
  let mut hdr = Vec::with_capacity(32);
  hdr.extend_from_slice(b"DKIF");
  hdr.extend_from_slice(&0u16.to_le_bytes());
  hdr.extend_from_slice(&32u16.to_le_bytes());
  hdr.extend_from_slice(b"AV01");
  hdr.extend_from_slice(&(w as u16).to_le_bytes());
  hdr.extend_from_slice(&(h as u16).to_le_bytes());
  hdr.extend_from_slice(&25u32.to_le_bytes());
  hdr.extend_from_slice(&1u32.to_le_bytes());
  hdr.extend_from_slice(&(frames.len() as u32).to_le_bytes());
  hdr.extend_from_slice(&0u32.to_le_bytes());
  f.write_all(&hdr).unwrap();
  for (i, data) in frames.iter().enumerate() {
    f.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(i as u64).to_le_bytes()).unwrap();
    f.write_all(data).unwrap();
  }
}

fn write_y4m(path: &str, w: usize, h: usize, rec: &Frame<u8>) {
  use std::io::Write;
  let mut f = fs::File::create(path).unwrap();
  writeln!(f, "YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C420jpeg").unwrap();
  writeln!(f, "FRAME").unwrap();
  let cw = w.div_ceil(2);
  let ch = h.div_ceil(2);
  for (p, &(pw, ph)) in
    rec.planes.iter().zip([(w, h), (cw, ch), (cw, ch)].iter())
  {
    for row in p.rows_iter().take(ph) {
      f.write_all(&row[..pw]).unwrap();
    }
  }
}

fn encode_one(
  planes: &[Vec<u8>; 3], w: usize, h: usize, speed: u8, q: usize, deep: bool,
  dump: Option<(&str, &str)>,
) -> Row {
  let mut ss = SpeedSettings::from_preset(speed);
  if deep {
    ss.partition.encode_bottomup = false;
    ss.partition.partition_range =
      PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_64X64);
    ss.partition.non_square_partition_max_threshold = BlockSize::BLOCK_64X64;
  }
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: q,
    min_quantizer: q as u8,
    still_picture: true,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let cw = w.div_ceil(2);
  let mut frame = ctx.new_frame();
  frame.planes[0].copy_from_raw_u8(&planes[0], w, 1);
  frame.planes[1].copy_from_raw_u8(&planes[1], cw, 1);
  frame.planes[2].copy_from_raw_u8(&planes[2], cw, 1);

  let start = Instant::now();
  ctx.send_frame(frame).unwrap();
  ctx.flush();
  let mut bytes = 0usize;
  let mut rec = None;
  let mut px_64x16 = 0usize;
  let mut px_16x64 = 0usize;
  let mut streams: Vec<Vec<u8>> = Vec::new();
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => {
        bytes += pkt.data.len();
        if dump.is_some() {
          streams.push(pkt.data.clone());
        }
        px_64x16 +=
          pkt.enc_stats.block_size_counts[BlockSize::BLOCK_64X16 as usize];
        px_16x64 +=
          pkt.enc_stats.block_size_counts[BlockSize::BLOCK_16X64 as usize];
        if pkt.rec.is_some() {
          rec = pkt.rec.clone();
        }
      }
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!("encode error: {e:?}"),
    }
  }
  let ms = start.elapsed().as_millis();
  let rec = rec.expect("still picture carries its recon");
  if let Some((dir, stem)) = dump {
    write_ivf(&format!("{dir}/{stem}.ivf"), w, h, &streams);
    write_y4m(&format!("{dir}/{stem}.rec.y4m"), w, h, rec.as_ref());
  }
  let ch = h.div_ceil(2);
  Row {
    bytes,
    psnr_y: psnr(plane_mse(&rec.planes[0], &planes[0], w, h)),
    psnr_u: psnr(plane_mse(&rec.planes[1], &planes[1], cw, ch)),
    psnr_v: psnr(plane_mse(&rec.planes[2], &planes[2], cw, ch)),
    ms,
    px_64x16,
    px_16x64,
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 5 {
    eprintln!(
      "usage: sliver64_rd <manifest.tsv> <speed> <q,q,...> <stock|deep>"
    );
    std::process::exit(2);
  }
  let speed: u8 = args[2].parse().expect("speed preset");
  let quantizers: Vec<usize> =
    args[3].split(',').map(|q| q.parse().expect("quantizer")).collect();
  let deep = match args[4].as_str() {
    "deep" => true,
    "stock" => false,
    other => panic!("mode must be `stock` or `deep`, got `{other}`"),
  };

  let dump_dir = std::env::var("SLIVER64_RD_DUMP").ok();
  if let Some(d) = &dump_dir {
    fs::create_dir_all(d).expect("dump dir");
  }

  let manifest = fs::read_to_string(&args[1]).expect("manifest");
  println!(
    "name\tw\th\tspeed\tmode\tq\tbytes\tbpp\tpsnr_y\tpsnr_u\tpsnr_v\t\
     psnr_avg\tms\tpx_64x16\tpx_16x64"
  );
  for line in manifest.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    let f: Vec<&str> = line.split('\t').collect();
    assert_eq!(f.len(), 4, "manifest row: name<TAB>path<TAB>w<TAB>h");
    let (name, path) = (f[0], f[1]);
    let w: usize = f[2].parse().expect("width");
    let h: usize = f[3].parse().expect("height");
    let rgb = fs::read(path).expect("raw rgb");
    let planes = rgb_to_i420(&rgb, w, h);
    for &q in &quantizers {
      let mode = args[4].as_str();
      let stem = format!("{name}_s{speed}_{mode}_q{q}");
      let r = encode_one(
        &planes,
        w,
        h,
        speed,
        q,
        deep,
        dump_dir.as_deref().map(|d| (d, stem.as_str())),
      );
      // 6:1:1 luma/chroma weighting, the usual 4:2:0 PSNR average.
      let psnr_avg = (6.0 * r.psnr_y + r.psnr_u + r.psnr_v) / 8.0;
      let bpp = 8.0 * r.bytes as f64 / (w * h) as f64;
      println!(
        "{name}\t{w}\t{h}\t{speed}\t{mode}\t{q}\t{}\t{bpp:.5}\t{:.4}\t{:.4}\t\
         {:.4}\t{psnr_avg:.4}\t{}\t{}\t{}",
        r.bytes, r.psnr_y, r.psnr_u, r.psnr_v, r.ms, r.px_64x16, r.px_16x64
      );
      use std::io::Write;
      std::io::stdout().flush().ok();
    }
  }
}
