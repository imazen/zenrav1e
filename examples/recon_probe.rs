// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Encoder-recon-vs-decoder desync probe (zenrav1e#32/#33 harness).
//!
//! Encodes a single-frame 8-bit 4:2:0 y4m as a still picture and writes both
//! the bitstream (IVF) and the encoder's own reconstruction (y4m), with
//! direct control over the `cdef`/`lrf` speed-setting toggles that the CLI
//! does not expose. Decode the IVF with a conforming decoder (aomdec /
//! rav1d-safe) and compare against the recon: any pixel difference is an
//! encoder bug (the encoder optimized against a reconstruction no decoder
//! produces).
//!
//! Usage:
//!   recon_probe <in.y4m> <out.ivf> <recon.y4m> <quantizer> <speed> \
//!               <cdef 0|1> <lrf 0|1>

use std::fs;
use std::io::{Read, Write};

use zenrav1e::config::SpeedSettings;
use zenrav1e::prelude::*;

fn parse_y4m(path: &str) -> (usize, usize, Vec<u8>) {
  let mut data = Vec::new();
  fs::File::open(path).unwrap().read_to_end(&mut data).unwrap();
  let hdr_end = data.iter().position(|&b| b == b'\n').unwrap();
  let hdr = std::str::from_utf8(&data[..hdr_end]).unwrap();
  assert!(hdr.starts_with("YUV4MPEG2"), "not a y4m file");
  let mut w = 0usize;
  let mut h = 0usize;
  for tok in hdr.split_whitespace().skip(1) {
    match tok.as_bytes()[0] {
      b'W' => w = tok[1..].parse().unwrap(),
      b'H' => h = tok[1..].parse().unwrap(),
      b'C' => assert!(
        tok[1..].starts_with("420"),
        "recon_probe only handles 4:2:0, got {tok}"
      ),
      _ => {}
    }
  }
  assert!(w > 0 && h > 0, "bad y4m header: {hdr}");
  let frame_hdr_end = hdr_end
    + 1
    + data[hdr_end + 1..].iter().position(|&b| b == b'\n').unwrap()
    + 1;
  assert!(data[hdr_end + 1..].starts_with(b"FRAME"));
  let cw = w.div_ceil(2);
  let ch = h.div_ceil(2);
  let frame_len = w * h + 2 * cw * ch;
  let frame = data[frame_hdr_end..frame_hdr_end + frame_len].to_vec();
  (w, h, frame)
}

fn write_ivf(path: &str, w: usize, h: usize, frames: &[Vec<u8>]) {
  let mut f = fs::File::create(path).unwrap();
  let mut hdr = Vec::with_capacity(32);
  hdr.extend_from_slice(b"DKIF");
  hdr.extend_from_slice(&0u16.to_le_bytes()); // version
  hdr.extend_from_slice(&32u16.to_le_bytes()); // header size
  hdr.extend_from_slice(b"AV01");
  hdr.extend_from_slice(&(w as u16).to_le_bytes());
  hdr.extend_from_slice(&(h as u16).to_le_bytes());
  hdr.extend_from_slice(&25u32.to_le_bytes()); // timebase den
  hdr.extend_from_slice(&1u32.to_le_bytes()); // timebase num
  hdr.extend_from_slice(&(frames.len() as u32).to_le_bytes());
  hdr.extend_from_slice(&0u32.to_le_bytes()); // unused
  f.write_all(&hdr).unwrap();
  for (i, data) in frames.iter().enumerate() {
    f.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(i as u64).to_le_bytes()).unwrap();
    f.write_all(data).unwrap();
  }
}

fn write_recon_y4m(path: &str, w: usize, h: usize, rec: &Frame<u8>) {
  let mut f = fs::File::create(path).unwrap();
  writeln!(f, "YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C420jpeg").unwrap();
  writeln!(f, "FRAME").unwrap();
  let dims =
    [(w, h), (w.div_ceil(2), h.div_ceil(2)), (w.div_ceil(2), h.div_ceil(2))];
  for (p, &(pw, ph)) in rec.planes.iter().zip(dims.iter()) {
    for row in p.rows_iter().take(ph) {
      f.write_all(&row[..pw]).unwrap();
    }
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 8 {
    eprintln!(
      "usage: recon_probe <in.y4m> <out.ivf> <recon.y4m> <quantizer> \
       <speed> <cdef 0|1> <lrf 0|1>"
    );
    std::process::exit(2);
  }
  let (w, h, yuv) = parse_y4m(&args[1]);
  let quantizer: usize = args[4].parse().unwrap();
  let speed: u8 = args[5].parse().unwrap();
  let cdef = args[6] != "0";
  let lrf = args[7] != "0";

  let mut speed_settings = SpeedSettings::from_preset(speed);
  speed_settings.cdef = cdef;
  speed_settings.lrf = lrf;

  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings,
    quantizer,
    min_quantizer: quantizer as u8,
    still_picture: true,
    chroma_sampling: ChromaSampling::Cs420,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();

  let mut frame = ctx.new_frame();
  let cw = w.div_ceil(2);
  let ch = h.div_ceil(2);
  let (y, uv) = yuv.split_at(w * h);
  let (u, v) = uv.split_at(cw * ch);
  frame.planes[0].copy_from_raw_u8(y, w, 1);
  frame.planes[1].copy_from_raw_u8(u, cw, 1);
  frame.planes[2].copy_from_raw_u8(v, cw, 1);

  ctx.send_frame(frame).unwrap();
  ctx.flush();

  let mut packets = Vec::new();
  let mut rec = None;
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => {
        if pkt.rec.is_some() {
          rec = pkt.rec.clone();
        }
        packets.push(pkt.data);
      }
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!("encode error: {e:?}"),
    }
  }
  write_ivf(&args[2], w, h, &packets);
  write_recon_y4m(&args[3], w, h, rec.expect("no recon frame").as_ref());
  eprintln!(
    "recon_probe: {} packets, cdef={cdef} lrf={lrf} q={quantizer} s={speed}",
    packets.len()
  );
}
