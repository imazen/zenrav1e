//! Small animated/video canvases must retain their actual dimensions. Padding
//! belongs to backend allocations, never to the signaled or visible picture.
use zenrav1e::prelude::*;

fn components<P: Pixel>(
  frame: &Frame<P>, width: usize, height: usize, sampling: ChromaSampling,
) -> Vec<u16> {
  let (sx, sy) = match sampling {
    ChromaSampling::Cs420 => (2, 2),
    ChromaSampling::Cs422 => (2, 1),
    _ => (1, 1),
  };
  let mut output = Vec::new();
  for p in 0..if sampling == ChromaSampling::Cs400 { 1 } else { 3 } {
    let (w, h) = if p == 0 {
      (width, height)
    } else {
      (width.div_ceil(sx), height.div_ceil(sy))
    };
    for row in frame.planes[p].slice(Default::default()).rows_iter().take(h) {
      output.extend(row[..w].iter().map(|&v| u16::cast_from(v)));
    }
  }
  output
}

fn roundtrip<P: Pixel>(
  w: usize, h: usize, bits: usize, sampling: ChromaSampling, reorder: bool,
  q: usize, preset: u8,
) {
  let mut speed = SpeedSettings::from_preset(preset);
  speed.rdo_lookahead_frames = 1;
  let config = EncoderConfig {
    width: w,
    height: h,
    bit_depth: bits,
    chroma_sampling: sampling,
    still_picture: false,
    low_latency: !reorder,
    quantizer: q,
    min_quantizer: 0,
    min_key_frame_interval: 8,
    max_key_frame_interval: 8,
    speed_settings: speed,
    ..Default::default()
  };
  let mut encoder: Context<P> = Config::new()
    .with_encoder_config(config)
    .with_threads(1)
    .new_context()
    .unwrap_or_else(|e| {
      panic!("{w}x{h} depth{bits} {sampling:?} reorder{reorder}: {e}")
    });
  let mut sources = Vec::new();
  let (sx, sy) = match sampling {
    ChromaSampling::Cs420 => (2, 2),
    ChromaSampling::Cs422 => (2, 1),
    _ => (1, 1),
  };
  for n in 0..5 {
    let mut frame = encoder.new_frame();
    for p in 0..if sampling == ChromaSampling::Cs400 { 1 } else { 3 } {
      let (pw, ph) =
        if p == 0 { (w, h) } else { (w.div_ceil(sx), h.div_ceil(sy)) };
      for (y, row) in frame.planes[p]
        .mut_slice(Default::default())
        .rows_iter_mut()
        .take(ph)
        .enumerate()
      {
        for (x, out) in row[..pw].iter_mut().enumerate() {
          *out = P::cast_from(
            ((17 * x + 29 * y + 43 * n + 71 * p) % (1 << bits)) as u16,
          );
        }
      }
    }
    sources.push(components(&frame, w, h, sampling));
    encoder.send_frame(frame).unwrap();
  }
  encoder.send_frame(None).unwrap();
  let mut decoder = rav1d_safe::Decoder::new().unwrap();
  let mut bitstream = Vec::new();
  let mut reconstructed = Vec::new();
  let mut decoded = Vec::new();
  loop {
    match encoder.receive_packet() {
      Ok(packet) => {
        bitstream.extend_from_slice(&packet.data);
        assert_eq!(packet.input_frameno as usize, reconstructed.len());
        let rec = components(
          packet.rec.as_ref().expect("encoder reconstruction"),
          w,
          h,
          sampling,
        );
        if q == 0 {
          assert_eq!(
            rec, sources[packet.input_frameno as usize],
            "lossless source {w}x{h} {sampling:?} {bits}bits reorder{reorder} q{q} s{preset} frame{}",
            packet.input_frameno
          );
        }
        reconstructed.push(rec);
        if let Some(frame) = decoder.decode(&packet.data).unwrap() {
          decoded.push(frame);
        }
        while let Some(frame) = decoder.get_frame().unwrap() {
          decoded.push(frame);
        }
      }
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!(
        "encode {w}x{h} {sampling:?} {bits}bits reorder{reorder} q{q}: {e}"
      ),
    }
  }
  if let Some(directory) = std::env::var_os("SMALL_VIDEO_ARTIFACTS") {
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).unwrap();
    let tag = format!(
      "{w}x{h}-{sampling:?}-{bits}bit-reorder{reorder}-q{q}-s{preset}"
    );
    std::fs::write(directory.join(format!("{tag}.obu")), bitstream).unwrap();
    let raw: Vec<u8> = sources
      .iter()
      .flatten()
      .flat_map(|&v| {
        if bits == 8 { vec![v as u8] } else { v.to_le_bytes().to_vec() }
      })
      .collect();
    std::fs::write(directory.join(format!("{tag}.source.yuv")), raw).unwrap();
    let recon: Vec<u8> = reconstructed
      .iter()
      .flatten()
      .flat_map(|&v| {
        if bits == 8 { vec![v as u8] } else { v.to_le_bytes().to_vec() }
      })
      .collect();
    std::fs::write(directory.join(format!("{tag}.recon.yuv")), recon).unwrap();
  }
  decoded.extend(decoder.flush().unwrap());
  assert_eq!(decoded.len(), 5);
  for (i, frame) in decoded.iter().enumerate() {
    assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
    assert_eq!(frame.bit_depth() as usize, bits);
    let mut actual = Vec::new();
    macro_rules! collect {
      ($planes:expr) => {{
        let planes = $planes;
        for row in planes.y().rows().take(h) {
          actual.extend(row[..w].iter().map(|&v| u16::from(v)));
        }
        if sampling != ChromaSampling::Cs400 {
          for p in [planes.u().unwrap(), planes.v().unwrap()] {
            for row in p.rows().take(h.div_ceil(sy)) {
              actual
                .extend(row[..w.div_ceil(sx)].iter().map(|&v| u16::from(v)));
            }
          }
        }
      }};
    }
    match frame.planes() {
      rav1d_safe::Planes::Depth8(p) => collect!(p),
      rav1d_safe::Planes::Depth16(p) => collect!(p),
    }
    assert_eq!(
      actual, reconstructed[i],
      "reconstruction {w}x{h} {sampling:?} {bits}bits frame{i} reorder{reorder} q{q}"
    );
  }
}

#[test]
fn lossless_reordered_partial_edges_are_exact() {
  for (w, h) in [(17, 17), (17, 33), (33, 17)] {
    for sampling in [
      ChromaSampling::Cs400,
      ChromaSampling::Cs420,
      ChromaSampling::Cs422,
      ChromaSampling::Cs444,
    ] {
      for preset in [2, 6, 10] {
        roundtrip::<u8>(w, h, 8, sampling, true, 0, preset);
      }
    }
  }
}

#[test]
fn small_video_frames_match_source_and_encoder_reconstruction() {
  for (w, h) in [(1, 1), (1, 17), (17, 1), (3, 5), (7, 9), (17, 13), (16, 16)]
  {
    for sampling in [
      ChromaSampling::Cs400,
      ChromaSampling::Cs420,
      ChromaSampling::Cs422,
      ChromaSampling::Cs444,
    ] {
      for reorder in [false, true] {
        for preset in [2, 6, 10] {
          for q in [0, 80] {
            roundtrip::<u8>(w, h, 8, sampling, reorder, q, preset);
            roundtrip::<u16>(w, h, 10, sampling, reorder, q, preset);
            roundtrip::<u16>(w, h, 12, sampling, reorder, q, preset);
          }
        }
      }
    }
  }
}
