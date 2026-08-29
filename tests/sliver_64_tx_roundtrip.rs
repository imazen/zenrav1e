//! Roundtrip gate for the 64-dimension sliver transforms TX_64X16/TX_16X64
//! (zenrav1e#28): the max rect transforms of the BLOCK_64X16/BLOCK_16X64
//! slivers that PARTITION_HORZ_4/VERT_4 produce at BLOCK_64X64 parents.
//!
//! Those transforms desynced every conforming decoder until the eob_pt CDF
//! selection in `encode_eob` was keyed on the coded (512-coefficient) size
//! instead of the nominal 1024 one (`ContextWriter::eob_multi_size`). The
//! encoder used to cap them to TX_32X16/TX_16X32 (3fa735dc) and gate the
//! 64x64-parent 4-way candidates off for inter frames and for
//! TX_MODE_LARGEST; both are gone, so this test drives the real path in
//! every configuration those guards used to exclude:
//!
//! - intra, TX_MODE_LARGEST (`rdo_tx_decision = false`: no tx-size symbol,
//!   the decoder derives TX_64X16/TX_16X64 itself);
//! - intra, TX_MODE_SELECT (`rdo_tx_decision = true`: depth 0 written);
//! - inter, with and without `enable_inter_tx_split`, on all three band
//!   layouts (`Bands::Both` is the inter-frame `has_tr` desync repro);
//! - 4:2:0 and 4:4:4 (4:4:4 slivers carry a 64x16 chroma block, coded as
//!   two TX_32X16 units via `largest_chroma_tx_size`).
//!
//! The gate is exact: rav1d-safe's decoded frame (and aomdec's, when the
//! caller sets `SLIVER64_AOMDEC`) must byte-equal the encoder's own
//! reconstruction (`Packet::rec`) on every plane, so a parse
//! desync AND a recon-side divergence (dequant / inverse transform) both
//! fail. Liveness: the sliver-armed stream must differ from the
//! sliver-free control, or HORZ_4/VERT_4 never fired and the gate is
//! vacuous. Mutation-verified 2026-08-27: with `eob_multi_size` reverted to
//! `tx_size.area_log2() - 4` every sliver-armed intra case fails the exact
//! compare (the control cases still pass).
#![cfg(not(target_arch = "wasm32"))]

use zenrav1e::prelude::*;

/// Photo-like content with strong 16px-period structure: 16px-high
/// horizontal bands over the top half (BLOCK_64X16 from HORZ_4 on 64x64
/// parents is the natural fit) and 16px-wide vertical bands over the
/// bottom half (BLOCK_16X64 from VERT_4), on a gradient so the bands carry
/// real luma and chroma residual rather than skipping.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Bands {
  /// Horizontal bands top half, vertical bands bottom half.
  Both,
  /// Horizontal bands everywhere (BLOCK_64X16 / TX_64X16 only).
  Horizontal,
  /// Vertical bands everywhere (BLOCK_16X64 / TX_16X64 only).
  Vertical,
}

fn synth_bands(
  w: usize, h: usize, shift: usize, bands: Bands,
) -> [Vec<u8>; 3] {
  let mut y = vec![0u8; w * h];
  let mut u = vec![0u8; w * h];
  let mut v = vec![0u8; w * h];
  for j in 0..h {
    for i in 0..w {
      let (si, sj) = (i + shift, j + shift);
      let mut l = (40 + (si * 5 / 4 + sj) / 3 % 150) as u8;
      let horizontal = match bands {
        Bands::Both => j < h / 2,
        Bands::Horizontal => true,
        Bands::Vertical => false,
      };
      if horizontal {
        // 16px horizontal bands with a texture ramp inside each band.
        if sj % 32 < 16 {
          l = l.saturating_add(50 + (sj % 16) as u8);
        }
      } else if si % 32 < 16 {
        l = l.saturating_sub(50 + (si % 16) as u8);
      }
      y[j * w + i] = l;
      u[j * w + i] = (90 + (si + sj * 2) / 3 % 70) as u8;
      v[j * w + i] = if horizontal {
        100 + ((sj / 16) % 4 * 12) as u8
      } else {
        160 - ((si / 16) % 4 * 12) as u8
      };
    }
  }
  [y, u, v]
}

fn downsample2(src: &[u8], w: usize, h: usize) -> Vec<u8> {
  let (cw, ch) = (w / 2, h / 2);
  let mut out = vec![0u8; cw * ch];
  for j in 0..ch {
    for i in 0..cw {
      let s = src[(2 * j) * w + 2 * i] as u32
        + src[(2 * j) * w + 2 * i + 1] as u32
        + src[(2 * j + 1) * w + 2 * i] as u32
        + src[(2 * j + 1) * w + 2 * i + 1] as u32;
      out[j * cw + i] = ((s + 2) / 4) as u8;
    }
  }
  out
}

#[derive(Clone, Copy, Debug)]
struct Case {
  chroma: ChromaSampling,
  q: usize,
  /// `rdo_tx_decision`: false = TX_MODE_LARGEST, true = TX_MODE_SELECT.
  tx_select: bool,
  /// Number of frames; > 1 makes the tail frames inter-coded.
  frames: usize,
  inter_tx_split: bool,
  bands: Bands,
}

struct Encoded {
  packets: Vec<(Vec<u8>, std::sync::Arc<Frame<u8>>)>,
  /// Per-packet `block_size_counts` (pixels coded per BlockSize), for the
  /// liveness report: which sizes each stream actually used.
  bsize_counts: Vec<[usize; BlockSize::BLOCK_SIZES_ALL]>,
}

const BSIZE_NAMES: [&str; BlockSize::BLOCK_SIZES_ALL] = [
  "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16",
  "32x32", "32x64", "64x32", "64x64", "64x128", "128x64", "128x128", "4x16",
  "16x4", "8x32", "32x8", "16x64", "64x16",
];

fn bsize_report(counts: &[usize; BlockSize::BLOCK_SIZES_ALL]) -> String {
  counts
    .iter()
    .enumerate()
    .filter(|(_, c)| **c > 0)
    .map(|(i, c)| format!("{}:{}", BSIZE_NAMES[i], c / 16))
    .collect::<Vec<_>>()
    .join(" ")
}

fn encode(case: Case, w: usize, h: usize, threshold: BlockSize) -> Encoded {
  let mut ss = SpeedSettings::from_preset(2);
  ss.partition.encode_bottomup = false;
  // The full 4..64 range: 64x64 parents must be reachable AND splittable
  // so the 4-way candidates are offered against NONE/HORZ/VERT/SPLIT.
  ss.partition.partition_range =
    PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_64X64);
  ss.partition.non_square_partition_max_threshold = threshold;
  ss.prediction.prediction_modes = PredictionModesSetting::Simple;
  ss.transform.rdo_tx_decision = case.tx_select;
  ss.transform.enable_inter_tx_split = case.inter_tx_split;
  let enc = EncoderConfig {
    width: w,
    height: h,
    speed_settings: ss,
    quantizer: case.q,
    min_quantizer: case.q as u8,
    still_picture: case.frames == 1,
    low_latency: true,
    min_key_frame_interval: case.frames as u64,
    max_key_frame_interval: case.frames as u64,
    chroma_sampling: case.chroma,
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let xdec = usize::from(case.chroma == ChromaSampling::Cs420);
  for n in 0..case.frames {
    // Sub-block motion between frames so inter blocks carry residual and
    // the inter sliver partitions are RD-live rather than skip.
    let [y, u, v] = synth_bands(w, h, n * 3, case.bands);
    let (u, v) = if xdec == 1 {
      (downsample2(&u, w, h), downsample2(&v, w, h))
    } else {
      (u, v)
    };
    let mut f = ctx.new_frame();
    f.planes[0].copy_from_raw_u8(&y, w, 1);
    f.planes[1].copy_from_raw_u8(&u, w >> xdec, 1);
    f.planes[2].copy_from_raw_u8(&v, w >> xdec, 1);
    ctx.send_frame(f).unwrap();
  }
  ctx.flush();
  let mut packets = Vec::new();
  let mut bsize_counts = Vec::new();
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => {
        let rec = pkt.rec.clone().expect("shown frame carries its recon");
        bsize_counts.push(pkt.enc_stats.block_size_counts);
        packets.push((pkt.data, rec));
      }
      Err(EncoderStatus::LimitReached) => break,
      Err(EncoderStatus::Encoded) => {}
      Err(e) => panic!("encode error: {e:?}"),
    }
  }
  assert_eq!(packets.len(), case.frames, "{case:?}: packet count");
  Encoded { packets, bsize_counts }
}

/// Strided plane -> tight `w*h` buffer of the visible area.
fn plane_pixels(p: &Plane<u8>, w: usize, h: usize) -> Vec<u8> {
  let stride = p.cfg.stride;
  let data = p.data_origin();
  let mut out = Vec::with_capacity(w * h);
  for j in 0..h {
    out.extend_from_slice(&data[j * stride..j * stride + w]);
  }
  out
}

/// Decodes every packet with rav1d-safe and byte-compares each decoded
/// frame against the encoder's reconstruction of that frame.
fn assert_decoder_matches_recon(
  enc: &Encoded, case: Case, w: usize, h: usize, label: &str,
) {
  let xdec = usize::from(case.chroma == ChromaSampling::Cs420);
  let (cw, ch) = (w >> xdec, h >> xdec);
  let mut dec = rav1d_safe::Decoder::new().expect("decoder");
  let mut decoded: Vec<[Vec<u8>; 3]> = Vec::new();
  let collect = |frame: rav1d_safe::Frame, decoded: &mut Vec<[Vec<u8>; 3]>| {
    assert_eq!(
      (frame.width() as usize, frame.height() as usize),
      (w, h),
      "{label} {case:?}: decoded dimensions"
    );
    let rav1d_safe::Planes::Depth8(p) = frame.planes() else {
      panic!("{label} {case:?}: 8-bit stream decoded as high bit depth");
    };
    let mut planes: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (j, row) in p.y().rows().enumerate().take(h) {
      planes[0].extend_from_slice(&row[..w]);
      let _ = j;
    }
    for (pi, cp) in [p.u(), p.v()].into_iter().enumerate() {
      for row in cp.expect("chroma plane").rows().take(ch) {
        planes[pi + 1].extend_from_slice(&row[..cw]);
      }
    }
    decoded.push(planes);
  };
  for (n, (data, _)) in enc.packets.iter().enumerate() {
    if let Some(frame) = dec.decode(data).unwrap_or_else(|e| {
      panic!("{label} {case:?}: packet {n} decode error {e:?}")
    }) {
      collect(frame, &mut decoded);
    }
  }
  for frame in dec
    .flush()
    .unwrap_or_else(|e| panic!("{label} {case:?}: flush error {e:?}"))
  {
    collect(frame, &mut decoded);
  }
  assert_eq!(
    decoded.len(),
    enc.packets.len(),
    "{label} {case:?}: decoded frame count"
  );
  for (n, ((_, rec), dec_planes)) in
    enc.packets.iter().zip(decoded.iter()).enumerate()
  {
    for (pi, name) in ["Y", "U", "V"].into_iter().enumerate() {
      let (pw, ph) = if pi == 0 { (w, h) } else { (cw, ch) };
      let recon = plane_pixels(&rec.planes[pi], pw, ph);
      let got = &dec_planes[pi];
      if recon != *got {
        let first = recon.iter().zip(got).position(|(a, b)| a != b).unwrap();
        let mismatches = recon.iter().zip(got).filter(|(a, b)| a != b).count();
        if std::env::var("SLIVER64_DUMP_IVF").is_ok() {
          let positions: Vec<(usize, usize, u8, u8)> = recon
            .iter()
            .zip(got)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i % pw, i / pw, *a, *b))
            .collect();
          eprintln!("MISMATCH {label} f{n} {name}: {positions:?}");
        }
        panic!(
          "{label} {case:?}: frame {n} plane {name}: rav1d-safe output \
           diverges from the encoder recon at pixel {first} ({mismatches} \
           of {} differ) -- 64-dim sliver TU desync or recon mismatch",
          pw * ph
        );
      }
    }
  }
}

fn luma_psnr(rec: &Plane<u8>, src: &[u8], w: usize, h: usize) -> f64 {
  let r = plane_pixels(rec, w, h);
  let sse: u64 = r
    .iter()
    .zip(src)
    .map(|(&a, &b)| {
      let d = a as i64 - b as i64;
      (d * d) as u64
    })
    .sum();
  if sse == 0 {
    return 100.0;
  }
  10.0 * (255.0f64 * 255.0 / (sse as f64 / (w * h) as f64)).log10()
}

fn run(case: Case) {
  run_dims(case, 256, 256);
}

/// `run` at an explicit frame size. Sizes that are not a multiple of the
/// 64x64 superblock exercise the partial superblocks at the right/bottom
/// edge, where the 4:1 sliver chroma pairs sit on the frame boundary.
fn run_dims(case: Case, w: usize, h: usize) {
  let narrow = encode(case, w, h, BlockSize::BLOCK_8X8);
  let wide = encode(case, w, h, BlockSize::BLOCK_64X64);
  let narrow_bytes: usize = narrow.packets.iter().map(|p| p.0.len()).sum();
  let wide_bytes: usize = wide.packets.iter().map(|p| p.0.len()).sum();
  // Luma PSNR of the first (intra) frame against its source, for the
  // report only: an indicative rate/quality readout, not a gate.
  let src0 = synth_bands(w, h, 0, case.bands)[0].clone();
  let (npsnr, wpsnr) = (
    luma_psnr(&narrow.packets[0].1.planes[0], &src0, w, h),
    luma_psnr(&wide.packets[0].1.planes[0], &src0, w, h),
  );
  println!(
    "{case:?}: control={narrow_bytes} bytes (Y-PSNR f0 {npsnr:.2}), \
     slivers={wide_bytes} bytes (Y-PSNR f0 {wpsnr:.2})"
  );
  for (name, enc) in [("control", &narrow), ("slivers", &wide)] {
    for (n, counts) in enc.bsize_counts.iter().enumerate() {
      println!("  {name} frame {n} (4x4 units): {}", bsize_report(counts));
    }
  }
  assert!(
    wide.packets.iter().map(|p| &p.0).ne(narrow.packets.iter().map(|p| &p.0)),
    "{case:?}: sliver-armed and control streams are byte-identical -- \
     HORZ_4/VERT_4 never fired on 64x64 parents; the gate is vacuous"
  );
  // Additive dev hook: dump both streams as IVF for external decoders
  // (aomdec / dav1d conformance checks). Never gates or skips anything.
  if let Ok(dir) = std::env::var("SLIVER64_DUMP_IVF") {
    for (name, enc) in [("control", &narrow), ("slivers", &wide)] {
      // The band layout is part of the stem: without it the three
      // `Bands` variants of an otherwise identical case overwrite each
      // other and the external-decoder sweep silently loses streams.
      let path = format!(
        "{dir}/sliver64_{name}_{w}x{h}_{:?}_q{}_sel{}_f{}_split{}_{:?}.ivf",
        case.chroma,
        case.q,
        case.tx_select,
        case.frames,
        case.inter_tx_split,
        case.bands
      );
      std::fs::write(path, ivf(&enc.packets, w, h)).unwrap();
    }
  }
  assert_decoder_matches_recon(&narrow, case, w, h, "control");
  assert_decoder_matches_recon(&wide, case, w, h, "slivers");
  if let Ok(aomdec) = std::env::var("SLIVER64_AOMDEC") {
    assert_aomdec_matches_recon(&aomdec, &narrow, case, w, h, "control");
    assert_aomdec_matches_recon(&aomdec, &wide, case, w, h, "slivers");
  }
}

/// Decodes the stream with libaom's `aomdec --rawvideo` and byte-compares
/// its planar output against the encoder's reconstruction of every frame.
/// Caller-selected via `SLIVER64_AOMDEC`; a failure to run the binary is a
/// test failure, never a skip.
fn assert_aomdec_matches_recon(
  aomdec: &str, enc: &Encoded, case: Case, w: usize, h: usize, label: &str,
) {
  let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
  std::fs::create_dir_all(dir).unwrap();
  let stem = format!(
    "sliver64_{label}_{w}x{h}_{:?}_q{}_sel{}_f{}_split{}_{:?}",
    case.chroma,
    case.q,
    case.tx_select,
    case.frames,
    case.inter_tx_split,
    case.bands
  );
  let ivf_path = dir.join(format!("{stem}.ivf"));
  let raw_path = dir.join(format!("{stem}.yuv"));
  std::fs::write(&ivf_path, ivf(&enc.packets, w, h)).unwrap();
  let out = std::process::Command::new(aomdec)
    .arg("--rawvideo")
    .arg("-o")
    .arg(&raw_path)
    .arg(&ivf_path)
    .output()
    .unwrap_or_else(|e| panic!("{label} {case:?}: running {aomdec}: {e}"));
  let stderr = String::from_utf8_lossy(&out.stderr);
  assert!(
    out.status.success() && !stderr.contains("Corrupt"),
    "{label} {case:?}: aomdec rejected {}: status {:?}\n{stderr}",
    ivf_path.display(),
    out.status
  );
  let raw = std::fs::read(&raw_path).unwrap();
  let xdec = usize::from(case.chroma == ChromaSampling::Cs420);
  let (cw, ch) = (w >> xdec, h >> xdec);
  let mut expected = Vec::with_capacity(raw.len());
  for (_, rec) in &enc.packets {
    for (pi, (pw, ph)) in [(w, h), (cw, ch), (cw, ch)].into_iter().enumerate()
    {
      expected.extend(plane_pixels(&rec.planes[pi], pw, ph));
    }
  }
  assert_eq!(
    raw.len(),
    expected.len(),
    "{label} {case:?}: aomdec raw output size (frames decoded)"
  );
  if raw != expected {
    let first = raw.iter().zip(&expected).position(|(a, b)| a != b).unwrap();
    let mismatches = raw.iter().zip(&expected).filter(|(a, b)| a != b).count();
    panic!(
      "{label} {case:?}: aomdec output diverges from the encoder recon at \
       byte {first} ({mismatches} of {} differ)",
      raw.len()
    );
  }
}

fn ivf(
  packets: &[(Vec<u8>, std::sync::Arc<Frame<u8>>)], w: usize, h: usize,
) -> Vec<u8> {
  let mut ivf = Vec::new();
  ivf.extend_from_slice(b"DKIF");
  ivf.extend_from_slice(&0u16.to_le_bytes());
  ivf.extend_from_slice(&32u16.to_le_bytes());
  ivf.extend_from_slice(b"AV01");
  ivf.extend_from_slice(&(w as u16).to_le_bytes());
  ivf.extend_from_slice(&(h as u16).to_le_bytes());
  ivf.extend_from_slice(&25u32.to_le_bytes());
  ivf.extend_from_slice(&1u32.to_le_bytes());
  ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
  ivf.extend_from_slice(&0u32.to_le_bytes());
  for (n, (data, _)) in packets.iter().enumerate() {
    ivf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    ivf.extend_from_slice(&(n as u64).to_le_bytes());
    ivf.extend_from_slice(data);
  }
  ivf
}

#[test]
fn intra_slivers_tx_mode_largest_420() {
  for q in [60usize, 120, 200] {
    run(Case {
      chroma: ChromaSampling::Cs420,
      q,
      tx_select: false,
      frames: 1,
      inter_tx_split: false,
      bands: Bands::Both,
    });
  }
}

#[test]
fn intra_slivers_tx_mode_select_420() {
  for q in [60usize, 120, 200] {
    run(Case {
      chroma: ChromaSampling::Cs420,
      q,
      tx_select: true,
      frames: 1,
      inter_tx_split: false,
      bands: Bands::Both,
    });
  }
}

#[test]
fn intra_slivers_444_both_tx_modes() {
  for tx_select in [false, true] {
    run(Case {
      chroma: ChromaSampling::Cs444,
      q: 100,
      tx_select,
      frames: 1,
      inter_tx_split: false,
      bands: Bands::Both,
    });
  }
}

/// Inter 64x16 (Horizontal) and 16x64 (Vertical) slivers, coded as one
/// TX_64X16/TX_16X64 unit (no tx split) and as a var-tx tree split into two
/// TX_32X16/TX_16X32 leaves (`enable_inter_tx_split`).
///
/// `Bands::Both` is the permanent reproduction of the inter-frame desync
/// fixed 2026-08-28 (`has_tr` in `partition.rs`): its third frame codes a
/// HORZ_4 32x32 parent whose 3rd 32x8 sliver was given a top-right spatial
/// MV candidate no decoder adds (rav1e tested `(y & h) != 0` where libaom
/// tests `(mi_row & (w - 1)) != 0`), shifting the NEWMV/REFMV contexts and
/// desyncing the tile from that sliver's inter mode symbol on: rav1d-safe
/// InvalidData, dav1d "Invalid argument", aomdec "Corrupted segment_ids".
/// Mutation-verified: with the two pre-fix tests restored this case fails
/// at packet 2 (the Horizontal/Vertical cases still pass).
///
/// Decoder oracles: rav1d-safe in-process (always), and aomdec when the
/// caller sets `SLIVER64_AOMDEC=<path>` (the `gate-sliver64` justfile
/// recipe and the CI "Gate A5" job do). No runtime skip: with the variable
/// unset the aomdec leg is not part of the test.
#[test]
fn inter_slivers_with_and_without_tx_split() {
  for bands in [Bands::Horizontal, Bands::Vertical, Bands::Both] {
    for inter_tx_split in [false, true] {
      run(Case {
        chroma: ChromaSampling::Cs420,
        q: 100,
        tx_select: false,
        frames: 3,
        inter_tx_split,
        bands,
      });
    }
  }
}

/// Four-frame 4:2:0 inter repro for the 4:1 / 1:4 sliver **chroma** pair
/// divergence fixed 2026-08-28 (`chroma_shared_with_neighbour`,
/// `src/encoder.rs`).
///
/// BLOCK_4X16 and BLOCK_16X4 share their chroma block with the block to
/// their left / above, so the joint 4x8 (8x4) chroma block must be
/// inter-predicted from both blocks' motion vectors. `motion_compensate`
/// selected that path with `bsize < BlockSize::BLOCK_8X8`, and BlockSize's
/// `PartialOrd` is partial: the slivers compare as neither, so they
/// predicted the whole joint chroma block from their own MV. The
/// encoder's reconstruction then diverged from rav1d-safe / aomdec /
/// dav1d by up to ~60 levels over the neighbour's half of the chroma
/// block (plane U/V only, 4:2:0 only), and the drift entered the
/// reference frames.
///
/// Three frames were not enough to expose it (the stock
/// `inter_slivers_with_and_without_tx_split` cases pass on the pre-fix
/// encoder); four frames of accumulated inter prediction are. Measured
/// pre-fix on the wider grid this subset is drawn from: 27 of 60 4:2:0
/// cases failed, 0 of 60 4:4:4 cases.
#[test]
fn inter_sliver_chroma_pairs_match_decoder() {
  for bands in [Bands::Horizontal, Bands::Vertical, Bands::Both] {
    for q in [40usize, 60] {
      run(Case {
        chroma: ChromaSampling::Cs420,
        q,
        tx_select: false,
        frames: 4,
        inter_tx_split: false,
        bands,
      });
    }
  }
  // 4:4:4 subsamples nothing, so no block shares a chroma block and the
  // pair path must stay off: `chroma_shared_with_neighbour` returns false,
  // and these blocks take the single-MV path (which is what the spec asks
  // for here, and which the pre-fix `bsize < BLOCK_8X8` test would have
  // routed into a 4:2:0-only `assert!` for BLOCK_4X4/4X8/8X4).
  run(Case {
    chroma: ChromaSampling::Cs444,
    q: 60,
    tx_select: false,
    frames: 4,
    inter_tx_split: false,
    bands: Bands::Both,
  });
  // Frame sizes that are not a multiple of the superblock: the partial
  // superblocks at the right/bottom edge put sliver chroma pairs on the
  // frame boundary. Pre-fix these failed in BOTH chroma planes (the
  // 256x256 cases above only surfaced U), 5 of 15 configurations over
  // {192x192, 240x176, 320x240, 200x130, 130x200} x q{40,100,200}.
  // Frame sizes that are not a whole number of 64x64 superblocks: the
  // partial superblocks at the right/bottom edge put the sliver chroma
  // pairs on the frame boundary. Both cases are mutation-verified (they
  // fail on the pre-fix `bsize < BLOCK_8X8` test), and pre-fix they broke
  // BOTH chroma planes where the 256x256 cases above only surfaced U.
  for (w, h, q) in [(320usize, 240usize, 100usize), (200, 136, 200)] {
    run_dims(
      Case {
        chroma: ChromaSampling::Cs420,
        q,
        tx_select: false,
        frames: 4,
        inter_tx_split: false,
        bands: Bands::Both,
      },
      w,
      h,
    );
  }
}
