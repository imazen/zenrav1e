//! Whole still-image encoding with compile-time assembly and Rust variants.
//! Run separately with default features and with --features asm. The build
//! label records the compiled path; these are independent-run comparisons.

use zenbench::prelude::*;
use zenrav1e::prelude::*;

/// Noise + patches. A gradient would produce degenerate residuals and
/// understate exactly the transform/prediction kernels this is measuring.
fn synth(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; w * h];
  let mut u = vec![0u8; (w / 2) * (h / 2)];
  let mut v = vec![0u8; (w / 2) * (h / 2)];
  let mut s = 0x9e37_79b9u32;
  let mut next = move || {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (s >> 24) as u8
  };
  for j in 0..h {
    for i in 0..w {
      let patch = ((i / 32 + j / 32) & 3) as u8;
      y[j * w + i] = next().wrapping_add(patch.wrapping_mul(40));
    }
  }
  for j in 0..h / 2 {
    for i in 0..w / 2 {
      let patch = ((i / 16 + j / 16) & 3) as u8;
      u[j * (w / 2) + i] = next().wrapping_add(patch.wrapping_mul(60));
      v[j * (w / 2) + i] = next().wrapping_add(patch.wrapping_mul(90));
    }
  }
  (y, u, v)
}

fn encode_once(
  w: usize, h: usize, planes: &(Vec<u8>, Vec<u8>, Vec<u8>), speed: u8,
) -> Packet<u8> {
  let enc = EncoderConfig {
    width: w,
    height: h,
    bit_depth: 8,
    chroma_sampling: ChromaSampling::Cs420,
    still_picture: true,
    low_latency: true,
    quantizer: 100,
    speed_settings: SpeedSettings::from_preset(speed),
    ..Default::default()
  };
  let cfg = Config::new().with_encoder_config(enc).with_threads(1);
  let mut ctx: Context<u8> = cfg.new_context().unwrap();
  let mut f = ctx.new_frame();
  f.planes[0].copy_from_raw_u8(&planes.0, w, 1);
  f.planes[1].copy_from_raw_u8(&planes.1, w / 2, 1);
  f.planes[2].copy_from_raw_u8(&planes.2, w / 2, 1);
  ctx.send_frame(f).unwrap();
  ctx.flush();
  let mut packet = None;
  loop {
    match ctx.receive_packet() {
      Ok(pkt) => {
        assert!(packet.is_none(), "still encoder emitted multiple packets");
        packet = Some(pkt);
      }
      Err(EncoderStatus::Encoded) => {}
      Err(EncoderStatus::LimitReached) => break,
      Err(e) => panic!("still encode failed after flush: {e}"),
    }
  }
  packet.expect("still encoder emitted no packet")
}

fn verify_recon(packet: &Packet<u8>, w: usize, h: usize) {
  let mut decoder = rav1d_safe::Decoder::new().expect("decoder creation");
  let mut frames = Vec::new();
  if let Some(frame) =
    decoder.decode(&packet.data).expect("decode encoded OBU")
  {
    frames.push(frame);
  }
  frames.extend(decoder.flush().expect("flush decoder"));
  assert_eq!(frames.len(), 1);
  let frame = &frames[0];
  assert_eq!((frame.width() as usize, frame.height() as usize), (w, h));
  let rav1d_safe::Planes::Depth8(planes) = frame.planes() else {
    panic!("expected 8-bit output");
  };
  let rec = packet.rec.as_ref().expect("encoder reconstruction");
  for (i, plane) in
    [Some(planes.y()), planes.u(), planes.v()].into_iter().enumerate()
  {
    let pw = if i == 0 { w } else { w / 2 };
    let ph = if i == 0 { h } else { h / 2 };
    let plane = plane.expect("4:2:0 plane");
    let expected = &rec.planes[i];
    let data = expected.data_origin();
    let rows: Vec<_> = plane.rows().collect();
    assert_eq!(rows.len(), ph);
    for (y, row) in rows.into_iter().enumerate() {
      assert_eq!(
        row,
        &data[y * expected.cfg.stride..][..pw],
        "recon plane {i}, row {y}"
      );
    }
  }
}

fn bench_encode(suite: &mut Suite) {
  // Label the arm by what was actually compiled in, so the two runs are not
  // confusable after the fact.
  let arm = if cfg!(asm_neon) {
    "neon_asm"
  } else if cfg!(nasm_x86_64) {
    "x86_asm"
  } else {
    "rust_fallback"
  };
  eprintln!("[tier_isolation] built with: {arm}");

  for &(label, w, h) in
    &[("256x256", 256usize, 256usize), ("512x512", 512, 512)]
  {
    let planes = synth(w, h);
    let packet = encode_once(w, h, &planes, 8);
    verify_recon(&packet, w, h);
    let root = std::env::var_os("ZENRAV1E_BENCH_ARTIFACT_DIR")
      .map(std::path::PathBuf::from)
      .unwrap_or_else(|| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
          .join("../../codec-artifacts/zenrav1e-arm-audit")
      });
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join(format!("{label}-{arm}.obu")), &packet.data)
      .unwrap();
    eprintln!(
      "fixture {label}/{arm}: {} bytes, exact decoder/reconstruction parity passed",
      packet.data.len()
    );
    suite.group(format!("encode_still/{label}"), |g| {
      g.bench(arm, move |b| {
        b.iter(|| encode_once(w, h, black_box(&planes), 8))
      });
    });
  }
}

zenbench::main!(bench_encode);
