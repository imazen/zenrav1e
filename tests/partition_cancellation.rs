#![cfg(feature = "stop")]

use std::sync::{
  Arc,
  atomic::{AtomicUsize, Ordering},
};
use zenrav1e::prelude::*;

struct FireOnce {
  calls: AtomicUsize,
  fire_at: usize,
}

impl enough::Stop for FireOnce {
  fn check(&self) -> Result<(), enough::StopReason> {
    if self.calls.fetch_add(1, Ordering::Relaxed) == self.fire_at {
      Err(enough::StopReason::Cancelled)
    } else {
      Ok(())
    }
  }
}

fn encode(
  bottomup: bool, stop: Arc<FireOnce>,
) -> Result<Vec<u8>, EncoderStatus> {
  let mut speed = SpeedSettings::from_preset(10);
  speed.partition.encode_bottomup = bottomup;
  speed.partition.partition_range =
    PartitionRange::new(BlockSize::BLOCK_4X4, BlockSize::BLOCK_4X4);
  let config = EncoderConfig {
    width: 64,
    height: 64,
    chroma_sampling: ChromaSampling::Cs400,
    still_picture: true,
    low_latency: true,
    speed_settings: speed,
    ..Default::default()
  };
  let mut ctx: Context<u8> = Config::new()
    .with_encoder_config(config)
    .with_threads(1)
    .new_context()
    .unwrap();
  ctx.set_stop(stop);
  let mut frame = ctx.new_frame();
  for (y, row) in frame.planes[0]
    .mut_slice(Default::default())
    .rows_iter_mut()
    .take(64)
    .enumerate()
  {
    for (x, value) in row[..64].iter_mut().enumerate() {
      *value = ((x * 17 + y * 31 + x * y * 7) % 256) as u8;
    }
  }
  ctx.send_frame(frame).unwrap();
  ctx.flush();
  loop {
    match ctx.receive_packet() {
      Ok(packet) => return Ok(packet.data),
      Err(EncoderStatus::Encoded) => continue,
      Err(err) => return Err(err),
    }
  }
}

#[test]
fn partition_walk_propagates_a_single_cancelled_poll() {
  for bottomup in [false, true] {
    let count =
      Arc::new(FireOnce { calls: AtomicUsize::new(0), fire_at: usize::MAX });
    assert!(!encode(bottomup, count.clone()).unwrap().is_empty());
    let polls = count.calls.load(Ordering::Relaxed);
    assert!(polls > 10, "must exercise nested partition polls");
    // Sample both early and late nodes, including the last observed poll.
    for fire_at in [0, 1, 2, 5, polls / 2, polls - 1] {
      let stop = Arc::new(FireOnce { calls: AtomicUsize::new(0), fire_at });
      assert_eq!(
        encode(bottomup, stop),
        Err(EncoderStatus::Cancelled),
        "bottomup={bottomup}, poll={fire_at}"
      );
    }
  }
}
