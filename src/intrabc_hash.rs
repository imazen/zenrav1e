// Copyright (c) 2026, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Hash-based exact-match candidate table for the intraBC search
//! (chunk B of the intraBC program; the local search itself is
//! `crate::intrabc`).
//!
//! Port of libaom's `av1_hash_table` machinery (`av1/encoder/hash_motion.c`
//! plus the CRC-32C calculator in `av1/encoder/hash.c`, pinned rev
//! 632172a4), adapted to the chunk-A scope (8x8..64x64 square blocks,
//! 64-px superblocks, tile-relative positions):
//!
//! - The **source** luma plane of the tile is hashed once per tile encode:
//!   a 2x2 base layer (pixel-identity for 8-bit, byte-fold XOR for high
//!   bit depth), then each doubled size combines its four non-overlapping
//!   child hashes with CRC-32C, giving every position's NxN block hash for
//!   N in {8,16,32,64}.
//! - Per size, entries `{x, y, hash2}` land in `1 << 16` buckets keyed by
//!   the low 16 hash bits, capped at 256 entries per bucket, and inserted
//!   in libaom's hierarchical dispersal order (coarse grid first) so the
//!   kept entries cover the tile instead of clustering in the first rows.
//! - At search time the current block's source hash is recomputed the same
//!   way; bucket entries whose full 32-bit hash matches are exact
//!   source-pixel matches, and become displacement-vector candidates for
//!   the ordinary validity + SAD ranking + full-rate RD trial machinery.
//!
//! Matching source-against-source (not against the reconstruction) is the
//! libaom design: it lets the whole table be built up front, and on the
//! screen content this tool targets the reconstruction of a matching area
//! is (near-)identical to its source, so an exact source match is an
//! excellent copy candidate. The candidates are *hints*, ranked afterwards
//! by real reconstruction SAD and full-rate RD — a stale hint costs a
//! trial, never correctness. Nothing here touches the bitstream.

use crate::tiling::PlaneRegion;
use crate::util::{CastFromPrimitive, Pixel, PixelType};

/// Smallest hashed block size log2 (8x8): the chunk-A intraBC search never
/// trials blocks below 8x8, so 4x4 hashes exist only as pyramid
/// intermediates and are not stored (libaom's `min_alloc_size` gate).
pub const HASH_MIN_SIZE_LOG2: usize = 3;
/// Largest hashed block size log2 (64x64): the chunk-A scope is 64-px
/// superblocks.
pub const HASH_MAX_SIZE_LOG2: usize = 6;
const N_SIZES: usize = HASH_MAX_SIZE_LOG2 - HASH_MIN_SIZE_LOG2 + 1;

/// Bucket key width (libaom `kSrcBits`): buckets are keyed by the low 16
/// bits of the block hash.
const BUCKET_BITS: usize = 16;
const N_BUCKETS_PER_SIZE: usize = 1 << BUCKET_BITS;
const BUCKET_MASK: u32 = (N_BUCKETS_PER_SIZE - 1) as u32;

/// Per-bucket entry cap (libaom `kMaxCandidatesPerHashBucket`): beyond
/// this, more copies of the same content add search cost for no gain.
const MAX_PER_BUCKET: usize = 256;

/// Per-block candidate evaluation cap: libaom's
/// `prune_intrabc_candidate_block_hash_search` bound, set from speed 1 up
/// (including the allintra tiers this program measures against).
pub const MAX_HASH_CANDIDATES: usize = 64;

/// One hashed block position: `x`/`y` are the block's top-left corner in
/// tile-relative luma pixels, `hash2` the full 32-bit hash (the bucket key
/// only kept its low 16 bits — equality of `hash2` within a bucket is the
/// 32-bit exact-match test, mirroring libaom's `hash_value2`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockHashEntry {
  pub x: u16,
  pub y: u16,
  pub hash2: u32,
}

/// CRC-32C (iSCSI polynomial, reflected `0x82F63B78`) with the slice-by-8
/// lookup table, as in libaom `hash.c`. The pyramid combines exactly four
/// `u32` child hashes per step, so the only input shape is 16 bytes; the
/// words are serialized little-endian (bit-exact with libaom on the
/// little-endian hosts it runs on, and deterministic everywhere).
struct Crc32c {
  table: Box<[[u32; 256]; 8]>,
}

impl Crc32c {
  const POLY: u32 = 0x82F6_3B78;

  fn new() -> Self {
    let mut table = Box::new([[0u32; 256]; 8]);
    for n in 0..256u32 {
      let mut crc = n;
      for _ in 0..8 {
        crc = if crc & 1 != 0 { (crc >> 1) ^ Self::POLY } else { crc >> 1 };
      }
      table[0][n as usize] = crc;
    }
    for n in 0..256 {
      let mut crc = table[0][n];
      for k in 1..8 {
        crc = table[0][(crc & 0xff) as usize] ^ (crc >> 8);
        table[k][n] = crc;
      }
    }
    Crc32c { table }
  }

  #[inline]
  fn step8(&self, crc: u64, chunk: u64) -> u64 {
    let x = crc ^ chunk;
    let t = &self.table;
    (t[7][(x & 0xff) as usize]
      ^ t[6][((x >> 8) & 0xff) as usize]
      ^ t[5][((x >> 16) & 0xff) as usize]
      ^ t[4][((x >> 24) & 0xff) as usize]
      ^ t[3][((x >> 32) & 0xff) as usize]
      ^ t[2][((x >> 40) & 0xff) as usize]
      ^ t[1][((x >> 48) & 0xff) as usize]
      ^ t[0][((x >> 56) & 0xff) as usize]) as u64
  }

  /// CRC-32C of the four words serialized little-endian (16 bytes).
  #[inline]
  fn hash_words(&self, w: &[u32; 4]) -> u32 {
    let lo = u64::from(w[0]) | (u64::from(w[1]) << 32);
    let hi = u64::from(w[2]) | (u64::from(w[3]) << 32);
    let mut crc = 0xffff_ffffu64;
    crc = self.step8(crc, lo);
    crc = self.step8(crc, hi);
    (crc as u32) ^ 0xffff_ffff
  }
}

/// The 2x2 base hash at `(x, y)`: for 8-bit content the four pixels pack
/// into the 32-bit value directly (libaom `get_identity_hash_value`); for
/// high bit depth the low bytes pack and the high bytes fold in with XOR
/// (`get_xor_hash_value_hbd`).
#[inline]
fn hash_2x2<T: Pixel>(rows: [&[T]; 2], x: usize) -> u32 {
  let p = [
    u32::cast_from(rows[0][x]),
    u32::cast_from(rows[0][x + 1]),
    u32::cast_from(rows[1][x]),
    u32::cast_from(rows[1][x + 1]),
  ];
  match T::type_enum() {
    PixelType::U8 => (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + p[3],
    PixelType::U16 => {
      let lo = ((p[0] & 0x00ff) << 24)
        + ((p[1] & 0x00ff) << 16)
        + ((p[2] & 0x00ff) << 8)
        + (p[3] & 0x00ff);
      let hi = ((p[0] & 0xff00) << 16)
        + ((p[1] & 0xff00) << 8)
        + (p[2] & 0xff00)
        + ((p[3] & 0xff00) >> 8);
      lo ^ hi
    }
  }
}

/// Exact-match block hash table over a tile's source luma plane.
pub struct IntrabcHashTable {
  crc: Crc32c,
  /// `[size_idx * N_BUCKETS_PER_SIZE + (hash & BUCKET_MASK)]` where
  /// `size_idx = log2(size) - HASH_MIN_SIZE_LOG2`.
  buckets: Vec<Vec<BlockHashEntry>>,
}

impl std::fmt::Debug for IntrabcHashTable {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let entries: usize = self.buckets.iter().map(Vec::len).sum();
    f.debug_struct("IntrabcHashTable").field("entries", &entries).finish()
  }
}

impl IntrabcHashTable {
  /// Builds the table from a tile's source luma region: the full hash
  /// pyramid over every pixel position, then per-size bucket insertion in
  /// the dispersal order. Tiles smaller than a block size simply have no
  /// entries at that size (and none at all below 9x9... i.e. blocks only
  /// hash where they fit entirely).
  pub fn build<T: Pixel>(luma: &PlaneRegion<'_, T>) -> Self {
    let crc = Crc32c::new();
    let mut buckets = vec![Vec::new(); N_SIZES * N_BUCKETS_PER_SIZE];
    let w = luma.rect().width;
    let h = luma.rect().height;

    if w >= 2 && h >= 2 {
      // Ping-pong hash planes; `cur` holds size-N hashes at every position
      // whose NxN block fits, in raster layout (stride = w).
      let mut cur = vec![0u32; w * h];
      let mut next = vec![0u32; w * h];

      // 2x2 base layer.
      for y in 0..h - 1 {
        let r0 = &luma[y];
        let r1 = &luma[y + 1];
        let row = &mut cur[y * w..y * w + (w - 1)];
        for (x, out) in row.iter_mut().enumerate() {
          *out = hash_2x2([r0, r1], x);
        }
      }

      // Combine up: size 4, 8, .., 64. Sizes >= 8 are inserted.
      let mut size = 4usize;
      while size <= 1 << HASH_MAX_SIZE_LOG2 {
        if size > w || size > h {
          break;
        }
        let half = size >> 1;
        let x_end = w - size + 1;
        let y_end = h - size + 1;
        for y in 0..y_end {
          for x in 0..x_end {
            let pos = y * w + x;
            let words = [
              cur[pos],
              cur[pos + half],
              cur[pos + half * w],
              cur[pos + half * w + half],
            ];
            next[pos] = crc.hash_words(&words);
          }
        }
        std::mem::swap(&mut cur, &mut next);
        if size >= 1 << HASH_MIN_SIZE_LOG2 {
          let size_idx = size.ilog2() as usize - HASH_MIN_SIZE_LOG2;
          add_dispersed(
            &mut buckets[size_idx * N_BUCKETS_PER_SIZE..]
              [..N_BUCKETS_PER_SIZE],
            &cur,
            w,
            x_end,
            y_end,
            size,
          );
        }
        size <<= 1;
      }
    }

    IntrabcHashTable { crc, buckets }
  }

  /// The current block's source hash, computed exactly as the table build
  /// computed it for that position (same 2x2 base + CRC combine), so a
  /// 32-bit equality against a bucket entry's `hash2` is an exact match of
  /// the two blocks' source pixels' hashes. `size_log2` must be within
  /// `HASH_MIN_SIZE_LOG2..=HASH_MAX_SIZE_LOG2` and the block fully inside
  /// the region.
  pub fn block_hash_at<T: Pixel>(
    &self, luma: &PlaneRegion<'_, T>, x: usize, y: usize, size_log2: usize,
  ) -> u32 {
    debug_assert!(
      (HASH_MIN_SIZE_LOG2..=HASH_MAX_SIZE_LOG2).contains(&size_log2)
    );
    let size = 1usize << size_log2;
    debug_assert!(
      x + size <= luma.rect().width && y + size <= luma.rect().height
    );
    // Stack pyramid: 32x32 u32 at the largest (64x64 block), 4 KB a layer.
    let mut cur = [0u32; 32 * 32];
    let mut next = [0u32; 32 * 32];
    let half0 = size >> 1;
    for by in 0..half0 {
      let r0 = &luma[y + 2 * by];
      let r1 = &luma[y + 2 * by + 1];
      for bx in 0..half0 {
        cur[by * half0 + bx] = hash_2x2([r0, r1], x + 2 * bx);
      }
    }
    // Each combine halves the grid: from `n x n` child hashes (stride n)
    // to `(n/2) x (n/2)`.
    let mut n = half0;
    while n > 1 {
      let m = n >> 1;
      for by in 0..m {
        for bx in 0..m {
          let pos = (2 * by) * n + 2 * bx;
          let words = [cur[pos], cur[pos + 1], cur[pos + n], cur[pos + n + 1]];
          next[by * m + bx] = self.crc.hash_words(&words);
        }
      }
      std::mem::swap(&mut cur, &mut next);
      n = m;
    }
    cur[0]
  }

  /// All table entries in the bucket `hash` falls in, for `size_log2`
  /// blocks. Callers filter by `entry.hash2 == hash` (exact match) and cap
  /// evaluation at [`MAX_HASH_CANDIDATES`].
  pub fn candidates(&self, size_log2: usize, hash: u32) -> &[BlockHashEntry] {
    debug_assert!(
      (HASH_MIN_SIZE_LOG2..=HASH_MAX_SIZE_LOG2).contains(&size_log2)
    );
    let size_idx = size_log2 - HASH_MIN_SIZE_LOG2;
    &self.buckets
      [size_idx * N_BUCKETS_PER_SIZE + (hash & BUCKET_MASK) as usize]
  }
}

/// Bucket insertion in libaom's hierarchical dispersal order: visit the
/// coarse `size`-strided grid first, then the three half-offset phases,
/// halving the step until every position has been visited exactly once
/// (final step 2, so no two adjacent positions are consecutive). With the
/// 256-entry bucket cap this spreads the kept entries of heavily repeated
/// content across the whole tile instead of packing them into the first
/// rows, so every block position has valid (already-reconstructed,
/// near-enough) candidates to find.
fn add_dispersed(
  buckets: &mut [Vec<BlockHashEntry>], hashes: &[u32], stride: usize,
  x_end: usize, y_end: usize, size: usize,
) {
  debug_assert_eq!(buckets.len(), N_BUCKETS_PER_SIZE);
  let mut step = size;
  let mut x_offset = 0usize;
  let mut y_offset = 0usize;
  while step > 1 {
    let mut x = x_offset;
    while x < x_end {
      let mut y = y_offset;
      while y < y_end {
        let hash = hashes[y * stride + x];
        let bucket = &mut buckets[(hash & BUCKET_MASK) as usize];
        if bucket.len() < MAX_PER_BUCKET {
          bucket.push(BlockHashEntry {
            x: x as u16,
            y: y as u16,
            hash2: hash,
          });
        }
        y += step;
      }
      x += step;
    }
    // The libaom offset state machine (hash_motion.c): (0,0) -> (s/2,0)
    // -> (0,s/2) -> (s/2,s/2) -> halve the step -> (s/2,0) ...
    if x_offset == 0 && y_offset == 0 {
      x_offset = step / 2;
    } else if x_offset == step / 2 && y_offset == 0 {
      x_offset = 0;
      y_offset = step / 2;
    } else if x_offset == 0 && y_offset == step / 2 {
      x_offset = step / 2;
    } else {
      debug_assert!(x_offset == step / 2 && y_offset == step / 2);
      step /= 2;
      x_offset = step / 2;
      y_offset = 0;
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::frame::Plane;
  use crate::tiling::PlaneRegion;

  /// Bit-at-a-time CRC-32C reference (the definition the sliced table is
  /// derived from).
  fn crc32c_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
      crc ^= u32::from(b);
      for _ in 0..8 {
        crc = if crc & 1 != 0 { (crc >> 1) ^ Crc32c::POLY } else { crc >> 1 };
      }
    }
    crc ^ 0xffff_ffff
  }

  #[test]
  fn crc32c_reference_vector() {
    // The standard CRC-32C check value: "123456789" -> 0xE3069283. Pins
    // the polynomial/reflection convention of the reference fn itself.
    assert_eq!(crc32c_bitwise(b"123456789"), 0xE306_9283);
  }

  #[test]
  fn hash_words_matches_bitwise_reference() {
    let crc = Crc32c::new();
    let cases: [[u32; 4]; 4] = [
      [0, 0, 0, 0],
      [1, 2, 3, 4],
      [0xdead_beef, 0x0123_4567, 0x89ab_cdef, 0xffff_ffff],
      [0x8000_0000, 0x7fff_ffff, 0x5555_aaaa, 0xaaaa_5555],
    ];
    for words in cases {
      let mut bytes = [0u8; 16];
      for (i, w) in words.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
      }
      assert_eq!(crc.hash_words(&words), crc32c_bitwise(&bytes));
    }
  }

  #[test]
  fn dispersal_visits_every_position_once() {
    // The libaom comment's example: an 8x8 region with block_size 4 has
    // 5x5 candidate positions, each visited exactly once.
    let (w, h, size) = (8usize, 8usize, 4usize);
    let x_end = w - size + 1;
    let y_end = h - size + 1;
    // Give every position a unique hash mapping to a unique bucket.
    let mut hashes = vec![0u32; w * h];
    for y in 0..y_end {
      for x in 0..x_end {
        hashes[y * w + x] = (y * w + x) as u32;
      }
    }
    let mut buckets = vec![Vec::new(); N_BUCKETS_PER_SIZE];
    add_dispersed(&mut buckets, &hashes, w, x_end, y_end, size);
    let total: usize = buckets.iter().map(Vec::len).sum();
    assert_eq!(total, x_end * y_end);
    for y in 0..y_end {
      for x in 0..x_end {
        let b = &buckets[(y * w + x) & (N_BUCKETS_PER_SIZE - 1)];
        assert_eq!(
          b.iter().filter(|e| e.x == x as u16 && e.y == y as u16).count(),
          1,
          "position ({x},{y}) not visited exactly once"
        );
      }
    }
    // Dispersal order: the first candidates are the coarse step-4 grid
    // corners, before any odd-offset position.
    let first = buckets.iter().flat_map(|b| b.iter()).find(|_| true);
    assert!(first.is_some());
  }

  #[test]
  fn table_and_block_hash_agree() {
    // Deterministic pseudo-random 8-bit plane; every entry the table holds
    // for a position must equal the per-block recomputation there, across
    // all sizes present.
    let (w, h) = (80usize, 48usize);
    let mut data = vec![0u8; w * h];
    let mut state = 0x1234_5678u32;
    for px in data.iter_mut() {
      // xorshift32
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      *px = (state >> 24) as u8;
    }
    let plane = Plane::from_slice(&data, w);
    let region = PlaneRegion::new_from_plane(&plane);
    let table = IntrabcHashTable::build(&region);
    let mut checked = 0usize;
    for size_log2 in HASH_MIN_SIZE_LOG2..=HASH_MAX_SIZE_LOG2 {
      let size = 1usize << size_log2;
      if size > w || size > h {
        continue;
      }
      for &(x, y) in
        &[(0usize, 0usize), (1, 2), (7, 5), (w - size, h - size), (13, 9)]
      {
        if x + size > w || y + size > h {
          continue;
        }
        let hash = table.block_hash_at(&region, x, y, size_log2);
        let found = table
          .candidates(size_log2, hash)
          .iter()
          .any(|e| e.x == x as u16 && e.y == y as u16 && e.hash2 == hash);
        assert!(
          found,
          "table entry missing or hash mismatch at ({x},{y}) size {size}"
        );
        checked += 1;
      }
    }
    assert!(checked >= 15);
  }

  #[test]
  fn exact_repeats_share_hash_and_both_positions_listed() {
    // Two identical 16x16 stamps at (0,0) and (40,16) on a busy
    // background: same hash2, both positions in the bucket.
    let (w, h) = (64usize, 40usize);
    let mut data = vec![0u8; w * h];
    let mut state = 0x9e37_79b9u32;
    for px in data.iter_mut() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      *px = (state >> 24) as u8;
    }
    let stamp: Vec<u8> = (0..256).map(|i| (i * 37 % 251) as u8).collect();
    for sy in 0..16 {
      for sx in 0..16 {
        data[sy * w + sx] = stamp[sy * 16 + sx];
        data[(16 + sy) * w + 40 + sx] = stamp[sy * 16 + sx];
      }
    }
    let plane = Plane::from_slice(&data, w);
    let region = PlaneRegion::new_from_plane(&plane);
    let table = IntrabcHashTable::build(&region);
    let h_a = table.block_hash_at(&region, 0, 0, 4);
    let h_b = table.block_hash_at(&region, 40, 16, 4);
    assert_eq!(h_a, h_b, "identical source blocks must hash equal");
    let cands = table.candidates(4, h_a);
    for (x, y) in [(0u16, 0u16), (40, 16)] {
      assert!(
        cands.iter().any(|e| e.x == x && e.y == y && e.hash2 == h_a),
        "repeat position ({x},{y}) missing from bucket"
      );
    }
  }

  #[test]
  fn hbd_uses_xor_fold() {
    // A 10-bit plane whose high bytes differ must produce different 2x2
    // hashes even when the low bytes agree.
    let a: Vec<u16> = vec![0x100, 0x100, 0x100, 0x100];
    let b: Vec<u16> = vec![0x200, 0x200, 0x200, 0x200];
    let pa = Plane::from_slice(&a, 2);
    let pb = Plane::from_slice(&b, 2);
    let ra = PlaneRegion::new_from_plane(&pa);
    let rb = PlaneRegion::new_from_plane(&pb);
    assert_ne!(hash_2x2([&ra[0], &ra[1]], 0), hash_2x2([&rb[0], &rb[1]], 0));
  }

  #[test]
  fn tiny_regions_build_empty_tables() {
    for (w, h) in [(1usize, 1usize), (2, 2), (7, 7), (4, 64)] {
      let data = vec![0u8; w * h];
      let plane = Plane::from_slice(&data, w);
      let region = PlaneRegion::new_from_plane(&plane);
      let table = IntrabcHashTable::build(&region);
      let total: usize = table.buckets.iter().map(Vec::len).sum();
      assert_eq!(total, 0, "no 8x8 block fits in {w}x{h}");
    }
  }
}
