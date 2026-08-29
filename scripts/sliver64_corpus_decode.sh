#!/bin/bash
# Corpus-sweep decode gate for the 64-dimension sliver transforms
# (zenrav1e#28 "done means", corpus half; the synthetic half is
# tests/sliver_64_tx_roundtrip.rs).
#
# Takes a directory of `<stem>.ivf` + `<stem>.rec.y4m` pairs written by
# `SLIVER64_RD_DUMP=<dir> cargo run --release --example sliver64_rd ...`
# and requires, for every stream, that aomdec AND dav1d decode it and that
# the decoded frame data is byte-equal to the encoder's own reconstruction.
# A decoder that merely accepts the stream is not enough: the encoder must
# have optimized against the reconstruction a decoder actually produces.
# Only the frame payload is compared -- each decoder writes its own y4m
# header text (frame rate, chroma siting), which is not part of the pixels.
#
# A third leg decodes with rav1d-safe when `IVF_RAW` points at an ivf_raw
# binary (same convention as scripts/gate_recon.sh). `just
# gate-sliver64-corpus` builds this repo's examples/ivf_raw and wires it, so
# the leg is on by default; it is skipped only when a caller clears IVF_RAW.
#
# usage: scripts/sliver64_corpus_decode.sh <dump-dir>
set -uo pipefail
dir=${1:?usage: sliver64_corpus_decode.sh <dump-dir>}
aomdec=${AOMDEC:-aomdec}
dav1d=${DAV1D:-dav1d}
ivf_raw=${IVF_RAW:-}
command -v "$aomdec" >/dev/null || { echo "aomdec not found"; exit 2; }
command -v "$dav1d" >/dev/null || { echo "dav1d not found"; exit 2; }
[ -n "$ivf_raw" ] && [ ! -x "$ivf_raw" ] && {
  echo "IVF_RAW not executable: $ivf_raw"; exit 2
}
leg_r="off"; [ -n "$ivf_raw" ] && leg_r="on ($ivf_raw)"
echo "[sliver64-corpus] legs: aomdec=$aomdec dav1d=$dav1d rav1d-safe=$leg_r"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Both decoders are asked for raw planar I420; the reference is the plane
# payload of the encoder's y4m reconstruction (its stream header and single
# `FRAME\n` marker stripped).
strip_y4m() {
  python3 - "$1" "$2" <<'PY'
import sys
b = open(sys.argv[1], 'rb').read()
i = b.index(b'\n') + 1                     # stream header
assert b[i:i + 5] == b'FRAME', sys.argv[1]
i = b.index(b'\n', i) + 1                  # frame header (single-frame still)
open(sys.argv[2], 'wb').write(b[i:])
PY
}

ok=0; fail=0
for ivf in "$dir"/*.ivf; do
  stem=$(basename "$ivf" .ivf)
  rec="$dir/$stem.rec.y4m"
  [ -f "$rec" ] || { echo "MISSING RECON $stem"; fail=$((fail+1)); continue; }
  strip_y4m "$rec" "$tmp/ref.yuv" || { echo "FAIL $stem (bad recon y4m)"; fail=$((fail+1)); continue; }
  bad=""
  if ! "$aomdec" --i420 -o "$tmp/a.yuv" "$ivf" >/dev/null 2>&1; then
    bad="aomdec-reject"
  elif ! cmp -s "$tmp/a.yuv" "$tmp/ref.yuv"; then
    bad="aomdec-mismatch"
  fi
  if [ -z "$bad" ]; then
    if ! "$dav1d" -i "$ivf" -o "$tmp/d.yuv" --muxer yuv >/dev/null 2>&1; then
      bad="dav1d-reject"
    elif ! cmp -s "$tmp/d.yuv" "$tmp/ref.yuv"; then
      bad="dav1d-mismatch"
    fi
  fi
  if [ -z "$bad" ] && [ -n "$ivf_raw" ]; then
    if ! "$ivf_raw" "$ivf" "$tmp/r.yuv" >/dev/null 2>&1; then
      bad="rav1d-safe-reject"
    elif ! cmp -s "$tmp/r.yuv" "$tmp/ref.yuv"; then
      bad="rav1d-safe-mismatch"
    fi
  fi
  if [ -n "$bad" ]; then
    echo "FAIL $stem ($bad)"
    fail=$((fail+1))
  else
    ok=$((ok+1))
  fi
done
echo "sliver64 corpus decode: $ok ok, $fail failed"
[ "$fail" -eq 0 ]
