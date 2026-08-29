#!/usr/bin/env bash
# gate-recon: encoder reconstruction must byte-agree with conforming
# decoders at every probed (speed, quantizer, filter-toggle) cell — the
# zenrav1e#32/#33 desync class as an executable gate (invariant A5 of
# zenavif docs/ENGINEERING_BASELINE.md).
#
# Drives examples/recon_probe.rs over the pinned gate images (emitted by
# examples/gate_identity.rs --emit-y4m, so recon and identity gates probe
# identical content), then decodes each IVF with:
#   - rav1d-safe, via zenavif's ivf_raw example (IVF_RAW), and/or
#   - libaom's reference decoder (AOMDEC, `aomdec --rawvideo`),
# and byte-compares each decode against the encoder's own reconstruction.
# Any difference is an encoder bug (the encoder optimized against a
# reconstruction no conforming decoder produces).
#
# Decoder legs are env-selected — the CALLER decides what runs (justfile
# defaults wire the rav1d-safe leg from the zenavif sibling checkout):
#   IVF_RAW   path to zenavif's ivf_raw example        (leg skipped if unset)
#   AOMDEC    path to aomdec                            (leg skipped if unset)
# At least ONE leg is required; zero legs is a loud failure, never a pass.
#
# Grid: 3 pinned images x s{2,6,8} x q{60,140,220} x {cdef1/lrf1, cdef0/lrf1}.
# The cdef-off+lrf-on corner is API-only (no preset reaches it) — the
# zenrav1e#32 latent sgrproj-skip class lived exactly there. filter-intra
# coverage rides the s2/s6 presets (prediction_modes >= ComplexKeyframes).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Scratch under $HOME/tmp, not /tmp: /tmp is wiped at unpredictable times on
# these dev boxes, and this script writes the y4m inputs, the IVFs and every
# decoder's raw output there. Override with WORK=<dir>.
WORK="${WORK:-$HOME/tmp/zenrav1e_gate_recon.$$}"
IVF_RAW="${IVF_RAW:-}"
AOMDEC="${AOMDEC:-}"

if [ -z "$IVF_RAW" ] && [ -z "$AOMDEC" ]; then
  echo "gate-recon: FATAL — no decoder leg configured." >&2
  echo "  Set IVF_RAW=path/to/zenavif ivf_raw example (build it with:" >&2
  echo "    cargo build --release --example ivf_raw   # in ../zenavif)" >&2
  echo "  and/or AOMDEC=path/to/aomdec." >&2
  exit 2
fi
[ -n "$IVF_RAW" ] && [ ! -x "$IVF_RAW" ] && {
  echo "gate-recon: FATAL — IVF_RAW not executable: $IVF_RAW" >&2
  exit 2
}
[ -n "$AOMDEC" ] && [ ! -x "$AOMDEC" ] && {
  echo "gate-recon: FATAL — AOMDEC not executable: $AOMDEC" >&2
  exit 2
}

leg_r="off"; [ -n "$IVF_RAW" ] && leg_r="on ($IVF_RAW)"
leg_a="off"; [ -n "$AOMDEC" ] && leg_a="on ($AOMDEC)"
echo "[gate-recon] legs: rav1d-safe=$leg_r aomdec=$leg_a"

cargo build --release --example recon_probe --example gate_identity \
  || exit 2
PROBE="$ROOT/target/release/examples/recon_probe"
GATEID="$ROOT/target/release/examples/gate_identity"

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
"$GATEID" --emit-y4m "$WORK/y4m" > /dev/null || exit 2

# image -> "w h" (must match examples/gate_identity.rs generators)
dims() {
  case "$1" in
    photo | screen) echo "128 128" ;;
    mixed) echo "131 97" ;;
    *) echo "0 0" ;;
  esac
}

fail=0
cells=0
for img in photo screen mixed; do
  read -r w h <<< "$(dims "$img")"
  [ "$w" -gt 0 ] || { echo "gate-recon: unknown image $img" >&2; exit 2; }
  cw=$(((w + 1) / 2)); ch=$(((h + 1) / 2))
  paylen=$((w * h + 2 * cw * ch))
  for s in 2 6 8; do
    for q in 60 140 220; do
      for filt in "1 1" "0 1"; do
        read -r cdef lrf <<< "$filt"
        cells=$((cells + 1))
        tag="$img/s$s/q$q/cdef$cdef-lrf$lrf"
        ivf="$WORK/c.ivf"; recon="$WORK/c.y4m"
        if ! "$PROBE" "$WORK/y4m/$img.y4m" "$ivf" "$recon" \
          "$q" "$s" "$cdef" "$lrf" 2> /dev/null; then
          echo "RECONFAIL $tag probe-error"; fail=$((fail + 1)); continue
        fi
        tail -c "$paylen" "$recon" > "$WORK/recon.raw"
        rmd5=$(md5sum < "$WORK/recon.raw" | cut -d' ' -f1)
        if [ -n "$IVF_RAW" ]; then
          if ! "$IVF_RAW" "$ivf" "$WORK/rav1d.raw" > /dev/null 2>&1; then
            echo "RECONFAIL $tag rav1d-decode-error"; fail=$((fail + 1))
            continue
          fi
          dmd5=$(md5sum < "$WORK/rav1d.raw" | cut -d' ' -f1)
          if [ "$rmd5" != "$dmd5" ]; then
            echo "RECONFAIL $tag recon!=rav1d-safe"; fail=$((fail + 1))
            continue
          fi
        fi
        if [ -n "$AOMDEC" ]; then
          if ! "$AOMDEC" --rawvideo -o "$WORK/aom.raw" "$ivf" \
            > /dev/null 2>&1; then
            echo "RECONFAIL $tag aomdec-rejects"; fail=$((fail + 1))
            continue
          fi
          amd5=$(md5sum < "$WORK/aom.raw" | cut -d' ' -f1)
          if [ "$rmd5" != "$amd5" ]; then
            echo "RECONFAIL $tag recon!=aomdec"; fail=$((fail + 1))
            continue
          fi
        fi
      done
    done
  done
done

echo "gate-recon: $cells cells, $fail failures" \
  "(legs: rav1d-safe=$leg_r aomdec=$leg_a)"
if [ "$fail" -gt 0 ]; then
  echo "gate-recon: FAIL"
  exit 1
fi
echo "gate-recon: PASS"
