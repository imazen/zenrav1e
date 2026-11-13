#!/usr/bin/env python3
"""Require full-range AV1 signaling for the trace example's full-range PPM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("encoder", type=Path)
  parser.add_argument("out", type=Path)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=False)
  width, height = 16, 16
  pixels = bytes(channel for y in range(height) for x in range(width)
                 for channel in ((0, 0, 0) if x < 8 else (255, 255, 255)))
  source = args.out / "black-white.ppm"
  source.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)
  stream = args.out / "trace.ivf"
  command = [str(args.encoder.resolve()), str(args.out / "trace.tsv"), str(source),
             "--speed", "10", "--quantizer", "100", "--ivf-out", str(stream)]
  with (args.out / "encode.log").open("x") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
  result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
      "-show_entries", "stream=codec_name,width,height,pix_fmt,color_range",
      "-of", "json", str(stream)], check=True, text=True, capture_output=True)
  report = json.loads(result.stdout)
  report["encoder_sha256"] = hashlib.sha256(args.encoder.read_bytes()).hexdigest()
  report["stream_sha256"] = hashlib.sha256(stream.read_bytes()).hexdigest()
  report["ffprobe_version"] = subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0]
  with (args.out / "independent-decode.log").open("x") as log:
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(stream),
                    "-f", "null", "-"], check=True, stdout=log, stderr=subprocess.STDOUT)
  (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  assert len(report["streams"]) == 1, report
  actual = report["streams"][0]
  assert (actual["codec_name"], actual["width"], actual["height"]) == ("av1", width, height), report
  assert actual.get("color_range") == "pc", f"full-range source signaled as {actual}"
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
