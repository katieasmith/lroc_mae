#!/usr/bin/env python3
"""Create an animated GIF from a folder of PNG images.

Usage:
    python make_gif.py /path/to/pngs output.gif --fps 10 --scale 0.5
"""
import argparse
import re
from pathlib import Path

from PIL import Image


def natural_key(path: Path):
    """Sort 'frame_2.png' before 'frame_10.png'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def main():
    p = argparse.ArgumentParser(description="Make a GIF from a folder of PNGs.")
    p.add_argument("folder", type=Path, help="Folder containing .png files")
    p.add_argument("output", type=Path, help="Output .gif path")
    p.add_argument("--fps", type=float, default=10, help="Frames per second (default 10)")
    p.add_argument("--scale", type=float, default=1.0, help="Resize factor, e.g. 0.5 (default 1.0)")
    p.add_argument("--step", type=int, default=1, help="Use every Nth frame (default 1)")
    p.add_argument("--loop", type=int, default=0, help="Loop count, 0 = forever (default 0)")
    args = p.parse_args()

    files = sorted(args.folder.glob("*.PNG"), key=natural_key)[:: args.step]
    if not files:
        raise SystemExit(f"No .png files found in {args.folder}")
    print(f"Found {len(files)} frames")

    frames = []
    size = None
    for i, f in enumerate(files, 1):
        with Image.open(f) as im:
            im = im.convert("RGB")
            if size is None:
                size = (round(im.width * args.scale), round(im.height * args.scale))
            if im.size != size:
                im = im.resize(size, Image.LANCZOS)
            # Quantize each frame to a 256-color palette (GIF limit)
            frames.append(im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))
        if i % 50 == 0 or i == len(files):
            print(f"  loaded {i}/{len(files)}")

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),  # ms per frame
        loop=args.loop,
        optimize=False,
        disposal=2,
    )
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
