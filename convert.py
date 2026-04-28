"""Convert raw_demos*/*.npz trajectory files to a single MPAIL-format .pt file.

Usage:
    python convert.py                          # merges raw_demos2 only (default)
    python convert.py --dirs raw_demos2        # same
    python convert.py --dirs raw_demos raw_demos2 raw_demos3   # all demos
    python convert.py --out my_demos.pt        # custom output path
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def convert(source_dirs: list[str], out_path: str) -> None:
    all_data: dict = {}
    total_files = 0

    for dir_name in source_dirs:
        src = Path(dir_name)
        if not src.exists():
            print(f"[WARN] {src} does not exist — skipping")
            continue
        npz_files = sorted(src.glob("*.npz"))
        if not npz_files:
            print(f"[WARN] {src} contains no .npz files — skipping")
            continue
        for npz_file in npz_files:
            d = np.load(str(npz_file), allow_pickle=True)
            for key in d.files:
                arr = torch.from_numpy(d[key].astype(np.float32))
                all_data.setdefault(key, []).append(arr)
            total_files += 1
            print(f"  loaded {npz_file}  keys={list(d.files)}")

    if not all_data:
        raise RuntimeError("No data found in any of the source directories.")

    merged = {k: torch.cat(vs, dim=0) for k, vs in all_data.items()}

    # Keep only MPAIL-format keys: shape (N, 2, *obs_shape) — drop "actions" etc.
    demos = {k: v for k, v in merged.items() if v.dim() >= 3 and v.shape[1] == 2}

    if not demos:
        raise RuntimeError(
            "No MPAIL-format tensors found (expected shape [N, 2, *obs_shape]). "
            "Keys found: " + str(list(merged.keys()))
        )

    torch.save(demos, out_path)
    print(f"\nSaved {out_path}")
    print(f"  Source files : {total_files}")
    for k, v in demos.items():
        print(f"  {k}: {tuple(v.shape)}  ({v.numel() * 4 / 1e6:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs", nargs="+", default=["raw_demos2"],
        help="Source directories containing .npz trajectory files (default: raw_demos2)",
    )
    parser.add_argument(
        "--out", default="raw_demos2_master.pt",
        help="Output .pt file path (default: raw_demos2_master.pt)",
    )
    args = parser.parse_args()
    convert(args.dirs, args.out)
