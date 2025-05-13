#!/usr/bin/env python3
"""
scripts/download_data.py

Generic downloader for text files in a Hugging Face dataset repo.

Example usage for TinyStories:

  # Original version, train+validation
  python scripts/download_data.py \
    --repo-id roneneldan/TinyStories \
    --prefix TinyStories \
    --split all

  # GPT-4 version, train only
  python scripts/download_data.py \
    --repo-id roneneldan/TinyStories \
    --prefix TinyStoriesV2-GPT4 \
    --split train
"""

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

def download_file(repo_id: str, filename: str, out_dir: Path) -> None:
    """
    Download a single file from HF hub and save it as out_dir/filename.
    """
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    dest = out_dir / filename
    try:
        print(f"⏬ Downloading {filename} …")
        urlretrieve(url, dest)
        print(f"✔ Saved to {dest}")
    except HTTPError as e:
        print(f"❌ HTTP error {e.code} for {filename}", file=sys.stderr)
    except URLError as e:
        print(f"❌ URL error {e.reason} for {filename}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Download text files from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo identifier, e.g. roneneldan/TinyStories",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help=(
            "Filename prefix in the repo, e.g. 'TinyStories' or "
            "'TinyStoriesV2-GPT4'; script will fetch PREFIX-train.txt "
            "and/or PREFIX-valid.txt"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "all"],
        default="all",
        help="Which split(s) to download: train, valid, or all",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save downloaded files (default: data/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {"train": "train", "valid": "valid"}
    to_fetch = []

    if args.split in ("train", "all"):
        to_fetch.append(f"{args.prefix}-{suffix_map['train']}.txt")
    if args.split in ("valid", "all"):
        to_fetch.append(f"{args.prefix}-{suffix_map['valid']}.txt")

    if not to_fetch:
        print("❌ No files to download for split =", args.split, file=sys.stderr)
        sys.exit(1)

    for fname in to_fetch:
        download_file(args.repo_id, fname, out_dir)

if __name__ == "__main__":
    main()
