"""Download Yambda files from the Hugging Face Hub.

Usage::

    uv run yambda-download                      # flat/50m/multi_event + artist/album mappings
    uv run yambda-download --embeddings         # + audio embeddings (~13.8 GB, all 7.7M tracks)
    uv run yambda-download --out /content/drive/MyDrive/yambda/raw
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "yandex/yambda"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--size", default="50m", choices=["50m", "500m", "5b"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("RECSYS_DATA_ROOT", "data")) / "raw",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=["multi_event"],
        help="flat/<size>/<event>.parquet files: multi_event, listens, likes, dislikes, ...",
    )
    parser.add_argument("--embeddings", action="store_true", help="also fetch embeddings.parquet")
    args = parser.parse_args()

    patterns = [f"flat/{args.size}/{event}.parquet" for event in args.events]
    patterns += ["artist_item_mapping.parquet", "album_item_mapping.parquet"]
    if args.embeddings:
        patterns.append("embeddings.parquet")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=args.out,
    )
    print(f"Downloaded {patterns} -> {path}")


if __name__ == "__main__":
    main()
