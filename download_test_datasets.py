#!/usr/bin/env python3
"""
The Turing Project (2026) — Out-of-Distribution Test Dataset Downloader
Downloads an independent, new set of benchmarks to test models on:
1. Image Test Benchmark: Hemg AI-Generated-vs-Real-Images (~25 MB parquet)
2. Text Test Benchmark: Ateeqq AI-and-Human-Generated-Text (~8.3 MB CSV)
3. Audio Test Benchmark: Hemg Deepfake-Audio-Dataset (~81 MB parquet)
Total download size: ~114 MB (strictly respecting storage budget).
"""

import os
import sys
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_benchmark")
os.makedirs(os.path.join(DATA_DIR, "image"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "text"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "video"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "document"), exist_ok=True)

DATASETS = [
    {
        "name": "Text Test Set (Ateeqq AI & Human Generated)",
        "url": "https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text/resolve/main/test.csv",
        "dest": os.path.join(DATA_DIR, "text", "ai_human_test.csv"),
    },
    {
        "name": "Real Art Image Test Set (Hemg Real Art Shard 1)",
        "url": "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/resolve/main/data/train-00001-of-00006-8ad2d550254dea81.parquet",
        "dest": os.path.join(DATA_DIR, "image", "real_art_test.parquet"),
    },
    {
        "name": "AI Art Image Test Set (Hemg AI Art Shard 4)",
        "url": "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/resolve/main/data/train-00004-of-00006-1101eaf5152e1c5f.parquet",
        "dest": os.path.join(DATA_DIR, "image", "ai_art_test.parquet"),
    },
    {
        "name": "Audio Test Set (Hemg Deepfake Audio)",
        "url": "https://huggingface.co/datasets/Hemg/Deepfake-Audio-Dataset/resolve/main/data/train-00000-of-00001-ab2dff7d513c15ff.parquet",
        "dest": os.path.join(DATA_DIR, "audio", "deepfake_audio_test.parquet"),
    },
    # Real & Deepfake MP4 Video Benchmark Samples
    *[
        {
            "name": f"Real Video Test Sample {i}",
            "url": f"https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset/resolve/main/Real/v{i}.mp4",
            "dest": os.path.join(DATA_DIR, "video", f"real_v{i}.mp4"),
        }
        for i in [1, 2, 3, 4, 5]
    ],
    *[
        {
            "name": f"Deepfake Video Test Sample {i}",
            "url": f"https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset/resolve/main/Fake/vs{i}.mp4",
            "dest": os.path.join(DATA_DIR, "video", f"fake_vs{i}.mp4"),
        }
        for i in [1, 2, 3, 4, 5]
    ],
]


def download_file(url: str, dest_path: str, name: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"✓ Already present: {name} ({size_mb:.2f} MB)")
        return

    print(f"\n⬇ Downloading {name}...")
    print(f"  Source: {url}")
    print(f"  Target: {dest_path}")

    start_time = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()

    duration = time.time() - start_time
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"\n  ✓ Download complete in {duration:.1f}s ({size_mb:.2f} MB)")


def main():
    print("=" * 70)
    print("THE TURING PROJECT: DOWNLOADING NEW OUT-OF-DISTRIBUTION TEST SETS")
    print("=" * 70)
    for ds in DATASETS:
        download_file(ds["url"], ds["dest"], ds["name"])

    print("\n" + "=" * 70)
    print("All test evaluation datasets ready in data/test_benchmark/!")
    print("=" * 70)


if __name__ == "__main__":
    main()
