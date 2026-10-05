#!/usr/bin/env python3
"""Fetch pinned public Qwen3-4B weights using resumable parallel HTTP ranges."""

import os
from pathlib import Path
import subprocess

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DIRECTORY = Path("/SSD/00/wja/GRACE/data/Qwen3-4B")


def main():
    snapshot_download("Qwen/Qwen3-4B", revision=REVISION, local_dir=DIRECTORY,
                      allow_patterns=["*.json", "*.jinja", "*.txt"], max_workers=4)
    jobs = []
    for index in range(1, 4):
        name = f"model-{index:05d}-of-00003.safetensors"
        url = hf_hub_url("Qwen/Qwen3-4B", name, revision=REVISION)
        metadata = get_hf_file_metadata(url)
        command = ["aria2c", "--continue=true", "--auto-file-renaming=false",
                   "--allow-overwrite=true", "--max-connection-per-server=16", "--split=16",
                   "--min-split-size=1M", "--max-tries=0", "--retry-wait=10", "--timeout=60",
                   "--connect-timeout=30", "--summary-interval=30", "--console-log-level=warn",
                   "--check-integrity=true", f"--checksum=sha-256={metadata.etag}",
                   f"--dir={DIRECTORY}", f"--out={name}", url + "?download=true"]
        print(f"download={name} expected_bytes={metadata.size} sha256={metadata.etag}", flush=True)
        jobs.append(subprocess.Popen(command))
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f"weight download failed: {codes}")
    (DIRECTORY / "thinking_download_verified.txt").write_text(REVISION + "\n")
    print("all pinned thinking model files verified", flush=True)


if __name__ == "__main__":
    main()
