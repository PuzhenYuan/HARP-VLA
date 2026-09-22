"""Download the inference checkpoint to a simple local directory."""
import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="checkpoints/calvin")
    args = parser.parse_args()
    snapshot_download(repo_id="ypz21/HARP_VLA_calvin", local_dir=args.output_dir)
