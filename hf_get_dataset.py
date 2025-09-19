import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="The repository ID of the dataset to download.")
    parser.add_argument("--local_dir", type=str, default=".", help="The local directory to download the dataset to.")
    args = parser.parse_args()

    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=args.local_dir)

if __name__ == "__main__":
    main()

