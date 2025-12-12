"""
Preprocess videos by extracting frames before training.
"""

import argparse
from pathlib import Path

from utils.video_processor import load_config, process_videos


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "--video_dir", type=str, default=None, help="Directory containing videos"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save frames"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Use command line args or config values
    video_dir = args.video_dir or config["data"]["video_dir"]
    output_dir = args.output_dir or config["data"]["output_dir"]

    print(f"Processing videos from: {video_dir}")
    print(f"Saving frames to: {output_dir}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process all videos
    process_videos(video_dir, output_dir, config)
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
