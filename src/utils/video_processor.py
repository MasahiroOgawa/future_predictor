import os
from pathlib import Path

import cv2
import yaml


def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_frames(video_path, output_dir, target_size=(320, 240)):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        target_size: Tuple of (width, height) to resize frames

    Returns:
        List of saved frame paths
    """
    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, target_size)

        # Save frame
        frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_path}")
    return frame_paths


def process_videos(video_dir, output_dir, config):
    """
    Process all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        config: Configuration dictionary
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_dir = Path(video_dir)

    target_size = (config["image"]["width"], config["image"]["height"])

    for video_path in video_dir.glob("*"):
        if video_path.suffix.lower() in video_extensions:
            extract_frames(video_path, output_dir, target_size)
