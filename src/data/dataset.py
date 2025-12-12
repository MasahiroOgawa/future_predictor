from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset):
    """
    Dataset for sequential video frames.
    Returns a sequence of input frames and target output frames.
    """

    def __init__(
        self,
        frame_dir,
        input_frames=5,
        output_frames=1,
        transform=None,
        target_size=None,
    ):
        """
        Args:
            frame_dir: Directory containing extracted frames (organized by video)
            input_frames: Number of input frames in sequence
            output_frames: Number of output frames to predict
            transform: Optional transform to apply to frames
            target_size: Tuple of (width, height) to resize frames to
        """
        self.frame_dir = Path(frame_dir)
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.transform = transform
        self.target_size = target_size

        # Get all video directories
        self.video_dirs = sorted([d for d in self.frame_dir.iterdir() if d.is_dir()])

        # Build sequence indices
        self.sequences = []
        for video_dir in self.video_dirs:
            frames = sorted(list(video_dir.glob("*.png")))
            num_frames = len(frames)

            # Create sliding window sequences
            for i in range(num_frames - input_frames - output_frames + 1):
                input_seq = frames[i : i + input_frames]
                output_seq = frames[i + input_frames : i + input_frames + output_frames]
                self.sequences.append((input_seq, output_seq))

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, path):
        """Load and process a single frame."""
        img = Image.open(path).convert("RGB")
        if self.target_size:
            img = img.resize(self.target_size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img

    def __getitem__(self, idx):
        input_paths, output_paths = self.sequences[idx]

        # Load input frames
        input_frames = [self._load_frame(path) for path in input_paths]

        # Load output frames
        output_frames = [self._load_frame(path) for path in output_paths]

        # Stack frames: [num_frames, C, H, W]
        input_tensor = torch.stack(input_frames)
        output_tensor = torch.stack(output_frames)

        return input_tensor, output_tensor
