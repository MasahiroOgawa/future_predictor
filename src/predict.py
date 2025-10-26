import torch
import yaml
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from models.transformer_predictor import FramePredictor


def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path, config, device):
    """Load trained model from checkpoint."""
    model = FramePredictor(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_all_frames_from_video(video_path, target_size):
    """Load all frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def frames_to_tensor(frames):
    """Convert list of frames to tensor."""
    # frames: list of numpy arrays [H, W, C]
    # output: tensor [num_frames, C, H, W]
    tensor_frames = []
    for frame in frames:
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor_frames.append(frame_tensor)
    return torch.stack(tensor_frames)


def tensor_to_frame(tensor):
    """Convert tensor to numpy frame."""
    # tensor: [C, H, W]
    # output: numpy array [H, W, C]
    frame = tensor.permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    return frame


def predict_all_frames(model, all_frames_tensor, input_frames_count, device):
    """
    Predict all frames from the video using sliding window.

    Args:
        model: Trained model
        all_frames_tensor: All video frames as tensor [total_frames, C, H, W]
        input_frames_count: Number of input frames needed for prediction
        device: Device to run on

    Returns:
        List of predicted frames as numpy arrays
    """
    model.eval()
    predicted_frames = []
    total_frames = all_frames_tensor.shape[0]

    with torch.no_grad():
        # Predict for each possible position in the video
        for i in range(total_frames - input_frames_count):
            # Get input sequence: frames [i, i+1, ..., i+input_frames_count-1]
            input_sequence = all_frames_tensor[i:i + input_frames_count]
            input_batch = input_sequence.unsqueeze(0).to(device)  # [1, input_frames_count, C, H, W]

            # Predict next frame: frame [i+input_frames_count]
            output = model(input_batch)  # [1, 1, C, H, W]
            predicted_frame = output[0, 0]  # [C, H, W]

            predicted_frames.append(tensor_to_frame(predicted_frame))
            print(f"Predicted frame {i + input_frames_count}/{total_frames}")

    return predicted_frames


def save_frames(frames, output_dir, prefix="predicted"):
    """Save frames as images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        output_path = output_dir / f"{prefix}_{i:06d}.png"
        Image.fromarray(frame).save(output_path)

    print(f"Saved {len(frames)} frames to {output_dir}")


def main():
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(config['paths']['best_model'], config, device)

    # Load input video path from config
    video_path = config['paths']['input_video']
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        print("Please specify a valid video path in config/config.yaml (paths.input_video)")
        return

    # Load all frames from video
    print(f"Loading all frames from {video_path}...")
    target_size = (config['image']['width'], config['image']['height'])
    all_frames_numpy = load_all_frames_from_video(video_path, target_size)

    print(f"Loaded {len(all_frames_numpy)} frames from video")

    if len(all_frames_numpy) < config['model']['input_frames'] + 1:
        print(f"Not enough frames in video. Need at least {config['model']['input_frames'] + 1}, got {len(all_frames_numpy)}")
        return

    # Convert to tensor
    all_frames_tensor = frames_to_tensor(all_frames_numpy)

    # Predict all frames
    print("Predicting frames...")
    predicted_frames = predict_all_frames(
        model,
        all_frames_tensor,
        config['model']['input_frames'],
        device
    )

    # Save results
    print("Saving predictions...")
    save_frames(predicted_frames, config['paths']['output_dir'])

    print(f"\nPrediction complete! Saved {len(predicted_frames)} frames to {config['paths']['output_dir']}")


if __name__ == "__main__":
    main()
