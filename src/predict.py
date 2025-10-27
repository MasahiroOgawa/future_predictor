import torch
import yaml
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import argparse

from models.transformer_predictor import FramePredictor


def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def override_config(config, args):
    """Override config values with command-line arguments."""
    if args.input_video is not None:
        config['paths']['input_video'] = args.input_video
    if args.output_dir is not None:
        config['paths']['output_dir'] = args.output_dir
    if args.model_path is not None:
        config['paths']['best_model'] = args.model_path
    if args.device is not None:
        config['device'] = args.device
    return config


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
    Predict all frames from the video using autoregressive sliding window.

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
        # Start with initial real frames
        current_sequence = all_frames_tensor[:input_frames_count].clone()  # [input_frames_count, C, H, W]

        # Predict for each position using sliding window with previous predictions
        for i in range(total_frames - input_frames_count):
            # Use current sequence (mix of real and predicted frames)
            input_batch = current_sequence.unsqueeze(0).to(device)  # [1, input_frames_count, C, H, W]

            # Predict next frame
            output = model(input_batch)  # [1, 1, C, H, W]
            predicted_frame = output[0, 0]  # [C, H, W]

            # Save predicted frame
            predicted_frames.append(tensor_to_frame(predicted_frame))
            print(f"Predicted frame {i + input_frames_count}/{total_frames}")

            # Update sliding window: remove oldest frame, add predicted frame
            current_sequence = torch.cat([
                current_sequence[1:],  # Remove first frame
                predicted_frame.unsqueeze(0).cpu()  # Add predicted frame
            ], dim=0)

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict future video frames")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--input_video", type=str, help="Input video path")
    parser.add_argument("--output_dir", type=str, help="Output directory for predictions")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()

    # Load configuration and override with arguments
    config = load_config(args.config)
    config = override_config(config, args)

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
