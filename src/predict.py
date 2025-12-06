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


def infer_checkpoint_image_size(checkpoint):
    """Infer the image size used during training from checkpoint weights."""
    state_dict = checkpoint['model_state_dict']

    # Get decoder's first ConvTranspose2d weight shape: [in_ch, out_ch, H, W]
    decoder_weight = state_dict['frame_decoder.0.weight']
    spatial_h = decoder_weight.shape[2]
    spatial_w = decoder_weight.shape[3]

    # Original image size = spatial size * 8 (due to 3 stride-2 convolutions)
    return spatial_h * 8, spatial_w * 8


def load_model(model_path, config, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if checkpoint was trained with different image dimensions
    ckpt_h, ckpt_w = infer_checkpoint_image_size(checkpoint)
    cfg_h, cfg_w = config['image']['height'], config['image']['width']

    if ckpt_h != cfg_h or ckpt_w != cfg_w:
        print(f"\nError: Image dimension mismatch!")
        print(f"  Checkpoint was trained on: {ckpt_w}x{ckpt_h} images")
        print(f"  Current config expects:    {cfg_w}x{cfg_h} images")
        print(f"\nTo fix this, either:")
        print(f"  1. Update config/config.yaml to use width: {ckpt_w}, height: {ckpt_h}")
        print(f"  2. Use a checkpoint trained with {cfg_w}x{cfg_h} images")
        raise SystemExit(1)

    model = FramePredictor(config).to(device)
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


def predict_future_frames(model, all_frames_tensor, input_frames_count, output_frames_count, device, save_debug=False, output_dir=None):
    """
    Predict future frames using the last N frames from the input video.

    Args:
        model: Trained model
        all_frames_tensor: All video frames as tensor [total_frames, C, H, W]
        input_frames_count: Number of input frames needed for prediction
        output_frames_count: Number of future frames to predict
        device: Device to run on
        save_debug: If True, save intermediate residual images for debugging
        output_dir: Output directory for debug images

    Returns:
        List of predicted frames as numpy arrays
    """
    model.eval()
    predicted_frames = []
    residual_frames = []

    with torch.no_grad():
        # Use the last input_frames_count frames from the video as context
        input_sequence = all_frames_tensor[-input_frames_count:]  # [input_frames_count, C, H, W]
        input_batch = input_sequence.unsqueeze(0).to(device)  # [1, input_frames_count, C, H, W]

        # Predict future frames (with residual for debugging)
        output, residual = model(input_batch, return_residual=True)  # [1, output_frames_count, C, H, W]

        # Print residual statistics for debugging
        print(f"\n=== Residual Debug Info ===")
        print(f"Residual shape: {residual.shape}")
        print(f"Residual min: {residual.min().item():.6f}")
        print(f"Residual max: {residual.max().item():.6f}")
        print(f"Residual mean: {residual.mean().item():.6f}")
        print(f"Residual std: {residual.std().item():.6f}")
        print(f"Residual abs mean: {residual.abs().mean().item():.6f}")
        print(f"===========================\n")

        # Convert all predicted frames to numpy
        for i in range(output_frames_count):
            predicted_frame = output[0, i]  # [C, H, W]
            predicted_frames.append(tensor_to_frame(predicted_frame))
            print(f"Predicted future frame {i + 1}/{output_frames_count}")

        # Save debug images if requested
        if save_debug and output_dir:
            debug_dir = Path(output_dir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Save the base frame (last input frame)
            base_frame = input_batch[0, -1]  # [C, H, W]
            base_frame_np = tensor_to_frame(base_frame)
            Image.fromarray(base_frame_np).save(debug_dir / "base_frame.png")
            print(f"Saved base frame to {debug_dir / 'base_frame.png'}")

            for i in range(output_frames_count):
                residual_frame = residual[0, i]  # [C, H, W]

                # Normalize residual to [0, 1] for visualization
                # Residual is in [-1, 1] from Tanh, map to [0, 1]
                residual_vis = (residual_frame + 1) / 2  # Map [-1, 1] to [0, 1]
                residual_vis_np = tensor_to_frame(residual_vis)

                # Also save the absolute residual (amplified for visibility)
                residual_abs = residual_frame.abs()
                residual_abs_amplified = (residual_abs * 10).clamp(0, 1)  # Amplify by 10x
                residual_abs_np = tensor_to_frame(residual_abs_amplified)

                # Save residual images
                Image.fromarray(residual_vis_np).save(debug_dir / f"residual_{i:06d}.png")
                Image.fromarray(residual_abs_np).save(debug_dir / f"residual_abs_{i:06d}.png")

                # Print per-frame residual stats
                print(f"Frame {i}: residual mean={residual_frame.mean().item():.6f}, "
                      f"std={residual_frame.std().item():.6f}, "
                      f"abs_max={residual_frame.abs().max().item():.6f}")

            print(f"\nSaved {output_frames_count} residual debug images to {debug_dir}")

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
    parser.add_argument("--debug", action="store_true", help="Save intermediate residual images for debugging")
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

    # Predict future frames using the last frames of the video
    print(f"Using last {config['model']['input_frames']} frames to predict {config['model']['output_frames']} future frame(s)...")
    predicted_frames = predict_future_frames(
        model,
        all_frames_tensor,
        config['model']['input_frames'],
        config['model']['output_frames'],
        device,
        save_debug=args.debug,
        output_dir=config['paths']['output_dir']
    )

    # Save results
    print("Saving predictions...")
    save_frames(predicted_frames, config['paths']['output_dir'], prefix="future")

    print(f"\nPrediction complete! Saved {len(predicted_frames)} future frame(s) to {config['paths']['output_dir']}")


if __name__ == "__main__":
    main()
