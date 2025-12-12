import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset import VideoFrameDataset
from losses import CombinedLoss
from models.transformer_predictor import FramePredictor
from utils.visualization import save_prediction_sample


def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def override_config(config, args):
    """Override config values with command-line arguments."""
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.input_frames is not None:
        config["model"]["input_frames"] = args.input_frames
    if args.output_frames is not None:
        config["model"]["output_frames"] = args.output_frames
    if args.device is not None:
        config["device"] = args.device
    return config


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    # Track individual losses
    loss_components_sum = {}

    pbar = tqdm(dataloader, desc="Training")
    for i, (input_frames, target_frames) in enumerate(pbar):
        input_frames = input_frames.to(device)
        target_frames = target_frames.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_frames)

        # Calculate loss
        loss, components = criterion(output, target_frames)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        # Aggregate components for logging
        for k, v in components.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            loss_components_sum[k] = loss_components_sum.get(k, 0) + v

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components_sum.items()}

    return avg_loss, avg_components


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    loss_components_sum = {}

    with torch.no_grad():
        for input_frames, target_frames in tqdm(dataloader, desc="Validation"):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            # Forward pass
            output = model(input_frames)

            # Calculate loss
            loss, components = criterion(output, target_frames)

            total_loss += loss.item()

            # Aggregate components
            for k, v in components.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                loss_components_sum[k] = loss_components_sum.get(k, 0) + v

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components_sum.items()}

    return avg_loss, avg_components


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train video frame prediction model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--input_frames", type=int, help="Number of input frames")
    parser.add_argument("--output_frames", type=int, help="Number of output frames")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to use"
    )
    args = parser.parse_args()

    # Load configuration and override with arguments
    config = load_config(args.config)
    config = override_config(config, args)

    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if frames directory exists
    frame_dir = Path(config["data"]["output_dir"])
    if not frame_dir.exists() or not any(frame_dir.iterdir()):
        print(f"\nError: Frame directory '{frame_dir}' is empty or doesn't exist!")
        print("Please run preprocessing first:")
        print("  python src/preprocess.py")
        print("\nThis will extract frames from videos in the video directory.")
        return

    # Create dataset
    print("Loading dataset...")
    target_size = (config["image"]["width"], config["image"]["height"])
    dataset = VideoFrameDataset(
        frame_dir=config["data"]["output_dir"],
        input_frames=config["model"]["input_frames"],
        output_frames=config["model"]["output_frames"],
        target_size=target_size,
    )

    # Split dataset
    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    # Create model
    print("Creating model...")
    model = FramePredictor(config).to(device)

    # Loss and optimizer
    # Modified to use CompositeLoss
    print("Initializing CombinedLoss...")
    criterion = CombinedLoss(config, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
    )

    # Create checkpoint directory
    Path(config["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Format components for printing
        comp_str = ", ".join([f"{k}: {v:.6f}" for k, v in train_components.items()])
        print(f"Train Loss: {train_loss:.6f} ({comp_str})")

        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)

        val_comp_str = ", ".join([f"{k}: {v:.6f}" for k, v in val_components.items()])
        print(f"Val Loss: {val_loss:.6f} ({val_comp_str})")

        # Step scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                config["paths"]["best_model"],
            )
            print(f"Saved best model with val_loss: {val_loss:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            checkpoint_path = (
                Path(config["paths"]["checkpoint_dir"])
                / f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
