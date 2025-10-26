import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml
from pathlib import Path
from tqdm import tqdm

from models.transformer_predictor import FramePredictor
from data.dataset import VideoFrameDataset


def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for input_frames, target_frames in tqdm(dataloader, desc="Training"):
        input_frames = input_frames.to(device)
        target_frames = target_frames.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(input_frames)

        # Calculate loss
        loss = criterion(output, target_frames)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_frames, target_frames in tqdm(dataloader, desc="Validation"):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            # Forward pass
            output = model(input_frames)

            # Calculate loss
            loss = criterion(output, target_frames)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    print("Loading dataset...")
    dataset = VideoFrameDataset(
        frame_dir=config['data']['output_dir'],
        input_frames=config['model']['input_frames'],
        output_frames=config['model']['output_frames']
    )

    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Create model
    print("Creating model...")
    model = FramePredictor(config).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create checkpoint directory
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, config['paths']['best_model'])
            print(f"Saved best model with val_loss: {val_loss:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
