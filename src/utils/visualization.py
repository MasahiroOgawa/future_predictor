from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(tensor):
    """
    Convert tensor [C, H, W] in range [0, 1] to numpy array [H, W, C] in range [0, 255]
    """
    img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def save_prediction_sample(inputs, targets, preds, epoch, batch_idx, save_dir):
    """
    Save a grid of samples: Input Seq | Target Seq | Predicted Seq
    inputs: [B, T_in, C, H, W]
    targets: [B, T_out, C, H, W]
    preds: [B, T_out, C, H, W]
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Take first sample in batch
    input_seq = inputs[0]
    target_seq = targets[0]
    pred_seq = preds[0]

    num_in = input_seq.shape[0]
    num_out = target_seq.shape[0]

    # Create figure
    # Rows: Input, Target, Prediction, Difference (Target - Pred)
    # Cols: Time steps (max(num_in, num_out))

    max_frames = max(num_in, num_out)
    rows = 4
    cols = max_frames

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # 1. Inputs
    for t in range(cols):
        ax = axes[0, t]
        if t < num_in:
            img = denormalize(input_seq[t])
            ax.imshow(img)
            ax.set_title(f"In t={t}")
        else:
            ax.axis("off")
        ax.axis("off")

    # 2. Targets
    for t in range(cols):
        ax = axes[1, t]
        if t < num_out:
            img = denormalize(target_seq[t])
            ax.imshow(img)
            ax.set_title(f"Tgt t={t}")
        else:
            ax.axis("off")
        ax.axis("off")

    # 3. Predictions
    for t in range(cols):
        ax = axes[2, t]
        if t < num_out:
            img = denormalize(pred_seq[t])
            ax.imshow(img)
            ax.set_title(f"Pred t={t}")
        else:
            ax.axis("off")
        ax.axis("off")

    # 4. Difference (Heatmap)
    for t in range(cols):
        ax = axes[3, t]
        if t < num_out:
            # Basic difference
            tgt = denormalize(target_seq[t]).astype(float)
            prd = denormalize(pred_seq[t]).astype(float)
            diff = np.abs(tgt - prd).mean(axis=2)  # [H, W] intensity diff

            # Normalize diff to [0, 1] for visibility enhancement
            # or just show raw diff magnitude
            # Let's show raw diff but enhanced

            im = ax.imshow(
                diff, cmap="hot", vmin=0, vmax=50
            )  # Assuming 0-255 inputs, 50 is reasonable diff cap for vis
            # fig.colorbar(im, ax=ax)
            ax.set_title(f"Diff t={t}")
        else:
            ax.axis("off")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()


def save_motion_heatmap(targets, preds, epoch, batch_idx, save_dir):
    """
    Visualize flow/motion difference: |Frame[t] - Frame[t-1]|
    """
    # ... implementation can be expanded similarly if needed ...
    pass
