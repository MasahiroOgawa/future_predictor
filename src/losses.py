import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TemporalDifferenceLoss(nn.Module):
    """
    Penalizes incorrect temporal changes.
    Computes L1/L2 distance between (frame[t] - frame[t-1]) of prediction and target.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        # pred, target: [B, T, C, H, W]
        # Calculate diffs: frame[t] - frame[t-1]
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]

        loss = self.criterion(pred_diff, target_diff)
        return loss * self.weight


class GradientLoss(nn.Module):
    """
    Penalizes differences in spatial gradients (edges).
    Helps preserve sharpness and structure.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        # pred, target: [B, T, C, H, W]
        # Combine B and T dimensions
        b, t, c, h, w = pred.shape
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)

        # Calculate gradients in X and Y
        # dy: diff between row i and row i+1
        # dx: diff between col j and col j+1

        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss_dx = self.criterion(pred_dx, target_dx)
        loss_dy = self.criterion(pred_dy, target_dy)

        return (loss_dx + loss_dy) * self.weight


class PerceptualLoss(nn.Module):
    """
    Uses ImageNet-pretrained VGG16 to compute feature-level loss.
    """

    def __init__(self, weight=1.0, device="cpu"):
        super().__init__()
        self.weight = weight
        self.device = device

        # Load VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # We only need the first few layers for low-level features (edges, textures)
        # and some mid-level features.
        # Slice indices matching common perceptual loss implementations
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        # Relu1_2, Relu2_2, Relu3_3
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()
        self.to(device)

    def forward(self, pred, target):
        # pred, target: [B, T, C, H, W]
        # Reshape to [B*T, C, H, W]
        b, t, c, h, w = pred.shape
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)

        # Normalize if needed (VGG expects specific mean/std, but commonly simple 0-1 works ok for relative loss)
        # Proper way: (x - mean) / std. Assuming inputs are 0-1.
        # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        # Features
        h_pred = self.slice1(pred_norm)
        h_target = self.slice1(target_norm)
        loss = self.criterion(h_pred, h_target)

        h_pred = self.slice2(h_pred)
        h_target = self.slice2(h_target)
        loss += self.criterion(h_pred, h_target)

        h_pred = self.slice3(h_pred)
        h_target = self.slice3(h_target)
        loss += self.criterion(h_pred, h_target)

        return loss * self.weight


class CombinedLoss(nn.Module):
    """
    Combines MSE, Temporal Difference, Gradient, and Perceptual losses.
    """

    def __init__(self, config, device="cpu"):
        super().__init__()
        self.weights = config["loss"]

        # Base loss
        self.mse = nn.MSELoss()

        # Specialized losses
        self.temporal_loss = TemporalDifferenceLoss(
            weight=self.weights.get("temporal_weight", 1.0)
        )
        self.gradient_loss = GradientLoss(
            weight=self.weights.get("gradient_weight", 1.0)
        )

        if self.weights.get("perceptual_weight", 0.0) > 0:
            self.perceptual_loss = PerceptualLoss(
                weight=self.weights.get("perceptual_weight", 0.1), device=device
            )
        else:
            self.perceptual_loss = None

    def forward(self, pred, target):
        losses = {}

        # 1. MSE Loss (Pixel Consistency)
        mse_loss = self.mse(pred, target)
        losses["mse"] = mse_loss * self.weights.get("mse_weight", 1.0)

        # 2. Temporal Loss (Motion Consistency)
        # Comparing frame differences
        temp_loss = self.temporal_loss(pred, target)
        losses["temporal"] = temp_loss

        # 3. Gradient Loss (Spatial Structure)
        grad_loss = self.gradient_loss(pred, target)
        losses["gradient"] = grad_loss

        # 4. Perceptual Loss (High-level Features)
        perc_loss = 0
        if self.perceptual_loss:
            perc_loss = self.perceptual_loss(pred, target)
            losses["perceptual"] = perc_loss

        # Total
        total_loss = losses["mse"] + losses["temporal"] + losses["gradient"]
        if isinstance(perc_loss, torch.Tensor):
            total_loss += perc_loss

        losses["total"] = total_loss

        return total_loss, losses
