import torch
import torch.nn as nn


class FramePredictor(nn.Module):
    """
    Transformer-based frame prediction model with residual learning.

    The model predicts the RESIDUAL (change) from the last input frame,
    not the absolute output. This makes learning easier because:
    - Consecutive video frames are similar
    - Learning "what changes" is easier than "what the frame looks like"
    - Initial output is the last input frame (identity mapping)
    """

    def __init__(self, config):
        super().__init__()

        self.input_frames = config['model']['input_frames']
        self.output_frames = config['model']['output_frames']
        self.feature_dim = config['model']['feature_dim']
        img_h = config['image']['height']
        img_w = config['image']['width']
        img_c = config['image']['channels']

        # Compute channel sizes based on image dimensions with minimum values
        # to ensure sufficient model capacity even for small images
        ch1 = max(32, img_w // 5)
        ch2 = max(64, img_w // 2)
        ch3 = max(128, img_w)
        self.bottleneck_ch = ch3

        # Calculate spatial dimensions after encoder (each conv halves the size)
        self.spatial_h = img_h // 8
        self.spatial_w = img_w // 8

        # Encode each frame to feature vector
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(img_c, ch1, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch2, ch3, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch3, self.feature_dim)
        )

        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['num_layers']
        )

        # Residual predictor: predicts the change for each output frame
        # Input: temporal features, Output: residual features for each frame
        self.residual_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim * self.output_frames)
        )

        # Decode features back to images (symmetric to encoder)
        self.decoder_fc = nn.Linear(self.feature_dim, ch3)

        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(ch3, ch3, kernel_size=(self.spatial_h, self.spatial_w)),
            nn.ReLU(),
            nn.ConvTranspose2d(ch3, ch2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch2, ch1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ch1, img_c, 7, stride=2, padding=3, output_padding=1),
            nn.Tanh()  # Output in [-1, 1] range for residual
        )

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c

        # Initialize residual path to output zeros initially
        # This makes the initial output equal to the last input frame
        self._init_residual_predictor_zero()

    def _init_residual_predictor_zero(self):
        """Initialize the last layer of residual predictor to output zeros."""
        # Set the last linear layer's weights and bias to zero
        # This makes initial residual = 0, so output = last input frame
        last_layer = self.residual_predictor[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        # Also initialize decoder's last conv layer to output zeros
        # This ensures the entire residual path outputs zero initially
        last_conv = self.frame_decoder[-2]  # Last ConvTranspose2d before Tanh
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, x, return_residual=False):
        """
        Forward pass for frame prediction with residual learning.

        Args:
            x: Input frames [batch, num_frames, C, H, W]
            return_residual: If True, also return the raw residual for debugging

        Returns:
            Predicted frames [batch, output_frames, C, H, W]
            If return_residual=True, returns (output, residual) tuple
        """
        batch_size = x.shape[0]
        num_frames = x.shape[1]

        # Store last input frame as base for residual
        last_input_frame = x[:, -1]  # [batch, C, H, W]

        # Encode each frame independently using CNN
        x_reshaped = x.reshape(batch_size * num_frames, self.img_c, self.img_h, self.img_w)
        features = self.frame_encoder(x_reshaped)  # [batch * num_frames, feature_dim]

        # Reshape back to sequences
        features = features.reshape(batch_size, num_frames, self.feature_dim)

        # Apply transformer to learn temporal patterns
        temporal_features = self.transformer(features)  # [batch, num_frames, feature_dim]

        # Use last frame's temporal feature (contains context from all frames)
        last_temporal_feature = temporal_features[:, -1]  # [batch, feature_dim]

        # Predict residual features for each output frame
        residual_features = self.residual_predictor(last_temporal_feature)  # [batch, feature_dim * output_frames]
        residual_features = residual_features.reshape(batch_size * self.output_frames, self.feature_dim)

        # Decode residual to images
        decoded = self.decoder_fc(residual_features)
        decoded = decoded.view(-1, self.bottleneck_ch, 1, 1)
        residual = self.frame_decoder(decoded)  # [batch * output_frames, C, H, W]
        residual = residual.reshape(batch_size, self.output_frames, self.img_c, self.img_h, self.img_w)

        # Add residual to last input frame (broadcast across output_frames)
        # last_input_frame: [batch, C, H, W] -> [batch, 1, C, H, W]
        base_frame = last_input_frame.unsqueeze(1)

        # Output = base + residual, clamped to [0, 1]
        output = base_frame + residual
        output = output.clamp(0, 1)

        if return_residual:
            return output, residual
        return output
