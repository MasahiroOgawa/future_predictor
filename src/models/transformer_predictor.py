import torch
import torch.nn as nn


class FramePredictor(nn.Module):
    """
    Transformer-based frame prediction model.
    Predicts future frames based on sequence of previous frames.
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
        ch1 = max(32, img_w // 5)   # e.g., 320 -> 64, 32 -> 32 (min)
        ch2 = max(64, img_w // 2)   # e.g., 320 -> 160, 32 -> 64 (min)
        ch3 = max(128, img_w)       # e.g., 320 -> 320, 32 -> 128 (min)
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

        # Predict features for output frames
        self.frame_predictor = nn.Linear(self.feature_dim, self.feature_dim * self.output_frames)

        # Decode features back to images (symmetric to encoder)
        self.decoder_fc = nn.Linear(self.feature_dim, ch3)

        self.frame_decoder = nn.Sequential(
            # 1x1xch3 -> spatial_h x spatial_w x ch3
            nn.ConvTranspose2d(ch3, ch3, kernel_size=(self.spatial_h, self.spatial_w)),
            nn.ReLU(),
            # -> 2x spatial size, ch2 channels
            nn.ConvTranspose2d(ch3, ch2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # -> 4x spatial size, ch1 channels
            nn.ConvTranspose2d(ch2, ch1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # -> 8x spatial size = original, img_c channels
            nn.ConvTranspose2d(ch1, img_c, 7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c

    def forward(self, x):
        """
        Forward pass for frame prediction.

        Args:
            x: Input frames [batch, num_frames, C, H, W]
               batch: number of training samples (can be from different videos)
               num_frames: number of input frames per sample

               Example with batch_size=8, input_frames=5:
               - Sample 1: frames [0,1,2,3,4] from video_A
               - Sample 2: frames [1,2,3,4,5] from video_A
               - Sample 3: frames [0,1,2,3,4] from video_B
               - ... (8 samples total, possibly from multiple videos)

        Returns:
            Predicted frames [batch, output_frames, C, H, W]
        """
        batch_size = x.shape[0]
        num_frames = x.shape[1]

        # Encode each frame independently using CNN
        #
        # Why reshape? CNN expects 4D input [batch, C, H, W] but we have 5D [batch, frames, C, H, W]
        # Solution: Treat all frames from all samples as independent images
        #
        # Example: batch_size=8, input_frames=5
        # Before: [8, 5, 3, 240, 320] = 8 samples, each with 5 frames
        # After:  [40, 3, 240, 320] = 40 independent images (8 samples × 5 frames)
        x = x.reshape(batch_size * num_frames, self.img_c, self.img_h, self.img_w)
        features = self.frame_encoder(x)  # [batch * num_frames, feature_dim]

        # Reshape back to group features by their original sequences
        # This is necessary for the transformer to understand temporal relationships
        # [40, feature_dim] -> [8, 5, feature_dim] = 8 sequences of 5 features
        features = features.reshape(batch_size, num_frames, self.feature_dim)

        # Apply transformer to learn temporal patterns
        # Transformer uses self-attention to relate frames within each sequence
        temporal_features = self.transformer(features)  # [batch, num_frames, feature_dim]

        # Use all temporal features (pooled) to predict future
        # Mean pooling across all frames to capture full temporal context
        pooled_features = temporal_features.mean(dim=1)  # [batch, feature_dim]

        # Predict future frame features
        future_features = self.frame_predictor(pooled_features)  # [batch, feature_dim * output_frames]
        future_features = future_features.reshape(batch_size * self.output_frames, self.feature_dim)

        # Decode to images
        decoded = self.decoder_fc(future_features)
        decoded = decoded.view(-1, self.bottleneck_ch, 1, 1)
        output = self.frame_decoder(decoded)
        output = output.reshape(batch_size, self.output_frames, self.img_c, self.img_h, self.img_w)

        return output
