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

        # Encode each frame to feature vector
        # Conv2d arguments: (in_channels, out_channels, kernel_size, stride, padding)
        # Output size = (Input size - kernel_size + 2*padding) / stride + 1
        #
        # Size transformations for 320x240x3 input:
        # Input:  320x240x3
        # Conv1:  (320-7+2*3)/2+1 = 160, (240-7+2*3)/2+1 = 120  -> 160x120x64
        # Conv2:  (160-3+2*1)/2+1 = 80,  (120-3+2*1)/2+1 = 60   -> 80x60x128
        # Conv3:  (80-3+2*1)/2+1 = 40,   (60-3+2*1)/2+1 = 30    -> 40x30x256
        # AdaptiveAvgPool2d: 40x30x256 -> 1x1x256
        # Flatten + Linear: 256 -> feature_dim (e.g., 256)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(img_c, 64, 7, stride=2, padding=3),  # 320x240x3 -> 160x120x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),     # 160x120x64 -> 80x60x128
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),    # 80x60x128 -> 40x30x256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                   # 40x30x256 -> 1x1x256
            nn.Flatten(),                                    # 1x1x256 -> 256
            nn.Linear(256, self.feature_dim)                # 256 -> feature_dim
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

        # Decode features back to images
        self.frame_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_c * img_h * img_w),
            nn.Sigmoid()  # Output in [0, 1] range
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

        # Use last frame's features to predict future
        last_feature = temporal_features[:, -1, :]  # [batch, feature_dim]

        # Predict future frame features
        future_features = self.frame_predictor(last_feature)  # [batch, feature_dim * output_frames]
        future_features = future_features.reshape(batch_size * self.output_frames, self.feature_dim)

        # Decode to images
        output = self.frame_decoder(future_features)  # [batch * output_frames, C*H*W]
        output = output.reshape(batch_size, self.output_frames, self.img_c, self.img_h, self.img_w)

        return output
