# Future Predictor

A video frame prediction network using transformer-based architecture. This model predicts future video frames based on a sequence of previous frames.

## Overview

Given N continuous frames as input, the model learns temporal patterns and predicts the next M frames. This is useful for:
- Video prediction and generation
- Motion understanding
- Video compression and interpolation
- Anomaly detection in videos

## Architecture

The model uses a transformer-based architecture with the following pipeline:

```
Input frames
   ↓
Frame Encoder (CNN) - processes each frame independently to extract spatial features
   ↓
Frame features
   ↓
Transformer Encoder - learns temporal relationships between frames
   ↓
Temporal features
   ↓
Frame Predictor - predicts future frame features
   ↓
Future frame features
   ↓
Frame Decoder (MLP) - converts features back to pixels
   ↓
Output frames
```

### Components

1. **Frame Encoder (CNN)**: Converts each input image into a compact feature vector
   - Uses strided convolutions to progressively reduce spatial dimensions
   - Processes each frame independently

2. **Transformer Encoder**: Takes the sequence of frame features
   - Learns temporal patterns (motion, changes between frames)
   - Uses self-attention to model relationships between frames

3. **Frame Predictor**: Predicts future frame features from temporal features
   - Linear projection layer

4. **Frame Decoder (MLP)**: Converts predicted features back to images
   - Multi-layer perceptron that reconstructs full image from features

## Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended) or CPU
- uv package manager (or pip)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd future_predictor
```

### 2. Set up Python environment

Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

Using standard `venv`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

Using `uv` (recommended):
```bash
uv sync
```

Using `pip`:
```bash
pip install -e .
```

For development (includes testing tools):
```bash
uv sync --extra dev
# or with pip
pip install -e ".[dev]"
```

## Project Structure

```
future_predictor/
├── config/
│   └── config.yaml          # All hyperparameters and settings
├── src/
│   ├── models/
│   │   └── transformer_predictor.py  # Model architecture
│   ├── data/
│   │   └── dataset.py       # Dataset classes
│   ├── utils/
│   │   └── video_processor.py  # Video processing utilities
│   ├── preprocess.py        # Frame extraction script
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── data/
│   ├── videos/              # Training videos (place your videos here)
│   └── frames/              # Extracted frames (auto-generated)
├── checkpoints/             # Saved model weights
└── outputs/                 # Prediction outputs
```

## Quick Start

### 1. Prepare Your Data

Place your training videos in the `data/videos/` directory. Supported formats:
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

```bash
# Example:
data/videos/
├── video1.mp4
├── video2.mov
└── video3.avi
```

### 2. Extract Frames from Videos

Before training, you need to extract frames from your videos:

```bash
python src/preprocess.py
```

This will:
- Read all videos from `data/videos/`
- Extract frames and resize them according to `config.yaml`
- Save frames to `data/frames/` organized by video name

**Example output:**
```
Processing videos from: data/videos
Saving frames to: data/frames
Extracted 49 frames from data/videos/mousemove.MOV
Preprocessing complete!
```

You can also specify custom directories:
```bash
python src/preprocess.py --video_dir path/to/videos --output_dir path/to/frames
```

### 3. Train the Model

Start training with default config settings:

```bash
python src/train.py
```

Or override specific parameters:

```bash
# Override training parameters
python src/train.py --epochs 50 --batch_size 4 --learning_rate 0.0001

# Override model parameters
python src/train.py --input_frames 10 --output_frames 2

# Use CPU instead of GPU
python src/train.py --device cpu

# Use a different config file
python src/train.py --config path/to/custom_config.yaml
```

**Available arguments:**
- `--config`: Path to config file (default: `config/config.yaml`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--input_frames`: Number of input frames
- `--output_frames`: Number of output frames to predict
- `--device`: Device to use (`cuda` or `cpu`)

The script will:
- Load frames from `data/frames/`
- Split into training and validation sets (80/20 by default)
- Train the model for the specified number of epochs
- Save checkpoints periodically
- Save the best model based on validation loss

**Training output example:**
```
Using device: cuda
Loading dataset...
Train samples: 35, Val samples: 9
Creating model...

Epoch 1/100
Training: 100%|████████| 5/5 [00:02<00:00, 2.34it/s]
Train Loss: 0.024531
Validation: 100%|████████| 2/2 [00:00<00:00, 8.12it/s]
Val Loss: 0.019234
Saved best model with val_loss: 0.019234
```

**Saved files:**
- `checkpoints/best_model.pth` - Best model (lowest validation loss)
- `checkpoints/checkpoint_epoch_10.pth` - Periodic checkpoints (every 10 epochs by default)

### 4. Run Inference

After training, predict future frames from a video using default config settings:

```bash
python src/predict.py
```

Or override specific parameters:

```bash
# Specify input video and output directory
python src/predict.py --input_video data/videos/test.mp4 --output_dir results/

# Use a specific model checkpoint
python src/predict.py --model_path checkpoints/checkpoint_epoch_50.pth

# Use CPU for inference
python src/predict.py --device cpu

# Combine multiple overrides
python src/predict.py --input_video test.mp4 --output_dir predictions/ --device cpu
```

**Available arguments:**
- `--config`: Path to config file (default: `config/config.yaml`)
- `--input_video`: Input video path
- `--output_dir`: Output directory for predictions
- `--model_path`: Path to trained model checkpoint
- `--device`: Device to use (`cuda` or `cpu`)

The script will:
- Load the trained model from `checkpoints/best_model.pth` (or specified path)
- Read the input video frame by frame
- Use a sliding window to predict future frames
- Save predictions to `outputs/` directory (or specified directory)

**Prediction output example:**
```
Using device: cuda
Loading model...
Loading all frames from data/videos/mousemove.MOV...
Loaded 49 frames from video
Predicting frames...
Predicted frame 5/49
Predicted frame 6/49
...
Saving predictions...
Saved 44 frames to outputs

Prediction complete!
```

**Output files:**
```
outputs/
├── predicted_000000.png
├── predicted_000001.png
├── predicted_000002.png
└── ...
```

## Configuration

All parameters are in `config/config.yaml`. The file is well-documented with inline comments explaining each setting.

## Tips for Better Results

1. **More data is better**: Use multiple videos with diverse motion patterns
2. **Appropriate frame count**: For quick tests, use 5 input frames and 1 output frame
3. **Image size**: Start with 320x240 for faster training, increase for better quality
4. **Monitor validation loss**: Training should show decreasing loss over epochs
5. **GPU recommended**: Training on CPU is very slow; use CUDA if available

## Troubleshooting

### Error: "Frame directory doesn't exist"
```
Error: Frame directory 'data/frames' is empty or doesn't exist!
```
**Solution**: Run preprocessing first: `python src/preprocess.py`

### Error: "Video not found"
```
Video not found: data/videos/test_video.mp4
```
**Solution**: Update the `input_video` path in `config/config.yaml` to point to an existing video

### Error: "Not enough frames"
```
Not enough frames in video. Need at least 6, got 3
```
**Solution**: Use a longer video, or reduce `input_frames` + `output_frames` in config

### Out of memory errors
**Solution**: Reduce `batch_size` or image dimensions (`width`/`height`) in `config/config.yaml`

## Development

### Running tests
```bash
pytest
```

### Code formatting
```bash
black src/
isort src/
```

## License

See [LICENSE](LICENSE) file for details.
