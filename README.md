# Future Predictor

A video frame prediction network using transformer-based architecture. This model predicts future video frames based on a sequence of previous frames, focusing on learning temporal dynamics and motion changes.

## Setup

### Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast package management, but also supports standard `pip`.

#### 1. Clone the repository

```bash
git clone <repository-url>
cd future_predictor
```

#### 2. Create Virtual Environment & Install Dependencies

**Using `uv` (Recommended):**

```bash
# Create virtual environment
uv venv

# Activate on Linux/Mac
source .venv/bin/activate
# OR Activate on Windows
# .venv\Scripts\activate

# Install dependencies (syncs from pyproject.toml / requirements)
uv sync
```

**Using `pip`:**

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/Mac
source .venv/bin/activate
# OR Activate on Windows
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR install in editable mode
pip install -e .
```

For development (includes testing tools):
```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

## Usage

### 1. Prepare Data
Place your training videos in `data/videos/`. Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`.

```bash
# Example structure
data/videos/
├── video1.mp4
└── video2.mov
```

### 2. Preprocess Videos
Extract frames from videos before training. This step resizes frames and saves them to `data/frames/`.

```bash
python src/preprocess.py
```

### 3. Training
Train the model. The script will automatically load frames, split data, and save checkpoints to `checkpoints/`.

```bash
# Default training
python src/train.py

# Custom arguments
python src/train.py --epochs 100 --batch_size 8 --device cuda
```

**Key Arguments:**
- `--config`: Path to config file (default: `config/config.yaml`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--device`: `cuda` or `cpu`

### 4. Inference
Predict future frames from a video.

```bash
# Predict using the best saved model
python src/predict.py --input_video data/videos/test.mp4 --output_dir outputs/

# Enable debug mode to see residual layers
python src/predict.py --input_video data/videos/test.mp4 --debug
```

## Architecture

For a detailed explanation of the model architecture, including the **Transformer-based Residual Prediction** pipeline and the **Composite Loss Function** strategy, please refer to [doc/architecture.md](doc/architecture.md).

### Core Components Summary

1.  **Frame Encoder (CNN)**: Compresses frames to feature vectors.
2.  **Transformer Temporal Encoder**: Learns temporal dynamics.
3.  **Residual Predictor**: Predicts motion changes ($\Delta$).
4.  **Composite Loss**: Combines MSE, Temporal, Gradient, and Perceptual losses for high-quality generation.

## Configuration

All hyperparameters are defined in `config/config.yaml`.

```yaml
model:
  input_frames: 10
  output_frames: 10
  predict_residual: true

loss:
  mse_weight: 1.0
  temporal_weight: 1.0
  gradient_weight: 1.0
  perceptual_weight: 0.1
```

## Troubleshooting

-   **OOM Errors**: Reduce `batch_size` or image dimensions in `config.yaml`.
-   **Static Predictions**: Increase `temporal_weight` in config or ensure `predict_residual` is enabled.
-   **"Frame directory empty"**: Run `python src/preprocess.py` first.
