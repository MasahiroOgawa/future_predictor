# future_predictor

Image future prediction network using transformer-based architecture.

## Overview

This project predicts future video frames based on a sequence of previous frames. Given N continuous frames as input, the model predicts the next M frames.

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

### Components:

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

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

All parameters are configured in `config/config.yaml`.

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/predict.py
```
