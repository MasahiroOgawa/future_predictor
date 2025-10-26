# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

future_predictor is a video frame prediction network that predicts future frames based on a sequence of previous frames. The model takes continuous previous frames as input and outputs n future frames.

## Architecture

- **Input**: Sequence of continuous video frames
- **Output**: N predicted future frames
- **Base Model**: Transformer-based architecture for temporal prediction
- **Training Data**: Video files that are processed into sequential frame datasets

## Project Structure

```
future_predictor/
├── config/
│   └── config.yaml      # All hyperparameters and settings
├── src/
│   ├── models/          # Neural network architectures
│   ├── data/            # Dataset classes and data loaders
│   ├── utils/           # Utility functions (video processing, etc.)
│   ├── train.py         # Training script
│   └── predict.py       # Inference script
├── data/                # Training videos and processed frames
├── checkpoints/         # Saved model weights
└── outputs/             # Prediction outputs
```

## Development Commands

### Environment Setup
```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
uv pip install -e .
```

### Training
```bash
python src/train.py
```

All training parameters are configured in `config/config.yaml`.

### Inference
```bash
python src/predict.py
```

All inference parameters are configured in `config/config.yaml`.

## Configuration

All parameters are centralized in `config/config.yaml`:
- Image dimensions (default: 320x240 for small tests)
- Number of input/output frames
- Model hyperparameters
- Training settings (batch size, learning rate, epochs)
- Data paths

## Key Components

- **Video Processing**: Videos are split into sequential frames for training
- **Sequential Dataset**: DataLoader handles sliding window of frames (input sequence → output frames)
- **Transformer Model**: Uses transformer architecture to learn temporal patterns and predict future frames
