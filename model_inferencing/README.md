# Model Inference System

This directory contains the code for model deployment and real-time inference.

## Structure

- `rnn_attention_model.py`: RNN with attention mechanism implementation
- `calling_model.py`: Model inference and prediction interface
- `full_model.pkl`: Serialized trained model
- `memmap_head.npy`: Memory-mapped model parameters

## Key Components

1. **Model Architecture**
   - RNN with attention mechanism
   - Model loading and initialization
   - Inference pipeline

2. **Inference Interface**
   - Real-time prediction
   - Input preprocessing
   - Result formatting

## Usage

The inference system is designed to be called by the main application for real-time stroke risk prediction. It handles:
- Model loading and initialization
- Input data preprocessing
- Real-time inference
- Result formatting and delivery

## Performance Considerations

- Memory-mapped model parameters for efficient loading
- Optimized inference pipeline
- Batch processing capabilities

## Integration

This system integrates with:
- Main application server
- Data preprocessing pipeline
- Web interface for result display 