# Data Science Components

This directory contains the machine learning and deep learning models for stroke risk prediction.

## Structure

### ECG Deep Learning (`ECG_deep_learning/`)
- ECG signal processing models
- Deep learning architectures for ECG analysis
- Training and evaluation scripts
- Model optimization code

### Vital Signs Machine Learning (`VS_machine_learning/`)
- Models for vital signs analysis
- Feature engineering pipelines
- Model training scripts
- Performance evaluation tools

## Key Features

1. **ECG Analysis**
   - Signal preprocessing
   - Feature extraction
   - Deep learning model architectures
   - Model training and validation

2. **Vital Signs Analysis**
   - Data preprocessing
   - Feature selection
   - Model training
   - Performance metrics

## Usage

These models are used by the model inference system for real-time stroke risk prediction. The trained models are saved and deployed through the `model_inferencing` package.

## Development

When developing new models:
1. Follow the established directory structure
2. Document model architectures and parameters
3. Include evaluation metrics and results
4. Provide clear instructions for model training and deployment 