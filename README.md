# Deepfake Audio Detection

This repository contains the code and deployment details for a deepfake audio detection model. The model, based on SENet, is trained to classify audio files as real or fake using Mel-spectrograms. It has been deployed on Hugging Face Spaces with an interactive demo using Gradio.

## Features
- **Deepfake Detection**: Classifies audio as real or fake with high accuracy.
- **SENet Model**: Uses a pre-trained SENet architecture fine-tuned for binary classification.
- **Mel-Spectrogram Processing**: Converts audio into spectrograms for robust feature extraction.
- **Gradio Interface**: Provides a user-friendly web app for easy testing.
- **Hugging Face Deployment**: Hosted on Hugging Face Spaces for seamless access.

## Model Details
The models are loaded from Hugging Face Hub:

- **Repository**: SanthoshPrabhu/Deepfake_audio_detection
- **Model Files**:
  - `senet_model_best_model.pth`
  - `resnet_model_best_model.pth`
  - `res2net_model_best_model.pth`
  - `gnn_model_best_model.pth`
  - `darts_model_best_model.pth`
  - `lstm_model_best_model.pth`

### Model Architectures:
- **SENet**:
  - **Architecture**: SENet154
  - **Description**: A pre-trained SENet model fine-tuned for binary classification.
  - **Implementation**: `SENetClassifier`
- **ResNet**:
  - **Architecture**: ResNet18
  - **Description**: A pre-trained ResNet model modified for binary classification.
  - **Implementation**: `ResNetClassifier`
- **Res2Net**:
  - **Architecture**: Res2Net50
  - **Description**: A pre-trained Res2Net model adapted for binary classification.
  - **Implementation**: `Res2NetClassifier`
- **GNN**:
  - **Architecture**: Graph Neural Network
  - **Description**: A GNN model designed for audio classification using graph-based features.
  - **Implementation**: `GNNClassifier`
- **DARTS**:
  - **Architecture**: Differentiable Architecture Search
  - **Description**: A model using DARTS for architecture search combined with LSTM layers.
  - **Implementation**: `PC_DARTS_Model`
- **LSTM**:
  - **Architecture**: Long Short-Term Memory
  - **Description**: An LSTM model for sequential audio data classification.
  - **Implementation**: `LSTMClassifier`

## Dataset
The model was trained on the *In The Wild* dataset, containing both real and AI-generated audio samples.

## Demo
Try out the live demo here:  
[Try out my Deepfake Audio Detector on Hugging Face](https://huggingface.co/spaces/SanthoshPrabhu/Deepfake_audio_detector)

### How the Demo Works:
1. Upload a `.wav` audio file using the provided upload button.
2. The model converts the audio into a Mel-spectrogram.
3. It processes the spectrogram through the SENet model to analyze the audio.
4. The system outputs:
   - Whether the audio is **real** or **fake**.
   - A **confidence score**, indicating how certain the model is about the prediction.

### Screenshots of the Demo:
![Demo Screenshot 1](./Demo%20Screenshot%201.png)  


![Demo Screenshot 2](./Demo%20Screenshot%202.png)  

## Installation
To run the project locally, follow these steps:

### 1. Clone the Repository
```sh
git clone https://github.com/SanthoshPrabhu1019/deepfake-audio-detection.git
cd deepfake-audio-detection
```

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
```sh
pip install torch torchaudio timm gradio huggingface_hub
```

### 3. Run the Gradio App
```sh
python app.py
```
This will start a local instance of the Gradio interface.

## Usage
1. Upload an audio file (WAV format).
2. The model processes it into a Mel-spectrogram.
3. It predicts whether the audio is real or fake with a confidence score.

## Deployment
The model is deployed on Hugging Face Spaces and can be accessed using the provided link above.

## Author
Santhosh Prabhu  

