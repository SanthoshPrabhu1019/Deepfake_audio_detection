import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import timm
import streamlit as st
from huggingface_hub import hf_hub_download

# Load the SENet model
class ResNetClassifier(nn.Module):

    def __init__(self):

        super(ResNetClassifier, self).__init__()

        self.model = resnet18(weights='IMAGENET1K_V1')



        # Replace the fully connected layer with a new one for binary classification

        num_features = self.model.fc.in_features  # Get the number of features from the original ResNet's last layer

        self.model.fc = nn.Linear(num_features, 1)  # Change output size to 1 for binary classification



    def forward(self, x):

        return self.model(x)  # Directly return the output from the modified model

# Load the trained model from Hugging Face
def load_model():
    model = ResNetClassifier()
    # Download model file from Hugging Face
    model_path = hf_hub_download(repo_id="SanthoshPrabhu/Deepfake_audio_detection", filename="resnet_model_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Set parameters for the Mel-spectrogram
height = 50
target_length = 250
sample_rate = 16000
n_fft = 2048
hop_length = 512
n_mels = height

# Define the Mel-spectrogram transform
mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# Load the trained model
model = load_model()

# Define a function to preprocess the audio
def preprocess_audio(audio_file):
    # Load the audio file
    waveform, _ = torchaudio.load(audio_file)
    # Convert to Mel-Spectrogram
    mel_spectrogram = mel_transform(waveform)
    # Normalize the spectrogram
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

    # Pad or truncate the spectrogram
    if mel_spectrogram.size(2) < target_length:
        padding = target_length - mel_spectrogram.size(2)
        mel_spectrogram = nn.functional.pad(mel_spectrogram, (0, padding), "constant", 0)
    elif mel_spectrogram.size(2) > target_length:
        mel_spectrogram = mel_spectrogram[:, :, :target_length]

    # Ensure mel_spectrogram has the shape [3, height, target_length]
    if mel_spectrogram.size(0) == 1:
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1)

    return mel_spectrogram

# Streamlit UI
st.title("Deepfake Audio Detection")
st.write("Upload an audio file to check if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Add the 'Detect' button
    if st.button("Detect"):
        # Preprocess the audio file
        mel_spectrogram = preprocess_audio(uploaded_file)
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = model(mel_spectrogram)
            prediction = torch.sigmoid(output).item()  # Get prediction probability

        # Display results
        if prediction > 0.5:
            st.success("The audio is classified as **Real**.")
        else:
            st.error("The audio is classified as **Fake**.")
