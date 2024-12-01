import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import timm
import gradio as gr
from huggingface_hub import hf_hub_download

# Load the SENet model
class SENetClassifier(nn.Module):
    def __init__(self):
        super(SENetClassifier, self).__init__()
        self.model = timm.create_model('senet154', pretrained=True)
        # Modify the classifier for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Change output to 1 for binary classification

    def forward(self, x):
        return self.model(x)

# Load the trained model from Hugging Face
def load_model():
    model = SENetClassifier()
    # Download model file from Hugging Face
    model_path = hf_hub_download(repo_id="SanthoshPrabhu/Deepfake_audio_detection", filename="senet_model_best_model.pth")
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

# Define a function for prediction
def detect_deepfake(audio_file):
    mel_spectrogram = preprocess_audio(audio_file)
    mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(mel_spectrogram)
        probability = torch.sigmoid(output).item()  # Get prediction probability

    # Determine the label and confidence
    if probability > 0.5:
        confidence = probability * 100  # Convert to percentage
        result = f"The audio is classified as Real with {confidence:.2f}% confidence."
    else:
        confidence = (1 - probability) * 100  # Convert to percentage
        result = f"The audio is classified as Fake with {confidence:.2f}% confidence."

    return result

# Gradio UI with both upload and record options
interface = gr.Interface(
    fn=detect_deepfake, 
    inputs=[
        gr.Audio(type="filepath", label="Upload an audio file (wav)"),
        gr.Audio(source="microphone", type="filepath", label="Record audio (wav)")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Deepfake Audio Detection",
    description="Upload or record an audio file to check if it's real or fake, along with the confidence level of the prediction."
)

# Launch the Gradio app
interface.launch()
