import gradio as gr
import torch
import librosa
import numpy as np
import torchvision.models as models
import torch.nn as nn
import os

# ==========================================
# 1. DEFINE AND LOAD MODELS
# ==========================================
class KhmerDeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

def get_resnet18():
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
    return resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = KhmerDeepfakeCNN().to(device)
# Note: In a real deployment, you would load the .pth file here. 
# For GitHub, we just define the architecture.

# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_deepfake(audio_file, model_choice):
    if audio_file is None:
        return {"Real (Human)": 0.5, "Fake (AI)": 0.5}, "Please upload an audio file."
        
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    max_len = 16000 * 3
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
        
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    channels = 3 if model_choice == "ResNet18 (Transfer Learning)" else 1
    tensor_spec = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if channels == 3:
        tensor_spec = tensor_spec.repeat(1, 3, 1, 1)
    tensor_spec = tensor_spec.to(device)
    
    model = cnn_model if model_choice == "Custom CNN (From Scratch)" else get_resnet18().to(device)
    
    with torch.no_grad():
        outputs = model(tensor_spec)
        probs = torch.softmax(outputs, dim=1)[0]
        
    fake_conf = probs[1].item()
    real_conf = probs[0].item()
    label_dict = {"Real (Human)": real_conf, "Fake (AI)": fake_conf}
    
    if fake_conf > real_conf:
        text_out = f"🤖 DETECTED: FAKE (AI) with {fake_conf*100:.2f}% confidence."
    else:
        text_out = f"👤 DETECTED: REAL (Human) with {real_conf*100:.2f}% confidence."
        
    return label_dict, text_out

# ==========================================
# 3. GRADIO UI
# ==========================================
demo = gr.Interface(
    fn=predict_deepfake,
    inputs=[
        gr.Audio(type="filepath", label=" Upload Audio or Record Voice"),
        gr.Radio(["Custom CNN (From Scratch)", "ResNet18 (Transfer Learning)"], value="ResNet18 (Transfer Learning)", label="🧠 Select Model")
    ],
    outputs=[
        gr.Label(label="Prediction Confidence"),
        gr.Textbox(label="Final Verdict")
    ],
    title="🇭 Khmer Deepfake Audio Detector",
    description="Upload a 3-second audio clip to detect if it is a Real Human voice or an AI-generated Fake voice."
)

if __name__ == "__main__":
    demo.launch()