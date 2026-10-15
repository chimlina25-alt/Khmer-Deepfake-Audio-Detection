import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class TeacherKhmerDataset(Dataset):
    def __init__(self, split, channels=1):
        self.files = []
        self.labels = []
        self.channels = channels
        base_path = f'/content/teacher_dataset_split/{split}'
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                        self.files.append(os.path.join(folder_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        max_len = 16000 * 3
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        tensor_spec = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)
        if self.channels == 3:
            tensor_spec = tensor_spec.repeat(3, 1, 1)
        return tensor_spec, torch.tensor(label, dtype=torch.long)
