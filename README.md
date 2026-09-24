# 🇰🇭 Khmer Deepfake Audio Detection System

> **A High-Accuracy Deep Learning Model for Detecting AI-Generated Speech in the Khmer Language.**

##  Project Overview
With the rapid advancement of Text-to-Speech (TTS) technology, distinguishing between real human voices and AI-generated "deepfake" audio has become a critical challenge. This project addresses this issue specifically for the **Khmer language**, a low-resource language where such tools are often under-represented in security research.

We developed a **Convolutional Neural Network (CNN)** that analyzes audio spectrograms to detect microscopic digital artifacts left by AI voice generators. The model achieves **99.8% accuracy** on unseen test data, demonstrating robust performance across different speakers and audio qualities.

---

##  Key Features
- **High Accuracy:** Achieved **99.8%** on a blind test set of 443 audio files.
- **Robust Architecture:** Uses Log-Mel Spectrograms to visualize frequency patterns invisible to the human ear.
- **Data Pipeline:** Automated handling of large datasets (1.7GB+), including folder flattening, gender balancing, and train/val/test splitting.
- **Adaptive Learning:** Implemented "Hard Negative Mining" to retrain the model on tricky samples, improving real-world reliability.
- **Language Specific:** Optimized for Khmer phonetics and vocal characteristics.

---

##  Tech Stack
| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning Framework** | PyTorch (with CUDA GPU acceleration) |
| **Audio Processing** | Librosa (for Mel-Spectrogram extraction) |
| **Data Handling** | Pandas, NumPy, OS/Shutil |
| **Environment** | Google Colab (Free Tier T4 GPU) |
| **Version Control** | Git & GitHub |

---

##  Methodology

### 1. Data Collection & Preparation
- **Source:** Combined public datasets (Mozilla Common Voice, Fleurs) with a custom teacher-provided dataset (~4,400 files).
- **Structure:** Organized into `real` (human) and `fake` (AI-generated) categories, further split by gender (`male`/`female`).
- **Preprocessing:**
  - Resampled all audio to **16kHz Mono**.
  - Padded or truncated clips to a fixed length of **3 seconds**.
  - Converted raw waveforms into **Log-Mel Spectrograms (64 mel bins)**, turning 1D audio into 2D images for CNN analysis.

### 2. Model Architecture
We designed a custom 2D CNN optimized for image-like spectrogram data:
- **Input Layer:** Accepts $1 \times 64 \times \text{time\_steps}$ tensors.
- **Convolutional Blocks:** Two blocks of `Conv2d` → `ReLU` → `MaxPool2d` to extract high-frequency features.
- **Fully Connected Layers:** Uses `LazyLinear` for dynamic input sizing, followed by `Dropout(0.5)` to prevent overfitting.
- **Output:** Binary classification (Real vs. Fake) using Softmax activation.

### 3. Training Strategy
- **Split:** 80% Train, 10% Validation, 10% Test.
- **Optimizer:** Adam ($lr=0.001$).
- **Loss Function:** CrossEntropyLoss.
- **Epochs:** 15 (Early stopping observed at Epoch 3 with 100% Val Acc).
- **Improvement Loop:** Detected a false negative on a compressed MP3 file, added it to the training set as a "Hard Negative," and retrained to improve robustness.

---

##  Results

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | ~99.5% |
| **Validation Accuracy** | 100.0% |
| **Final Test Accuracy** | **99.8%** |
| **Dataset Size** | 4,422 Files |

*Note: The high accuracy indicates the model successfully learned to identify specific "vocoder artifacts" unique to the AI generation process used in the dataset.*

---

##  How to Run This Project

### Prerequisites
- A Google account (to use Google Colab).
- Basic knowledge of Python.

### Step-by-Step Execution
1. **Clone or Download:**
   - Download the `Khmer_Deepfake_Detection.ipynb` file from this repository.
   
2. **Open in Google Colab:**
   - Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/).
   - Go to **Runtime** → **Change runtime type** → Select **GPU (T4)**.

3. **Run the Cells:**
   - **Cell 1:** Mount Google Drive (if using external data).
   - **Cell 2:** Install dependencies (`librosa`, `torch`, etc.).
   - **Cell 3:** Preprocess and Split Data (Flattens folders and creates Train/Val/Test sets).
   - **Cell 4:** **Train the Model** (Takes ~3 minutes).
   - **Cell 5:** **Evaluate** on the Test Set.
   - **Cell 6:** **Upload & Predict** (Upload your own `.mp3` or `.wav` file to test it live!).

---

##  Future Improvements
- **Real-Time Detection:** Integrate with a microphone stream for live deepfake detection.
- **Multi-Language Support:** Expand the model to detect deepfakes in Vietnamese, Thai, and Lao.
- **Web Interface:** Deploy the model using Streamlit or Gradio for a user-friendly drag-and-drop interface.
- **Explainable AI (XAI):** Use Grad-CAM to visualize exactly *which* part of the spectrogram triggered the "Fake" prediction.

---

##  Acknowledgments
- **Dataset Provider:** Special thanks to the instructor for providing the massive 1.7GB curated Khmer audio dataset.
- **Tools:** Built using PyTorch, Librosa, and Google Colab's free GPU resources.

---

##  License
This project is open-source and available for educational purposes. Feel free to use this code for your own research or learning projects.

###  Final Comparison of Approaches

Both models were evaluated on the exact same held-out test set (443 files: 182 Real, 261 Fake) using identical preprocessing (16kHz, 3-second Log-Mel Spectrograms).

| Metric | Approach 1: Custom CNN (From Scratch) | Approach 2: ResNet18 (Transfer Learning) |
| :--- | :--- | :--- |
| **Test Accuracy** | 99.8% | 100.0% |
| **Precision (Fake)** | 1.00 | 1.00 |
| **Recall (Fake)** | 1.00 | 1.00 |
| **F1-Score (Fake)** | 1.00 | 1.00 |
| **Trainable Parameters** | ~105,000 | ~11.2 Million |
| **Training Time (5 epochs)** | ~45 seconds | ~120 seconds |
| **Hardware** | Google Colab T4 GPU | Google Colab T4 GPU |

###  Hyperparameter Tuning (Section 5C)
For the best-performing approach (ResNet18 Transfer Learning), the following hyperparameters were systematically tested:
1. **Learning Rate:** Tested `[0.001, 0.0001, 0.00001]`. A learning rate of `0.0001` was selected. Higher rates caused the model to overwrite the valuable pre-trained ImageNet weights too quickly, while lower rates resulted in unnecessarily slow convergence.
2. **Regularization (Dropout):** For the Custom CNN, Dropout was tested at `[0.3, 0.5, 0.7]`. A value of `0.5` provided the best balance, preventing overfitting (keeping Train and Validation curves closely aligned) without underfitting.
3. **Batch Size:** Tested `[16, 32]`. Batch size 16 was chosen for ResNet18 to fit comfortably within the T4 GPU memory limits while maintaining stable gradient updates.

###  Error Analysis & Limitations (Section 5D)
- **Error Analysis:** The ResNet18 model achieved a perfect score (0 misclassifications) on the 443-file test set. The Custom CNN had a negligible error rate (<0.2%), occasionally misclassifying highly compressed `.mp3` files where high-frequency vocoder artifacts were smoothed out. 
- **Limitations:** The current model is optimized for clean, 3-second audio clips. It may struggle with audio containing heavy background noise (e.g., traffic, music) or languages other than Khmer.
- **Future Work:** Integrate data augmentation (adding Gaussian noise, time-stretching) during training to improve robustness, and deploy the model as a real-time web application.

  ###  AI Assistance Disclosure
- **Tools Used:** Qwen AI
- **Scope of Use:** Debugging PyTorch tensor shapes, drafting README markdown syntax, and generating boilerplate evaluation code.
- **Verification:** All code logic, hyperparameter choices, and analysis represent my own work and were manually verified.
