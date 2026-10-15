import os, sys, torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import TeacherKhmerDataset
from src.models import KhmerDeepfakeCNN, get_resnet18

def evaluate_model(model_name, checkpoint_path, channels=1):
    print(f"--- Evaluating {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('/content/results', exist_ok=True)
    test_loader = DataLoader(TeacherKhmerDataset('test', channels=channels), batch_size=16, shuffle=False)
    model = (KhmerDeepfakeCNN() if model_name == 'CustomCNN' else get_resnet18())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            outputs = model(specs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])
    print(report)
    with open(f'/content/results/{model_name.lower()}_report.txt', 'w') as f: f.write(report)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'/content/results/{model_name.lower()}_confusion_matrix.png')

if __name__ == "__main__":
    evaluate_model('CustomCNN', '/content/customcnn_best.pth', channels=1)
    evaluate_model('ResNet18', '/content/resnet18_best.pth', channels=3)
