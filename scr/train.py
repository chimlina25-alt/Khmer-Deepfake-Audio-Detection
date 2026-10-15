import os, sys, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import TeacherKhmerDataset
from src.models import KhmerDeepfakeCNN, get_resnet18

def train_model(model_name, epochs=2, batch_size=16, lr=0.001, channels=1):
    print(f"--- Training {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TeacherKhmerDataset('train', channels=channels), batch_size=batch_size, shuffle=True)
    model = (KhmerDeepfakeCNN() if model_name == 'CustomCNN' else get_resnet18()).to(device)
    if model_name != 'CustomCNN': lr = 0.0001
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            _, predicted = torch.max(model(specs).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss_sum/len(train_loader):.4f} | Acc: {100*correct/total:.1f}%")
    torch.save(model.state_dict(), f'/content/{model_name.lower()}_best.pth')

if __name__ == "__main__":
    train_model('CustomCNN', epochs=2, batch_size=32, channels=1)
    train_model('ResNet18', epochs=2, batch_size=16, channels=3)
