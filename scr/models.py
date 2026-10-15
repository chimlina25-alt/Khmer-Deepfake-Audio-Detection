import torch, torch.nn as nn, torchvision.models as models
class KhmerDeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
    def forward(self, x): return self.fc(self.conv(x))
def get_resnet18():
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
    return resnet18
