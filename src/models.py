import torch
import torch.nn as nn
import torchvision.models as models

class KhmerDeepfakeCNN(nn.Module):
    def __init__(self):
        super(KhmerDeepfakeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

def get_resnet18():
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    return resnet18
