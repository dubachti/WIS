import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(1, 3, 3, 1)
        self.resnet = resnet18()
        self.fc = nn.Linear(1000, 32)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,2,3,1), 
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(2,2,3,1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(2,3,3,1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(588, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        output = F.log_softmax(x, dim=1)
        return output