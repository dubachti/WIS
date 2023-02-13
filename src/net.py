import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1)
        self.resnet = resnet18(weights=None)
        self.fc2 = nn.Linear(1000, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.resnet(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output