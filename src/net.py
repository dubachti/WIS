import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,2,3,1), 
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(2,2,3,1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(2,3,3,1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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
