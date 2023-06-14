import torch
import torch.nn as nn
from torch import flatten

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,kernel_size=(3, 3),padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.fc1 = nn.Linear(in_features=80, out_features=40)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(40, 20)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        x = flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        output = self.fc3(x)

        return output
    
    def reset(self):
        return
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))