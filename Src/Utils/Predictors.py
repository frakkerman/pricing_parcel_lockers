import torch
import torch.nn as nn
from torch import flatten
import numpy as np

class CNN_2d(nn.Module):
    def __init__(self,dim,n_layers):
        super().__init__()
        
        kernel1 = (3,3)
        kernel2 = (2,2)
        dilation=0
        stride=(2,2)
        padding=1
        out_channels=20
        
        h1 = np.floor((dim+2*padding-dilation*(kernel1[0]-1)-1)/stride[0]+1)
        w1 = np.floor((dim+2*padding-dilation*(kernel1[1]-1)-1)/stride[1]+1)
        h2 = np.floor((h1+2*padding-kernel2[0])/stride[0]+1)
        w2 = np.floor((w1+2*padding-kernel2[1])/stride[0]+1)
        
        self.conv1 = nn.Conv2d(in_channels=n_layers, out_channels=out_channels,kernel_size=kernel1,padding=padding)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel2, stride=stride,padding=padding)
        
        self.fc1 = nn.Linear(in_features=int(out_channels*h2*w2), out_features=40)
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

class CNN_3d(nn.Module):
    def __init__(self,n_layers):
        super().__init__()
        
        print("Not supported yet, but should be useful for even more complex data, with 4th dimension")
        
        self.conv1 = nn.Conv3d(in_channels=n_layers, out_channels=20,kernel_size=(3, 3, 3),padding=(1,1,1))
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(3, 2, 2), stride=(2, 2))
        
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