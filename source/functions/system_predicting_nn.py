import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_processing import *


class systemPredictingNN(nn.Module):
    def __init__(self):
        super(systemPredictingNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        
        self.fc1 = nn.Linear(32 * 9 * 9+1,32*9*9)

        self.deconv1=nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)

        self.deconv2=nn.ConvTranspose2d(16, 1, kernel_size=8, stride=4)


    def forward(self,state,action):
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x = x.view((-1, 32 * 9 * 9))
        
        action = action.unsqueeze(1)
        x=torch.cat((x,action),dim=1)

        x=self.fc1(x)

        x=x.view((-1,32,9,9))

        x=F.relu(self.deconv1(x))
        x=torch.sigmoid(self.deconv2(x))

        x = x.squeeze(1)
        
        if not self.training:
            x = postprocess(x)
        return x

