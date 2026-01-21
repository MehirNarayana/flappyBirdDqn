import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    
    def __init__(self, input_shape, n_actions):
        super().__init__()

                
        input_shape = input_shape[0]
        
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
    )
        
       
   

    def forward(self, x):
        
        
        x = x.view(x.size(0), -1).float()
        return  self.fc(x)
        
