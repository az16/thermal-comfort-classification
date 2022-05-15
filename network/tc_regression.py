import torch
import torch.nn as nn
import torchvision
import numpy as np
import computations as cp

class BaseModel(nn.Module):
    def load(self, path):
        """
        Load model from file.
        Args:
           path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
        #pass
    
    def save(self, path):
        pass
        
        
class RegressionNet(BaseModel):
    def __init__(self, input_size):
        super(RegressionNet, self).__init__()

        """
        Simple linear regression classifier without activation layer
        """
        
        self.lin_1 = nn.Linear(input_size, 1)
        

    def forward(self, x):
        
        x = self.lin_1(x)
        
        return x



