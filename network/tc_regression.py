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
    def __init__(self, config):
        super(RegressionNet, self).__init__()

        """
        Simple linear regression classifier with softmax output layer
        """
        
        self.lin_1 = nn.Conv2d()
        self.softmax = None 
        
        self.f = None

    def forward(self, x):
        x = self.input_layer(x)
        x = self.f(self.lin_1(x))
        x = self.f(self.lin_2(x))
        y_hat = self.out(x)
        
        return y_hat



