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
        
        
class RegressionNet(BaseModel):
    def __init__(self, config):
        super(RegressionNet, self).__init__()

        """
        Simple regression classifier as a first step to find suitable classifier type
        """

    def forward(self, x):
        #return y_hat
        pass



