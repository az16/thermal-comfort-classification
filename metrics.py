from sklearn import metrics
import torch 
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from sklearn.metrics import confusion_matrix
import torchmetrics.functional as plf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from network.learning_models import Discretizer
import torch.nn.functional as F

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, use_weights=True):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.use_weights = use_weights
        self.num_classes = num_classes

    def forward(self, pred, target):
        weight = None
        if self.use_weights:
            counts = torch.bincount(target.view(-1).int(), minlength=self.num_classes)
            weight = 1.0 / counts
        return F.cross_entropy(pred, target, weight=weight)

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_hat, y):
        current_correct = torch.sum(torch.argmax(y_hat.softmax(dim=1), dim=1) == y).item()
        current_total = y_hat.shape[0]
        return current_correct/current_total

class OrderAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_correct = 0
        self.current_total = 0
    
    def reset(self):
        self.current_correct = 0
        self.current_total = 0
    
    def forward(self, y_hat, y):
        self.current_correct += torch.sum(torch.all((torch.round(y_hat) == y), dim=1)).item()
        self.current_total += y_hat.shape[0]
        return self.current_correct/self.current_total