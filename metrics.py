from sklearn import metrics
import torch 
import torch.nn as nn
# import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from sklearn.metrics import accuracy_score, confusion_matrix
import torchmetrics.functional as plf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from network.learning_models import Discretizer
"""
    This file includes all metric computations for model performance.
"""

class MetricLogger(object):
    def __init__(self, metrics, module, gpu):
        self.context = module
        self.computer = MetricComputation(metrics, gpu)
    
    def log_train(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = {"loss": loss}
        self.context.log("loss", loss)
        for name, value in zip(self.computer.names, values):
            self.context.log("train_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("train_{}(AVG)".format(name), self.computer.avg(name), logger=True, prog_bar=True)
            result[name] = value
        return result

    def log_val(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = {'val_loss': loss,
                  'preds': pred,
                  'target': target}
        self.context.log("val_loss", loss, logger=True, prog_bar=True)
        for name, value in zip(self.computer.names, values):
            self.context.log("val_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("val_{}(AVG)".format(name), self.computer.avg(name), logger=True, prog_bar=True)
            result[name] = value
        return result

    def log_test(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = {'test_loss': loss}
        self.context.log("test_loss", loss, logger=True, prog_bar=True)
        for name, value in zip(self.computer.names, values):
            self.context.log("test_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("test_{}(AVG)".format(name), self.computer.avg(name), logger=True, prog_bar=True)
            result[name] = value
        return result

    def reset(self):
        self.computer.reset()

class MetricComputation(object):
    def __init__(self, metrics, gpu):
        self.names = metrics
        self.metrics = [METRICS[m] for m in metrics]
        if gpu: self.metrics = [m.cuda() for m in self.metrics]
        self.reset()

    def reset(self):
        [metric.reset() for metric in self.metrics]

    def compute(self, pred, target):
        current_values = [metric(pred, target) for metric in self.metrics]
        return current_values

    def avg(self, metric):
        if isinstance(metric, str): return self.metrics[self.names.index(metric)].compute()
        assert False, "metric must be str"

class MAELoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        
        self.reduction = reduction 
        self.disc = Discretizer(7)
    
    def _reduction(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'{self.reduction} is not a valid reduction')
    
    def forward(self, y_hat, y):
        y_hat = self.disc(y_hat)
        l = torch.abs(torch.subtract(y_hat, y))
        return self._reduction(l)


def compute_confusion_matrix(preds, labels, label_names, current_epoch, context, mode):
    """
        This method is used to create confusion matrices from predictions and labels.

        Args:
            preds (tensor): the model predictions
            labels (tensor): the labels
            label_names (list): the label names
            current_epoch (int): the current epoch
            context (logger): the logger to which the confusion matrix should be added
            mode (str): string that should denote whether this is a training or validation matrix
    """
    #print(preds, labels)
    cfm = confusion_matrix(labels, preds, labels=label_names, normalize='true')
    #print(cfm)
    df = pd.DataFrame(cfm, index=label_names, columns=label_names)

    #visualization

    m_val = sns.heatmap(df, annot=True, fmt=".1%", cmap="Blues")
    m_val.set_yticklabels(m_val.get_yticklabels(), rotation=0, ha='right', size=10)
    m_val.set_xticklabels(m_val.get_xticklabels(), rotation=30, ha='right', size=10)
    plt.ylabel('Target Labels')
    plt.xlabel('Predicted Label')
    fig = m_val.get_figure()
    #plt.close(fig)
    context.logger.experiment.add_figure("Confusion Matrix {0}".format(mode), fig, current_epoch)

def visualize_feature_importance(feature_imp, feature_names, i=None):
    """
        Used for random forest feature importance visualization.

        Args:
            feature_imp (dict): a dict of feature importances
            feature_names (list): the feature names to be included
            i (int, optional): The run id. Defaults to None.
    """
    plt.rcParams["figure.figsize"] = [16,9]
    plt.rcParams.update({'font.size': 12})
    feature_imp = pd.Series(feature_imp, index=feature_names).sort_values(ascending=False)
    m = sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Feature Importance Visualization")
    fig = m.get_figure()
    
    if not i is None:
        fig.savefig("sklearn_logs/media/{0}_{1}.png".format("tree_importance_scores", i))
    else:
        fig.savefig("sklearn_logs/media/{0}.png".format("test"))
    
    plt.close(fig)

def accuracy_score(preds, labels):
    """
        Computes the accuracy score for one epoch using pytorch metrics.
        
        Args:
            preds (tensor): the model predictions
            labels (tensor): the actual labels
    """
    return metrics.accuracy_score(labels,preds)

def mean_accuracy(preds, labels):
    """
        Computes the mean accuracy over one epoch.

        Args:
            preds (tensor): the model predictions
            labels (tensor): the actual labels

    """
    n = len(preds)
    correct = 0
    for i in range(n):
        if preds[i] == labels[i]:
            correct += 1
    
    return correct/n

def mean_absolute_error(preds, labels):
    """
        Computes the mean absolute error over one epoch.

        Args:
            preds (tensor): the model predictions
            labels (tensor): the actual labels

    """
    n = len(preds)
    distance = 0
    labels = labels.values
    for i in range(n):
        distance += np.abs(preds[i]-labels[i])
    
    return distance/n
        

def rmse(preds, targets):
    """
        Computes the root mean squared error over one epoch.

        Args:
            preds (tensor): the model predictions
            labels (tensor): the actual labels

    """
    B = 1
    try: B = preds.shape[0]
    except: pass
    return torch.sqrt(torch.sum(torch.pow(targets-preds, 2))*(1/B))

def mae(preds, targets):
    """
        Computes the mean absolute error over one epoch.

        Args:
            preds (tensor): the model predictions
            labels (tensor): the actual labels

    """
    B = 1
    try: B = preds.shape[0]
    except: pass
    tmp = torch.abs(targets-preds)
    return torch.sum(tmp)*(1/B)


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_hat, y):
        current_correct = torch.sum(torch.all((torch.round(y_hat) == y), dim=1)).item()
        current_total = y_hat.shape[0]
        return current_correct/current_total
    

# METRICS = plf.__dict__ #pl.metrics.functional.__dict__ 
# METRICS['mse'] = METRICS['mean_squared_error']
# METRICS['msle'] = METRICS['mean_squared_log_error']
# METRICS['mae'] = METRICS['mean_absolute_error']
# METRICS['accuracy'] = Accuracy(num_classes=7)
# METRICS['precision'] = Precision(num_classes=7)
# METRICS['recall'] = Recall(num_classes=7)
# METRICS['f1-score'] = F1Score(num_classes=7)

if __name__ == "__main__":
    # preds = torch.Tensor([[1.1],
    #                       [3.4],
    #                       [1.9],
    #                       [-3.7]])
    
    preds = torch.Tensor([2.4,-3.1,2.2,1.6])
    target = torch.IntTensor([2.2,4.2,2.2,1.2])
    
    print(rmse(preds, target))
    print((mae(preds, target)))