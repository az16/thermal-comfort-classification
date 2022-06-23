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

def compute_confusion_matrix(preds, labels, label_names, current_epoch, context, mode):
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
    return metrics.accuracy_score(labels,preds)

def top_k_accuracy_score(preds, labels, k=2, normalize=True):
    k_l = k-1
    total = len(labels)
    correct = 0
    i = 0
    for l in labels:
        y_hat = preds[i]
        if l in [x for x in range(y_hat-k_l,y_hat+k+1)]:
            #print(l,y_hat)
            correct += 1
    
    if normalize:
        return correct/total 
    return correct

def mean_accuracy(preds, labels):
    n = len(preds)
    correct = 0
    for i in range(n):
        if preds[i] == labels[i]:
            correct += 1
    
    return correct/n

def mean_absolute_error(preds, labels):
    n = len(preds)
    distance = 0
    labels = labels.values
    for i in range(n):
        distance += np.abs(preds[i]-labels[i])
    
    return distance/n
        

def rmse(preds, targets):
    B = 1
    try: B = preds.shape[0]
    except: pass
    return torch.sqrt(torch.sum(torch.pow(targets-preds, 2))*(1/B))

def mae(preds, targets):
    B = 1
    try: B = preds.shape[0]
    except: pass
    tmp = torch.abs(targets-preds)
    return torch.sum(tmp)*(1/B)


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_correct = 0
        self.current_total = 0
    
    def reset(self):
        self.current_correct = 0
        self.current_total = 0
    
    def forward(self, y_hat, y):
        self.current_correct += torch.sum((y_hat == y)).item()
        self.current_total += y_hat.shape[0]
        return self.current_correct/self.current_total

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