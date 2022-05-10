import torch 
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.metric import Metric
import torchmetrics.functional as plf

class MetricLogger(object):
    def __init__(self, metrics, module):
        self.context = module
        self.computer = MetricComputation(metrics)
    
    def log_train(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = {"loss": loss}
        self.context.log("loss", loss)
        for name, value in zip(self.computer.names, values):
            self.context.log("train_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("train_{}(AVG)".format(name), self.computer.avg(name), logger=False, prog_bar=True)
            result[name] = value
        return result

    def log_val(self, pred, target):
        values = self.computer.compute(pred, target)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("val_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("val_{}(AVG)".format(name), self.computer.avg(name), logger=False, prog_bar=True)
            result[name] = value
        return result

    def log_test(self, pred, target):
        values = self.computer.compute(pred, target)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("test_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("test_{}(AVG)".format(name), self.computer.avg(name), logger=False, prog_bar=True)
            result[name] = value
        return result

    def reset(self):
        self.computer.reset()

class MetricComputation(object):
    def __init__(self, metrics):
        self.names = metrics
        self.metrics = [METRICS[m] for m in metrics]
        self.reset()

    def reset(self):
        [metric.reset() for metric in self.metrics]

    def compute(self, pred, target):
        current_values = [metric(pred, target) for metric in self.metrics]
        return current_values

    def avg(self, metric):
        if isinstance(metric, str): return self.metrics[self.names.index(metric)].compute()
        assert False, "metric must be str"


METRICS = plf.__dict__ #pl.metrics.functional.__dict__ 
METRICS['mse'] = METRICS['mean_squared_error']
METRICS['msle'] = METRICS['mean_squared_log_error']
METRICS['mae'] = METRICS['mean_absolute_error']
METRICS['accuracy'] = Accuracy(top_k=1,multiclass=True)
METRICS['precision'] = Precision(top_k=1,multiclass=True)
METRICS['recall'] = Recall(top_k=1,multiclass=True)
METRICS['f1-score'] = F1Score(top_k=1,multiclass=True)
