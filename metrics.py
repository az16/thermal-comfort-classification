import torch 
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.metric import Metric
import torchmetrics.functional as plf


class MetricLogger(object):
    def __init__(self, metrics, module, gpu):
        self.context = module
        self.computer = MetricComputation(metrics, gpu)
    
    def log_train(self, pred, target, loss):
        values = self.computer.compute(pred, target, m=0)
        result = {"loss": loss}
        self.context.log("loss", loss)
        for name, value in zip(self.computer.names, values):
            self.context.log("train_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("train_{}(AVG)".format(name), self.computer.avg(name, m=0), logger=False, prog_bar=True)
            result[name] = value
        return result

    def log_val(self, pred, target):
        values = self.computer.compute(pred, target, m=1)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("val_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("val_{}(AVG)".format(name), self.computer.avg(name, m=1), logger=False, prog_bar=True)
            result[name] = value
        return result

    def log_test(self, pred, target):
        values = self.computer.compute(pred, target, m=2)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("test_{}".format(name), value, logger=True, on_epoch=True)
            self.context.log("test_{}(AVG)".format(name), self.computer.avg(name, m=2), logger=False, prog_bar=True)
            result[name] = value
        return result

    def reset(self):
        self.computer.reset()

class MetricComputation(object):
    def __init__(self, metrics, gpu):
        self.names = metrics
        self.train_metrics = [METRICS[m] for m in metrics]
        self.val_metrics = [METRICS[m] for m in metrics]
        self.test_metrics = [METRICS[m] for m in metrics]
        self.all_metrics = [self.test_metrics, self.val_metrics, self.test_metrics]
        if gpu: [m.cuda() for m in self.train_metrics];[m.cuda() for m in self.val_metrics];[m.cuda() for m in self.test_metrics]
        self.reset()

    def reset(self):
        for metric in self.all_metrics:
            [m.reset() for m in metric]

    def compute(self, pred, target, m=0):
        current_values = [metric(pred, target) for metric in self.all_metrics[m]]
        return current_values

    def avg(self, metric_name, m=0):
        if isinstance(metric_name, str): 
            metric = self.all_metrics[m]; 
            return metric[self.names.index(metric_name)].compute()
        assert False, "metric_name must be str"


METRICS = plf.__dict__ #pl.metrics.functional.__dict__ 
METRICS['mse'] = METRICS['mean_squared_error']
METRICS['msle'] = METRICS['mean_squared_log_error']
METRICS['mae'] = METRICS['mean_absolute_error']
METRICS['accuracy'] = Accuracy(multiclass=True, num_classes=7)
METRICS['precision'] = Precision(multiclass=True, num_classes=7)
METRICS['recall'] = Recall(multiclass=True, num_classes=7)
METRICS['f1-score'] = F1Score(multiclass=True, num_classes=7)
