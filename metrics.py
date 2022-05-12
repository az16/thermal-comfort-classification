import torch 
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from torchmetrics.metric import Metric
import torchmetrics.functional as plf
import numpy as np 
import cv2 


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
        result = {'val_loss': loss}
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


def computue_confusion_matrix(pred, gt, num_classes, cell_size):
    confmat = ConfusionMatrix(num_classes=num_classes, normalize='true')
    matrix = confmat(pred.cpu().long(), gt.cpu().long()).numpy()
   
    image = np.repeat(matrix, cell_size, axis=1)
    image = np.repeat(image, cell_size, axis=0)
    image *= 255
    image = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_HOT)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for row in range(num_classes):
        for col in range(num_classes):
            text = str(np.round(matrix[row, col]*100, 1)) + "%"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(image, text, (col * cell_size + cell_size - (cell_size + textsize[0]) // 2, row * cell_size + (cell_size + textsize[1]) // 2), font, 1, (255, 255, 255), 1, cv2.LINE_4, False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

METRICS = plf.__dict__ #pl.metrics.functional.__dict__ 
METRICS['mse'] = METRICS['mean_squared_error']
METRICS['msle'] = METRICS['mean_squared_log_error']
METRICS['mae'] = METRICS['mean_absolute_error']
METRICS['accuracy'] = Accuracy(num_classes=7)
METRICS['precision'] = Precision(num_classes=7)
METRICS['recall'] = Recall(num_classes=7)
METRICS['f1-score'] = F1Score(num_classes=7)

if __name__ == "__main__":
    preds = torch.Tensor([[1,2,3,0,5,1],
                          [1,3,3,0,6,1],
                          [1,1,3,0,0,1],
                          [1,5,3,0,0,1]])
    
    #preds = torch.Tensor([2,4,2,1])
    target = torch.IntTensor([2,4,2,1])
    
    acc = Accuracy()
    prec = Precision()
    recall = Recall() 
    f_1 = F1Score()
    
    print("accuracy: {0}".format(acc(preds, target)))
    print("precision: {0}".format(prec(preds, target)))
    print("recall: {0}".format(recall(preds, target)))
    print("f-1: {0}".format(f_1(preds, target)))