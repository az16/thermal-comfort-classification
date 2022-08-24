from multiprocessing import cpu_count
from metrics import Accuracy, OrderAccuracy, WeightedCrossEntropyLoss, OrderAccuracy
import numpy as np 
import torch
import pytorch_lightning as pl
from network.learning_models import RNN

from dataloaders.path import *
from dataloaders.utils import class2order, order2class
from metrics import OrderAccuracy, Accuracy

"""
    The training module for the LSTM architecture is defined here. If LSTM training is supposed to be adjusted, change this.
"""
class Oracle(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.accuracy = Accuracy()
        self.order_acc = OrderAccuracy()

    def predict(self, batch):
        _, _, y_order = batch
        return y_order

    def test_step(self, batch, batch_idx):
        _, y_class, y_order = batch
        y_hat = self.predict(batch)
        B = y_order.shape[0]
        
        loss = self.mse(y_hat, y_order)
        pred_class = order2class(y_hat)
        pred_order = y_hat

        mse = self.mse(pred_order, y_order)

        accuracy = self.accuracy(pred_class, y_class)
        order_acc = self.order_acc(pred_order, y_order)

        results = {
            "loss": loss,
            "acc": accuracy,
            "order_acc": order_acc,
            "mse": mse            
            }

        self.log_metrics(B, results, 'test')
        return results

    def log_metrics(self, batch_size, metrics, split):
        for name, value in metrics.items():
            self.log("{}_{}".format(split, name), value, prog_bar=True, logger=True, batch_size=batch_size)


class RandomGuess(Oracle):
    def predict(self, batch):
        _, _, y_order = batch
        return torch.rand_like(y_order)
    
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, opt):
        super().__init__()
        opt.num_features = len(opt.columns)-1 #-1 to neglect labels
        opt.num_categories = 7 #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        self.save_hyperparameters()
        self.opt = opt        

        self.wce = WeightedCrossEntropyLoss(self.opt.num_categories, self.opt.use_weighted_loss)
        self.mse = torch.nn.MSELoss()
        self.accuracy = Accuracy()
        self.order_acc = OrderAccuracy() 
                
        self.model = RNN(self.opt.num_features, self.opt.num_categories, hidden_dim=self.opt.hidden, n_layers=self.opt.layers, dropout=self.opt.dropout, latent_size=self.opt.latent_size)

        
    def configure_optimizers(self):
        """
            Sets up optmizers and defines learning rate decay.
        """
        train_param = self.model.parameters()
        # Training parameters
        optimizer = torch.optim.Adam(train_param, lr=self.opt.learning_rate)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]

    def forward(self, batch, batch_idx, name):        
        x, y_class, y_order = batch
        B = y_class.shape[0]
        x_input = torch.zeros((B, self.opt.sequence_window, self.opt.num_features)).to(x)
        x_input[:, 0:x.shape[1]] = x
        y_hat = self.model(x)
        if self.opt.loss == 'wce':
            loss = self.wce(y_hat, y_class)
            pred_class = y_hat
            pred_order = class2order(y_hat)            
        elif self.opt.loss == 'mse':
            loss = self.mse(y_hat, y_order)
            pred_class = order2class(y_hat)
            pred_order = y_hat

        wce = self.wce(pred_class, y_class)
        mse = self.mse(pred_order, y_order)

        accuracy = self.accuracy(pred_class, y_class)
        order_acc = self.order_acc(pred_order, y_order)

        results = {
            "loss": loss,
            "acc": accuracy,
            "order_acc": order_acc,
            "wce": wce, 
            "mse": mse            
            }

        self.log_metrics(B, results, name)
        return results

    def log_metrics(self, batch_size, metrics, split):
        for name, value in metrics.items():
            self.log("{}_{}".format(split, name), value, prog_bar=True, logger=True, batch_size=batch_size)
                                         
    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")