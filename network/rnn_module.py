from multiprocessing import cpu_count
from metrics import Accuracy, OrderAccuracy
from metrics import WeightedCrossEntropyLoss
import numpy as np 
import torch
import pytorch_lightning as pl
from network.learning_models import RNN

from dataloaders.path import *
from dataloaders.utils import class2order, order2class

"""
    The training module for the LSTM architecture is defined here. If LSTM training is supposed to be adjusted, change this.
"""
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt

        num_features = len(self.opt.columns)-1 #-1 to neglect labels
        num_categories = self.opt.scale #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot

        self.wce = WeightedCrossEntropyLoss(num_categories, self.opt.use_weighted_loss)
        self.mse = torch.nn.MSELoss()
        self.class_accuracy = Accuracy()        
        self.order_accuracy = OrderAccuracy()
        
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        
        self.model = RNN(num_features, num_categories, hidden_dim=self.opt.hidden, n_layers=self.opt.layers, dropout=self.opt.dropout, latent_size=self.latent_size)

        
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

    def forward(self, batch, name):
        x, y_class, y_order = batch
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

        accuracy_class = self.class_accuracy(pred_class, y_class)
        accuracy_order = self.order_accuracy(pred_order, y_order)

        self.log("{}_loss".format(name), loss, prog_bar=True, logger=True)
        self.log("{}_class_acc".format(name), accuracy_class, prog_bar=True, logger=True)
        self.log("{}_order_acc".format(name), accuracy_order, prog_bar=True, logger=True)
        self.log("{}_wce".format(name), wce, prog_bar=True, logger=True)
        self.log("{}_mse".format(name), mse, prog_bar=True, logger=True)
        return {
            "loss": loss, 
            "{}_wce".format(name): wce, 
            "{}_mse".format(name): mse,
            "{}_class_acc".format(name): accuracy_class,
            "{}_order_acc".format(name): accuracy_order
            }
                                         

    def training_step(self, batch, batch_idx):
        return self(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, "test")