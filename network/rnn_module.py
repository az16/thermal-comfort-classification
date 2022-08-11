from multiprocessing import cpu_count
from metrics import Accuracy
from metrics import rmse, mae, compute_confusion_matrix
import numpy as np 
import torch
import pytorch_lightning as pl
from network.learning_models import RNN

from dataloaders.path import *

"""
    The training module for the LSTM architecture is defined here. If LSTM training is supposed to be adjusted, change this.
"""
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
                
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        self.accuracy = Accuracy()
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        
        num_features = len(self.opt.columns)-1 #-1 to neglect labels
        num_categories = self.opt.scale #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        self.model = RNN(num_features, num_categories, hidden_dim=self.opt.hidden, n_layers=self.opt.layers, dropout=self.opt.dropout)

        
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
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("{}_loss".format(name), loss, prog_bar=True, logger=True)
        self.log("{}_acc".format(name), accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "{}_loss".format(name): loss, "{}_acc".format(name): accuracy}
                                         

    def training_step(self, batch, batch_idx):
        return self(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, "test")