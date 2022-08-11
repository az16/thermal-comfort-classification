from multiprocessing import cpu_count
from metrics import Accuracy
from metrics import rmse, mae, compute_confusion_matrix
import numpy as np 
import torch
import pytorch_lightning as pl
from network.learning_models import RNN

from dataloaders.pmv_loader import PMV_Results
from dataloaders.path import *

"""
    The training module for the LSTM architecture is defined here. If LSTM training is supposed to be adjusted, change this.
"""
gpu_mode=False
# RDM_Net.use_cuda=False
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        
        #self.pmv_results = PMV_Results()
        self.classification_loss = False
        
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()
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

    def forward(self, x):
        # rgb, x = x
        x = x.float()
        # print(rgb.shape)
        out = self.model(x)
        return out
                                         

    def training_step(self, batch, batch_idx):
        """
            Defines what happens during one training step.
            
            Args:
                batch: the current training batch that is used
                batch_idx: the id of the batch
        """
        if batch_idx == 0: self.acc_train.reset(), self.train_preds.clear(), self.train_labels.clear()
        x, y = batch
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if self.classification_loss:
            y = y.long()
        else: y = y.float()
        #print(y_hat, y)
        loss = self.criterion(y_hat, y)
        accuracy = self.acc_train(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", accuracy, prog_bar=True, logger=True)
        # self.log("train_rsme", rmse(y_hat, y), prog_bar=True, logger=True)
        # self.log("train_mae", mae(y_hat, y), prog_bar=True, logger=True)
        
        #preds, y = self.prepare_cfm_data(y_hat, y)

        #self.train_preds.append(self.label_names[int(preds[0])])
        #self.train_labels.append(self.label_names[int(y[0])])
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """
            Defines what happens during one validation step.
            
            Args:
                batch: the current validation batch that is used
                batch_idx: the id of the batch
        """
        if batch_idx == 0: self.acc_val.reset(), self.val_preds.clear(), self.val_labels.clear()
        x, y = batch
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if self.classification_loss:
            y = y.long()
        else: y = y.float()
        loss = self.criterion(y_hat, y)
        accuracy = self.acc_val(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)
       
        
        #preds, y = self.prepare_cfm_data(y_hat, y)
        #self.val_preds.append(self.label_names[int(preds[0])])
        #self.val_labels.append(self.label_names[int(y[0])])
        
        return {"val_loss": loss, 'val_acc': accuracy}
        
    
    def prepare_cfm_data(self, preds, y):
        """
            This method is used to convert predictions and labels to a representation
            that can be turned into a confusion matrix.
            
            Args:
                preds: the model predictions
                y: the labels
        """
        preds = torch.sum(preds.cpu(), dim=1)
        preds = torch.add(preds, torch.multiply(torch.ones_like(preds), -1.0))
        # print(preds)
        # print(torch.round(preds))
        preds = torch.round(preds)
        y = torch.sum(y.cpu().long(), dim=1)
        y = torch.add(y, torch.multiply(torch.ones_like(y), -1.0))
        return preds, y