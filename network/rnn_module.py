from multiprocessing import cpu_count
from metrics import Accuracy
from metrics import rmse, mae, compute_confusion_matrix
import numpy as np 
import torch
import pytorch_lightning as pl
from network.learning_models import RNN
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.pmv_loader import PMV_Results
from dataloaders.path import *

"""
    The training module for the LSTM architecture is defined here. If LSTM training is supposed to be adjusted, change this.
"""
gpu_mode=False
# RDM_Net.use_cuda=False
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, cols, gpus, dropout, hidden, layers, preprocess, augmentation, skip, forecasting, scale, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        gpu_mode = not (gpus == 0)
        #self.metric_logger = MetricLogger(metrics=metrics, module=self, gpu=gpu_mode)
        self.label_names = ["-3", "-2", "-1", "0", "1", "2", "3"]
        
        if scale == 2:
            self.label_names = ["0","1"]
        elif scale == 3:
            self.label_names = ["-1", "0", "1"]
        
        mask = self.convert_to_list(cols)
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=preprocess, use_sequence=get_sequence_wise, data_augmentation=augmentation, sequence_size=sequence_size, cols=mask, downsample=skip, forecasting=forecasting, scale=scale),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=preprocess, use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask, downsample=skip, forecasting=forecasting, scale=scale),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=worker, 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", preprocess=preprocess, use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask, forecasting=forecasting),
                                                batch_size=1, 
                                                shuffle=True, 
                                                num_workers=worker, 
                                                pin_memory=True)
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
        
        num_features = len(mask)-1 #-1 to neglect labels
        num_categories = scale #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = RNN(num_features, num_categories, hidden_dim=hidden, n_layers=layers, dropout=dropout).cuda()#; self.acc_train = self.acc_train.cuda(); self.acc_val = self.acc_val.cuda();self.acc_test= self.acc_test.cuda()
        else: self.model = RNN(num_features, num_categories, hidden_dim=hidden, n_layers=layers, dropout=dropout)

    def convert_to_list(self, config_string):
        """
            Takes an input string that contains a list and turns it into a regular python list.
            
            Args:
                config_string: the string specified in the run script (contains all training params)
        """
        trimmed_brackets = config_string[1:len(config_string)-1]
        idx = trimmed_brackets.split(",")
        r = []
        for num in idx:
            r.append(int(num))
        #print(r)
        return r
        
    def configure_optimizers(self):
        """
            Sets up optmizers and defines learning rate decay.
        """
        train_param = self.model.parameters()
        # Training parameters
        optimizer = torch.optim.Adam(train_param, lr=self.hparams.learning_rate)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(0.9999999, self.global_step)),
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

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader  
    
    def test_dataloader(self):
        return self.test_loader                                          

    def training_step(self, batch, batch_idx):
        """
            Defines what happens during one training step.
            
            Args:
                batch: the current training batch that is used
                batch_idx: the id of the batch
        """
        if batch_idx == 0: self.acc_train.reset(), self.train_preds.clear(), self.train_labels.clear()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
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
        
        preds, y = self.prepare_cfm_data(y_hat, y)
        #print(int(preds[0]))
        # # print(self.label_names[y[0]])
        self.train_preds.append(self.label_names[int(preds[0])])
        self.train_labels.append(self.label_names[int(y[0])])
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx, name="val"):
        """
            Defines what happens during one validation step.
            
            Args:
                batch: the current validation batch that is used
                batch_idx: the id of the batch
        """
        if batch_idx == 0: self.acc_val.reset(), self.val_preds.clear(), self.val_labels.clear()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        else: y = y.float()
        loss = self.criterion(y_hat, y)
        accuracy = self.acc_val(y_hat, y)
        self.log("{}_loss".format(name), loss, prog_bar=True, logger=True)
        self.log("{}_acc".format(name), accuracy, prog_bar=True, logger=True)
       
        
        preds, y = self.prepare_cfm_data(y_hat, y)
        self.val_preds.append(self.label_names[int(preds[0])])
        self.val_labels.append(self.label_names[int(y[0])])
        
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, "test")
        
    def on_validation_end(self):
        """
            Defines what happens after validation is done for one epoch.
        """
        if len(self.train_preds) > 0:
            compute_confusion_matrix(self.train_preds, self.train_labels, self.label_names, self.current_epoch, self, "Training")
        if len(self.val_preds) > 0:
            compute_confusion_matrix(self.val_preds, self.val_labels, self.label_names, self.current_epoch, self, "Validation")
    
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
