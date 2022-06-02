from multiprocessing import cpu_count
from torchmetrics import Accuracy
from metrics import rmse, mae, compute_confusion_matrix
import pandas as pd
import torch
import pytorch_lightning as pl
from network.learning_models import RCNN
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.pmv_loader import PMV_Results
from dataloaders.path import *


gpu_mode=False
# RDM_Net.use_cuda=False
class TC_RCNN_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, cols, gpus, dropout, hidden, layers, im_path, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        gpu_mode = not (gpus == 0)
        #self.metric_logger = MetricLogger(metrics=metrics, module=self, gpu=gpu_mode)
        self.label_names = ["-3", "-2", "-1", "0", "1", "2", "3"]
        
        mask = self.convert_to_list(cols)
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=True, use_sequence=get_sequence_wise, data_augmentation=True, sequence_size=sequence_size, use_imgs=True, image_path=im_path, cols=mask),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=True, use_sequence=get_sequence_wise, sequence_size=sequence_size, use_imgs=True,  image_path=im_path, cols=mask),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                batch_size=1, 
                                                shuffle=True, 
                                                num_workers=cpu_count(), 
                                                pin_memory=True)
        #self.pmv_results = PMV_Results()
        self.classification_loss = True
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        
        num_features = len(mask)-1 #-1 to neglect labels
        num_categories = 7 #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = RCNN(num_features, num_categories, hidden=hidden, n_layers=layers, dropout=dropout).cuda()#; self.acc_train = self.acc_train.cuda(); self.acc_val = self.acc_val.cuda();self.acc_test= self.acc_test.cuda()
        else: self.model = RCNN(num_features, num_categories, hidden=hidden, n_layers=layers, dropout=dropout)

    def convert_to_list(self, config_string):
        trimmed_brackets = config_string[1:len(config_string)-1]
        idx = trimmed_brackets.split(",")
        r = []
        for num in idx:
            r.append(int(num))
        #print(r)
        return r
        
    def configure_optimizers(self):
        train_param = self.model.parameters()
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_acc'
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        rgb, seq = x
        seq = seq.float()
        # print(rgb.shape)
        out = self.model((rgb,seq))
        return out

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader  
    
    def test_dataloader(self):
        return self.test_loader                                          

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.acc_train.reset(), self.train_preds.clear(), self.train_labels.clear()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        #print(y_hat, y)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.acc_train(preds, y)
        accuracy = self.acc_train.compute()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", accuracy, prog_bar=True, logger=True)
        # self.log("train_rsme", rmse(y_hat, y), prog_bar=True, logger=True)
        # self.log("train_mae", mae(y_hat, y), prog_bar=True, logger=True)
        
        preds = preds.cpu()
        y = y.cpu().long()
        # print(self.label_names[preds[0]])
        # print(self.label_names[y[0]])
        self.train_preds.append(self.label_names[preds[0]])
        self.train_labels.append(self.label_names[y[0]])
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.acc_val.reset(), self.val_preds.clear(), self.val_labels.clear()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.acc_val(preds, y)
        accuracy = self.acc_val.compute()
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)
        # self.log("val_rsme", rmse(y_hat, y), prog_bar=True, logger=True)
        # self.log("val_mae", mae(y_hat, y), prog_bar=True, logger=True)
        
        preds = preds.cpu()
        y = y.cpu().long()
        # print(self.label_names[preds[0]])
        # print(self.label_names[y[0]])
        self.val_preds.append(self.label_names[preds[0]])
        self.val_labels.append(self.label_names[y[0]])
        
        return {"loss": loss}
        
    def on_validation_end(self):
        if len(self.train_preds) > 0:
            compute_confusion_matrix(self.train_preds, self.train_labels, self.label_names, self.current_epoch, self, "Training")
        if len(self.val_preds) > 0:
            compute_confusion_matrix(self.val_preds, self.val_labels, self.label_names, self.current_epoch, self, "Validation")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        loss = self.criterion(y_hat, y)
        # preds = torch.argmax(y_hat, dim=1)
        # self.acc_train(preds, y)
        # accuracy = self.acc_train.compute()
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_rsme", rmse(y_hat, y), prog_bar=True, logger=True)
        self.log("test_mae", mae(y_hat, y), prog_bar=True, logger=True)
        
        return {"loss": loss}
    