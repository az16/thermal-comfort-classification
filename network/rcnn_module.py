from multiprocessing import cpu_count
import numpy as np
import torch
import pytorch_lightning as pl
from network.learning_models import RCNN
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.path import *
from dataloaders.utils import order2class, class7To3, class7To2
from torchmetrics import Accuracy as TopK

"""
    The training module for the CNN-LSTM architecture is defined here. If CNN-LSTM training is supposed to be adjusted, change this.
"""

gpu_mode=False
class TC_RCNN_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, cols, gpus, dropout, hidden, layers, im_path, preprocess, augmentation, skip, scale, *args, **kwargs):
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
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=preprocess, use_sequence=get_sequence_wise, data_augmentation=augmentation, sequence_size=sequence_size, use_imgs=True, image_path=im_path, cols=mask, downsample=skip, scale=scale),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=preprocess, use_sequence=get_sequence_wise, sequence_size=sequence_size, use_imgs=True,  image_path=im_path, cols=mask, downsample=skip, scale=scale),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", preprocess=preprocess, use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                batch_size=1, 
                                                shuffle=True, 
                                                num_workers=cpu_count(), 
                                                pin_memory=True)
        self.classification_loss = False
        
        self.criterion = torch.nn.MSELoss()
        self.accuracy = TopK(num_classes=7)
        self.accuracy_3 = TopK(num_classes=3)
        self.accuracy_2 = TopK(num_classes=2)
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        
        num_features = len(mask)-1 #-1 to neglect labels
        num_categories = scale #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = RCNN(num_features, num_categories, hidden=hidden, n_layers=layers, dropout=dropout).cuda()#; self.acc_train = self.acc_train.cuda(); self.acc_val = self.acc_val.cuda();self.acc_test= self.acc_test.cuda()
        else: self.model = RCNN(num_features, num_categories, hidden=hidden, n_layers=layers, dropout=dropout)

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
        rgb, seq = x
        seq = seq.float()
        # print(rgb.shape)
        out = self.model((rgb,seq))
        return out

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader  
    
    def predict_dataloader(self):
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
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        else: y = y.float()
        loss = self.criterion(y_hat, y)
        y_hat_class_label = torch.argmax(order2class(y_hat), dim=-1)
        y_class_label    = torch.argmax(order2class(y), dim=-1)

        accuracy = self.accuracy(y_hat_class_label, y_class_label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", accuracy, prog_bar=True, logger=True)
        
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx, name="val"):
        """
            Defines what happens during one validation step.
            
            Args:
                batch: the current validation batch that is used
                batch_idx: the id of the batch
        """
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)#torch.squeeze(torch.multiply(self(x), 3.0), dim=1)
        if gpu_mode: y_hat = y_hat.cuda()  
        if self.classification_loss:
            y = y.long()
        else: y = y.float()
        loss = self.criterion(y_hat, y)
        mse = self.mse(y_hat, y)
        l1  = self.l1(y_hat, y)

        y_hat_class_label = torch.argmax(order2class(y_hat), dim=-1)
        y_class_label    = torch.argmax(order2class(y), dim=-1)

        accuracy = self.accuracy(y_hat_class_label, y_class_label)
        accuracy2 = self.accuracy_2(class7To2(y_hat_class_label), class7To2(y_class_label))
        accuracy3 = self.accuracy_3(class7To3(y_hat_class_label), class7To3(y_class_label))
        
        self.log("{}_mse".format(name), mse, prog_bar=True, logger=True)
        self.log("{}_l1".format(name), l1, prog_bar=True, logger=True)
        self.log("{}_accuracy".format(name), accuracy, prog_bar=True, logger=True)
        self.log("{}_accuracy3".format(name), accuracy3, prog_bar=True, logger=True)
        self.log("{}_accuracy2".format(name), accuracy2, prog_bar=True, logger=True)
        return {
            "{}_mse".format(name): mse,
            "{}_l1".format(name): l1,
            "{}_accuracy7".format(name): accuracy,
            "{}_accuracy3".format(name): accuracy3,
            "{}_accuracy2".format(name): accuracy2,
            }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)

        y_hat = order2class(y_hat)
        y = order2class(y)

        return y_hat, y