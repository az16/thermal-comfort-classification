from typing_extensions import final
from unicodedata import normalize
import torch
from torchvision import transforms
import torchmetrics.functional as plf
import pytorch_lightning as pl
from torch import cuda
from network.tc_rnn import RNN
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.path import *
from metrics import MetricLogger


gpu_mode=False
# RDM_Net.use_cuda=False
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, inputs, types, gpus, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        gpu_mode = not (gpus == 0)
        self.metric_logger = MetricLogger(metrics=metrics, module=self, gpu=gpu_mode)
        
        mask = self.convert_to_list(types)
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=False, use_sequence=get_sequence_wise, sequence_size=sequence_size, use_demographic=mask[0], use_imgs=mask[1], use_pmv_vars=mask[2], use_physiological=mask[3]),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=False, use_sequence=get_sequence_wise, sequence_size=sequence_size, use_demographic=mask[0], use_imgs=mask[1], use_pmv_vars=mask[2], use_physiological=mask[3]),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=worker, 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", preprocess=False, use_sequence=get_sequence_wise, sequence_size=sequence_size, use_demographic=mask[0], use_imgs=mask[1], use_pmv_vars=mask[2], use_physiological=mask[3]),
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=worker, 
                                                pin_memory=True)
        self.criterion = torch.nn.NLLLoss()
        
        
        hidden_state_size = 256
        sequence_size = sequence_size
        num_features = inputs
        num_categories = 7 #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = RNN(num_features, num_categories, n_layers=1, hidden_dim=hidden_state_size, dropout=dropout).cuda()
        else: self.model = RNN(num_features, num_categories, n_layers=1, hidden_dim=hidden_state_size, dropout=dropout)
        
        # if is_cuda:
        #     self.model = DepthEstimationNet(self.config).cuda()
        # else:
        #     self.model = DepthEstimationNet(self.config)

    def convert_to_list(self, config_string):
        # trimmed_brackets = config_string[1:len(config_string)-1]
        # idxs = trimmed_brackets.split(",")
        
        for num in config_string:
            num = (1 == num)
        return config_string
        

    def configure_optimizers(self):
        train_param = self.model.parameters()
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_accuracy(AVG)'
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        
        out = self.model(x)

        return out

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader  
    
    def test_dataloader(self):
        return self.test_loader                                          

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
         
        if gpu_mode: y_hat = y_hat.cuda()  
        # print("pred: {0}:".format(y_hat))
        # print("target: {0}:".format(y))
        loss = self.criterion(y_hat, y)
        # print(loss)
        # self.log("train_{}".format("NLLLoss"), loss, prog_bar=True)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()

        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        
        if gpu_mode: y_hat = y_hat.cuda()
        # loss = self.criterion(y_hat, y)
        # self.log("validation_{}".format("NLLLoss"), loss, prog_bar=True)
        return self.metric_logger.log_val(y_hat, y)
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()

        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        
        if gpu_mode: y_hat = y_hat.cuda()
        # loss = self.criterion(y_hat, y)
        # self.log("test_{}".format("NLLLoss"), loss, prog_bar=True)
        return self.metric_logger.log_test(y_hat, y)
    