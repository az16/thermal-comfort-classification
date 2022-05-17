import pandas as pd
import torch
import pytorch_lightning as pl
from network.learning_models import MLP
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.path import *
from metrics import MetricLogger
import matplotlib.pyplot as plt
import seaborn as sns


gpu_mode=False

class TC_MLP_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, cols, gpus, dropout, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        gpu_mode = not (gpus == 0)
        self.metric_logger = MetricLogger(metrics=metrics, module=self, gpu=gpu_mode)
        
        mask = self.convert_to_list(cols)
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=True, use_sequence=get_sequence_wise, sequence_size=sequence_size, data_augmentation=True, cols=mask),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=True, use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=1, 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=1, 
                                                pin_memory=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        num_features = len(mask)-1 #-1 to neglect labels
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = MLP(sequence_size, 1).cuda()
        else: self.model = MLP(sequence_size, 1)

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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_loss'
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
        #print(x.shape)
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        # print(torch.squeeze(y_hat,dim=-1),y)
        if gpu_mode: y_hat = y_hat.cuda()  
        # print("pred: {0}:".format(torch.squeeze(y_hat.topk(1)[1])))
        # print("target: {0}:".format(y))
        loss = self.criterion(y_hat, y)
        
        # print(loss)
        #self.log("train_loss", loss, prog_bar=True)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()

        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = torch.squeeze(self(x), dim=1)
        
        if gpu_mode: y_hat = y_hat.cuda()
        loss = self.criterion(y_hat, y)
        
        #self.log("val_{}".format("loss"), loss, prog_bar=True)
        return self.metric_logger.log_val(y_hat, y, loss)
    
    # def validation_epoch_end(self, outputs):
    #     #print(outputs)
    #     preds = torch.cat([tmp['preds'] for tmp in outputs])
    #     targets = torch.cat([tmp['target'] for tmp in outputs])
    #     confusion_matrix = ConfusionMatrix(num_classes=7)
    #     if gpu_mode: confusion_matrix = ConfusionMatrix(num_classes=7).cuda(); preds = preds.cuda(); targets=targets.cuda()
    #     matrix = confusion_matrix(preds, targets).cuda()

    #     df_cm = pd.DataFrame(matrix.numpy(), index = range(7), columns=range(7))
    #     plt.figure(figsize = (10,7))
    #     fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
    #     plt.close(fig_)
        
    #     self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()

        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = torch.squeeze(self(x), dim=1)
        
        if gpu_mode: y_hat = y_hat.cuda()
        loss = self.criterion(y_hat, y)
        #self.log("test_{}".format("MSE"), loss, prog_bar=True)
        return self.metric_logger.log_test(y_hat, y, loss)