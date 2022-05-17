from multiprocessing import cpu_count
from torchmetrics import Accuracy
import pandas as pd
import torch
from sklearn.metrics import *
import pytorch_lightning as pl
from network.learning_models import RNN
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.path import *
import matplotlib.pyplot as plt
import seaborn as sns


gpu_mode=False
# RDM_Net.use_cuda=False
class TC_RNN_Module(pl.LightningModule):
    def __init__ (self, path, batch_size, learning_rate, worker, metrics, get_sequence_wise, sequence_size, cols, gpus, dropout, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        gpu_mode = not (gpus == 0)
        #self.metric_logger = MetricLogger(metrics=metrics, module=self, gpu=gpu_mode)
        self.label_names = ["-3", "-2", "-1", "0", "1", "2", "3"]
        
        mask = self.convert_to_list(cols)
        self.train_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="training", preprocess=True, use_sequence=get_sequence_wise, sequence_size=sequence_size, data_augmentation=True, cols=mask),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="validation", preprocess=True, use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=cpu_count(), 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(TC_Dataloader(path, split="test", use_sequence=get_sequence_wise, sequence_size=sequence_size, cols=mask),
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=cpu_count(), 
                                                pin_memory=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()
        self.val_preds = []
        self.val_labels = []
        
        num_features = len(mask)-1 #-1 to neglect labels
        num_categories = 7 #Cold, Cool, Slightly Cool, Comfortable, Slightly Warm, Warm, Hot
        print("Use GPU: {0}".format(gpu_mode))
        if gpu_mode: self.model = RNN(num_features, num_categories).cuda(); self.acc_train = self.acc_train.cuda(); self.acc_val = self.acc_val.cuda();self.acc_test= self.acc_test.cuda()
        else: self.model = RNN(num_features, num_categories)

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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
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
        if batch_idx == 0: self.acc_train.reset()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        if gpu_mode: y_hat = y_hat.cuda()  
        
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.acc_train(preds, y)
        accuracy = self.acc_train.compute()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.acc_val.reset(), self.val_preds.clear(), self.val_labels.clear()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        if gpu_mode: y_hat = y_hat.cuda()  
        
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.acc_val(preds, y)
        accuracy = self.acc_val.compute()
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        
        preds = preds.cpu()
        y = y.cpu().long()
        # print(self.label_names[preds[0]])
        # print(self.label_names[y[0]])
        self.val_preds.append(self.label_names[preds[0]])
        self.val_labels.append(self.label_names[y[0]])
        
        return {"loss": loss, "accuracy": accuracy}

    def on_validation_epoch_end(self):
        print(self.val_labels)
        print(self.val_preds)
        cfm = confusion_matrix(self.val_labels, self.val_preds, labels=self.label_names)
        #print(cfm)
        df = pd.DataFrame(cfm, index=self.label_names, columns=self.label_names)
        
        #visualization
        
        m = sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
        m.set_yticklabels(m.get_yticklabels(), rotation=0, ha='right', size=10)
        m.set_xticklabels(m.get_xticklabels(), rotation=30, ha='right', size=10)
        plt.ylabel('Target Labels')
        plt.xlabel('Predicted Label')
        fig = m.get_figure()
        #plt.close(fig)
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.acc_test.reset()
        x, y = batch
        if gpu_mode: x, y = x.cuda(), y.cuda()
        
        y_hat = self(x)
        if gpu_mode: y_hat = y_hat.cuda()  
        
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.acc_test(preds, y)
        accuracy = self.acc_test.compute()
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": accuracy}
    