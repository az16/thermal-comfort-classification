from msilib.schema import Feature
import sys
from turtle import forward
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from network.rnn_module import RandomGuess, TC_RNN_Module, Oracle
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.utils import Feature
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

"""
    In this the training flags are defined. Training modules have to be included here so that flags can be passed when modules are called
"""

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')

    parser.add_argument('--dataset_path', required=True, help="Path to ThermalDataset")
    parser.add_argument('--ckpt',  type=str, required=True, help='Ckpt to evaluate.')
    
    
    args = parser.parse_args()

    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    use_gpu = not args.gpus == 0


    trainer = pl.Trainer(
        gpus=args.gpus,
    )

    if args.ckpt == "oracle":
        model = Oracle()
    if args.ckpt == "random":
        model = RandomGuess()
    else:
        print("Loading checkpoint: ", args.ckpt)
        model = TC_RNN_Module.load_from_checkpoint(args.ckpt)

    test_loader = torch.utils.data.DataLoader(TC_Dataloader(args.dataset_path, split="training", preprocess=True, data_augmentation=False, sequence_size=30, cols=Feature.BEST, downsample=5, scale=7),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)    
    
    
    
    #result = trainer.test(model=model, dataloaders=test_loader, verbose=False)
    #print(result)
    data = trainer.predict(model=model, dataloaders=test_loader)
    gt = [g.item() for (g, p) in data]
    pred = [torch.argmax(p).item() for (g, p) in data]
    
    ConfusionMatrixDisplay.from_predictions(gt, pred)
    plt.show()
