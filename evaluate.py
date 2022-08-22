import sys
import random
import torch
from dataloaders.path import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module
from network.regression_module import TC_MLP_Module
from network.rcnn_module import TC_RCNN_Module

"""
    In this the training flags are defined. Training modules have to be included here so that flags can be passed when modules are called
"""


if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--ckpt', required=True)

    args = parser.parse_args()


    trainer = pl.Trainer(gpus=-1)
    module = TC_RNN_Module.load_from_checkpoint(args.ckpt, scale=7)
    trainer.test(model=module)