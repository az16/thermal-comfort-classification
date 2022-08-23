import sys
import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module
from dataloaders.tc_dataloader import TC_Dataloader

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

    model = TC_RNN_Module.load_from_checkpoint(args.ckpt)

    test_loader = torch.utils.data.DataLoader(TC_Dataloader(args.dataset_path, stride=1, split="test", preprocess=True, use_sequence=model.opt.sequence_window>0, data_augmentation=False, sequence_size=model.opt.sequence_window, cols=model.opt.columns, downsample=model.opt.skiprows, forecasting=model.opt.forecasting, scale=model.opt.scale),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)    
    
    
    
    result = trainer.test(model=model, dataloaders=test_loader, verbose=True)
    print(result)
