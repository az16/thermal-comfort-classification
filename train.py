import sys
import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module
from dataloaders.tc_dataloader import TC_Dataloader
from dataloaders.utils import Feature, SCALARS

"""
    In this the training flags are defined. Training modules have to be included here so that flags can be passed when modules are called
"""


if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=32,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=1, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=50, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')

    parser.add_argument('--name', default=None, help="Name of the train run")
    parser.add_argument('--dataset_path', required=True, help="Path to ThermalDataset")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.99999, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence_window', type=int, default=0, help="Use thermal comfort dataset sequentially.")
    parser.add_argument('--columns', default=["Radiation-Temp", "Ambient_Temperature", "Ambient_Humidity", "Label"], nargs='+', help='The number of variables used for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='Model dropout rate')
    parser.add_argument('--hidden',type=int, default=128, help='Hidden states in LSTM')
    parser.add_argument('--image_path', default='/mnt/hdd/albin_zeqiri/ma/dataset/rgb/tcs_study/', help='Path to training images')
    parser.add_argument('--layers', type=int, default=2, help='Hidden layers')
    parser.add_argument('--preprocess', action='store_true', help='Make dataloaders perform data cleaning and normalization')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted cross entropy loss.')
    parser.add_argument('--data_augmentation', action='store_true', help='Do data augmentation on csv features.')
    parser.add_argument('--skiprows', type=int, default=26, help='How many rows to skip while reading data lines')
    parser.add_argument('--forecasting',  type=int, default=0, help='Use forecasting labels.')
    parser.add_argument('--scale',  type=int, default=7, help='Use forecasting labels.')
    parser.add_argument('--loss', default='wce', type=str, help='Loss function to use.')
    parser.add_argument('--latent_size', default=64, type=int, help='Latent vector size.')
    
    
    args = parser.parse_args()

    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    callbacks = []

    if args.dev: args.name = None
    if args.loss == 'mse': args.use_weighted_loss = False
    if Feature.ALL.value in args.columns:
        args.columns = SCALARS + [Feature.LABEL]

    if args.name:
        callbacks += [pl.callbacks.lr_monitor.LearningRateMonitor()]
    
        # Checkpoint callback to save best model parameters
        
        callbacks += [pl.callbacks.ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            filename='{epoch}-{valid_acc}',
            monitor='valid_acc',
            mode='max'
        )]



    use_gpu = not args.gpus == 0
    sequence_based = (args.sequence_window > 0)


    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        gpus=args.gpus,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        enable_model_summary=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.WandbLogger(project="ThermalComfort", name=args.name) if args.name and not args.dev else None,
        callbacks=callbacks
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })
    

    train_loader = torch.utils.data.DataLoader(TC_Dataloader(args.dataset_path, split="training", preprocess=args.preprocess, use_sequence=sequence_based, data_augmentation=args.data_augmentation, sequence_size=args.sequence_window, cols=args.columns, downsample=args.skiprows, forecasting=args.forecasting, scale=args.scale),
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(TC_Dataloader(args.dataset_path, split="validation", preprocess=True, use_sequence=sequence_based, data_augmentation=False, sequence_size=args.sequence_window, cols=args.columns, downsample=args.skiprows, forecasting=args.forecasting, scale=args.scale),
                                                batch_size=1, 
                                                shuffle=False, 
                                                num_workers=args.worker, 
                                                pin_memory=True)     
    
    model = TC_RNN_Module(args)
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
