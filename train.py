import sys
import random
import torch
from dataloaders.path import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module
from network.regression_module import TC_MLP_Module
from network.rcnn_module import TC_RCNN_Module

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=1, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=1, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--metrics', default=['accuracy'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence_window', type=int, default=0, help="Use thermal comfort dataset sequentially.")
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='Model dropout rate')
    parser.add_argument('--hidden',type=int, default=128, help='Hidden states in LSTM')
    parser.add_argument('--image_path', default='/mnt/hdd/albin_zeqiri/ma/dataset/rgb/tcs_study/', help='Path to training images')
    parser.add_argument('--layers', type=int, default=2, help='Hidden layers')
    parser.add_argument('--preprocess', action='store_true', help='Make dataloaders perform data cleaning and normalization')
    parser.add_argument('--data_augmentation', action='store_true', help='Do data augmentation on csv features.')
    parser.add_argument('--skiprows', type=int, default=26, help='How many rows to skip while reading data lines')



    args = parser.parse_args()

    if args.detect_anomaly: # for debugging
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)

    # windows safe
    if sys.platform in ["win32"]:
        args.worker = 0

    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)
    
    # Checkpoint callback to save best model parameters
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=1,
        dirpath="./checkpoints/{0}".format(args.module),
        filename='best-performance',
        monitor='val_acc',
        mode='max'
    )

    use_gpu = not args.gpus == 0
    sequence_based = (args.sequence_window > 0)


    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        profiler="simple",
        gpus=args.gpus,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        amp_backend='apex',
        enable_model_summary=True,
        min_epochs=args.min_epochs,
        # limit_train_batches=0.001,
        # limit_val_batches=0.001,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("tensorboard_logs", name=args.module),
        callbacks=[pl.callbacks.lr_monitor.LearningRateMonitor(), checkpoint_callback]
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })
    
    preprocessing = False
    augmentation = False
    if args.preprocess: preprocessing = True
    if args.data_augmentation: augmentation = True
    # choose module with respective network architecture here based on parser argument
    assert not args.module == ''; "Pass the module you would like to use as a parser argument to commence training." 
    tc_module = None 
    if args.module == "regression": tc_module = TC_MLP_Module(Path.db_root_dir("tcs"), args.batch_size, args.learning_rate, args.worker, args.metrics, sequence_based, args.sequence_window, args.columns, args.gpus, args.dropout, preprocessing, augmentation)
    elif args.module == "rnn": tc_module = TC_RNN_Module(Path.db_root_dir("tcs"), args.batch_size, args.learning_rate, args.worker, args.metrics, sequence_based, args.sequence_window, args.columns, args.gpus, args.dropout, args.hidden, args.layers, preprocessing, augmentation, args.skiprows)
    elif args.module == "rcnn": tc_module = TC_RCNN_Module(Path.db_root_dir("tcs"), args.batch_size, args.learning_rate, args.worker, args.metrics, sequence_based, args.sequence_window, args.columns, args.gpus, args.dropout, args.hidden, args.layers, args.image_path, preprocessing, augmentation) 
    print(args)
    print("Using {0} lightning module.".format(args.module.upper()))
    #print(tc_module)
    if args.find_learning_rate:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(tc_module)
        suggested_lr = lr_finder.suggestion()
        print("Old learning rate: ", args.learning_rate)
        args.learning_rate = suggested_lr
        print("Suggested learning rate: ", args.learning_rate)
    else:
        #train and test afterwards (uncomment testing if not enough data is available)
        trainer.fit(tc_module)
        #trainer.test(tc_module, verbose=True)
