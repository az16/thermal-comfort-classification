import sys
import random
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

if __name__ == "__main__":
    
    module_dict = {"regression": None,
                   "rnn": None,
                   "cnn": None,
                   "rf":  None,
                   "custom": None,}
    
    
    parser = ArgumentParser('Trains mono depth estimation models')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=1, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=1, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequency_based', action='store_true', help="Use thermal comfort dataset sequentially.")
    parser.add_argument('--module', type=int, default=0, help='The network module to be used for training')

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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        save_top_k=1,
        filename='{epoch}-{val_delta1}',
        monitor='val_delta1',
        mode='max'
    )

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=True,
        gpus=args.gpus,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("result", name="rd"),
        callbacks=[pl.callbacks.lr_monitor.LearningRateMonitor(), checkpoint_callback]
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })
    
    # choose module with respective network architecture here based on parser argument
    assert not args.module == ''; "Pass the module you would like to use as a parser argument to commence training." 
    tc_module = module_dict[args.module]

    if args.find_learning_rate:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(tc_module)
        suggested_lr = lr_finder.suggestion()
        print("Old learning rate: ", args.learning_rate)
        args.learning_rate = suggested_lr
        print("Suggested learning rate: ", args.learning_rate)
    else:
        trainer.fit(tc_module)