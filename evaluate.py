import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Evaluate model.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--path', required=True)

    args = parser.parse_args()

    trainer = pl.Trainer(gpus=-1)
    module = TC_RNN_Module.load_from_checkpoint(args.ckpt, scale=7, path=args.path)
    trainer.validate(model=module)
    result = trainer.predict(model=module)
    pred = [torch.argmax(p).item() for (p,g) in result]
    gt = [torch.argmax(g).item() for (p,g) in result]

    ConfusionMatrixDisplay.from_predictions(gt, pred)
    plt.show()