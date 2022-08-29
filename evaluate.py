import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import json
from dataloaders.utils import header

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Evaluate model.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--path', required=True)

    args = parser.parse_args()

    checkpoints = []
    if Path(args.ckpt).is_file():
        checkpoints.append(args.ckpt)
    if Path(args.ckpt).is_dir():
        checkpoints = [ckpt.as_posix() for ckpt in Path(args.ckpt).glob("**/*") if ckpt.suffix == ".ckpt"]

    trainer = pl.Trainer(gpus=-1)
    for i, ckpt in enumerate(checkpoints):
        val_file = Path(ckpt).parent/"val.json"
        test_file = Path(ckpt).parent/"test.json"
        cm_file = Path(ckpt).parent/"cm.pdf"
        if all([test_file.exists(), val_file.exists(), cm_file.exists()]): continue
        print("Evaluating ckpt: ", ckpt)
        module = TC_RNN_Module.load_from_checkpoint(ckpt, scale=7, path=args.path)
        
        if not test_file.exists():
            test_result = trainer.test(model=module)
            with open(test_file.as_posix(), "w") as jsonfile:
                json.dump(test_result, jsonfile, indent=4)

        if not val_file.exists():
            val_result = trainer.validate(model=module)
            with open(val_file.as_posix(), "w") as jsonfile:
                json.dump(val_result, jsonfile, indent=4)
        
        if not cm_file.exists():
            result = trainer.predict(model=module)
            pred = [torch.argmax(p).item() for (p,g) in result]
            gt = [torch.argmax(g).item() for (p,g) in result]

            ConfusionMatrixDisplay.from_predictions(gt, pred)
            plt.savefig(cm_file.as_posix())

        mid = "+".join([header[int(c)] for c in module.hparams.cols.replace("[", "").replace("]", "").split(",")])
        target = Path(ckpt).parents[2]/mid
        Path(ckpt).parents[1].rename(target)
