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
    import numpy as np
    
    parser = ArgumentParser('Evaluate model.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--overwrite_valid', action='store_true')

    args = parser.parse_args()

    checkpoints = []
    if Path(args.ckpt).is_file():
        checkpoints.append(args.ckpt)
    if Path(args.ckpt).is_dir():
        checkpoints = [ckpt.as_posix() for ckpt in Path(args.ckpt).glob("**/*") if ckpt.suffix == ".ckpt"]

    valid_files = []
    trainer = pl.Trainer(gpus=-1)
    for i, ckpt in enumerate(checkpoints):
        #mid = "+".join([header[int(c)] for c in torch.load(ckpt)["hyper_parameters"]['cols'].replace("[", "").replace("]", "").split(",")])
        #target = Path(ckpt).parents[2]/mid
        #new_path = Path(ckpt).parents[1].rename(target)

        #ckpt = (new_path/"checkpoints"/Path(ckpt).name).as_posix()
        val_file = Path(ckpt).parent/"val.json"
        cm_file = Path(ckpt).parent/"cm.pdf"
        valid_files.append(val_file)
        
        if all([val_file.exists(), cm_file.exists()]) and not args.overwrite_valid: continue
        print("Evaluating ckpt: ", ckpt)
        module = TC_RNN_Module.load_from_checkpoint(ckpt, scale=7, path=args.path, dataset="thermal_comfort")

        if ~val_file.exists() or args.overwrite_valid:
            val_result = trainer.validate(model=module)
            with open(val_file.as_posix(), "w") as jsonfile:
                json.dump(val_result, jsonfile, indent=4)
        
        if not cm_file.exists():
            result = trainer.predict(model=module)
            pred = [torch.argmax(p).item() for (p,g) in result]
            gt = [torch.argmax(g).item() for (p,g) in result]

            ConfusionMatrixDisplay.from_predictions(gt, pred)
            plt.savefig(cm_file.as_posix())

    """
    data = {}
    for valid_file in valid_files:
        with open(valid_file, "r") as jsonfile:
            vdata = json.load(jsonfile)[0]
            data[valid_file.parents[1].name] = vdata
    """
    """
    table = ["{GSR} & {AT} & {AH} & {RT} & {WS} & {val_acc:.1f}\% & {val_3_acc:.1f}\% & {val_2_acc:.1f}\% & {mse:.3f} & {l1:.3f}\\\\".format(
        GSR="X" if "GSR" in key else "", 
        AT="X" if "Ambient_Temperature" in key else "",
        AH="X" if "Ambient_Humidity" in key else "",
        RT="X" if "Radiation-Temp" in key else "",
        WS="X" if "Wrist_Skin_Temperature" in key else "",
        val_acc  =100*round(value['val_accuracy'], 3),
        val_3_acc=100*round(value['val_accuracy3'],3),
        val_2_acc=100*round(value['val_accuracy2'],3),
        mse      =    round(value['val_mse'],3),
        l1       =    round(value["val_l1"],3)
        ) for (key, value) in sorted(data.items(), key=lambda x:x[1]['val_accuracy'], reverse=True)]

    for row in table:
        print(row)
    """
    
    """
    mean_val_acc  = np.mean([v['val_accuracy'] for v in data.values()])
    mean_val_acc3 = np.mean([v['val_accuracy3'] for v in data.values()])
    mean_val_acc2 = np.mean([v['val_accuracy2'] for v in data.values()])
    mean_val_mse  = np.mean([v['val_mse'] for v in data.values()])
    mean_val_l1   = np.mean([v['val_l1'] for v in data.values()])

    std_val_acc  = np.std([v['val_accuracy'] for v in data.values()])
    std_val_acc3 = np.std([v['val_accuracy3'] for v in data.values()])
    std_val_acc2 = np.std([v['val_accuracy2'] for v in data.values()])
    std_val_mse  = np.std([v['val_mse'] for v in data.values()])
    std_val_l1   = np.std([v['val_l1'] for v in data.values()])

    table = "{val_acc:.1f}\% & {val_3_acc:.1f}\% & {val_2_acc:.1f}\% & {mse:.3f} & {l1:.3f}\\\\".format(
        val_acc  =100*round(mean_val_acc, 3),
        val_3_acc=100*round(mean_val_acc3,3),
        val_2_acc=100*round(mean_val_acc2,3),
        mse      =    round(mean_val_mse,3),
        l1       =    round(mean_val_l1,3)
        )

    print(table)

    table = "{val_acc:.1f}\% & {val_3_acc:.1f}\% & {val_2_acc:.1f}\% & {mse:.3f} & {l1:.3f}\\\\".format(
        val_acc  =100*round(std_val_acc, 3),
        val_3_acc=100*round(std_val_acc3,3),
        val_2_acc=100*round(std_val_acc2,3),
        mse      =    round(std_val_mse,3),
        l1       =    round(std_val_l1,3)
        )

    print(table)
    """