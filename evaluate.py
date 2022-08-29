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

    args = parser.parse_args()

    checkpoints = []
    if Path(args.ckpt).is_file():
        checkpoints.append(args.ckpt)
    if Path(args.ckpt).is_dir():
        checkpoints = [ckpt.as_posix() for ckpt in Path(args.ckpt).glob("**/*") if ckpt.suffix == ".ckpt"]

    valid_files, test_files = [],[]
    trainer = pl.Trainer(gpus=-1)
    for i, ckpt in enumerate(checkpoints):
        
        mid = "+".join([header[int(c)] for c in torch.load(ckpt)["hyper_parameters"]['cols'].replace("[", "").replace("]", "").split(",")])
        target = Path(ckpt).parents[2]/mid
        new_path = Path(ckpt).parents[1].rename(target)

        ckpt = (new_path/"checkpoints"/Path(ckpt).name).as_posix()
        val_file = Path(ckpt).parent/"val.json"
        test_file = Path(ckpt).parent/"test.json"
        cm_file = Path(ckpt).parent/"cm.pdf"
        valid_files.append(val_file)
        test_files.append(test_file)
        
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

    
    data = {}
    for valid_file in valid_files:
        with open(valid_file, "r") as jsonfile:
            vdata = json.load(jsonfile)[0]
            data[valid_file.parents[1].name] = vdata
    for test_file in test_files:
        with open(test_file, "r") as jsonfile:
            tdata = json.load(jsonfile)[0]
            data[test_file.parents[1].name].update(tdata)

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
