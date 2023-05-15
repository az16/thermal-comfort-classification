import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from network.rnn_module import TC_RNN_Module
from network.rcnn_module import TC_RCNN_Module

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import json
from dataloaders.utils import class7To2, class7To3, header

if __name__ == "__main__":
    import numpy as np
    
    parser = ArgumentParser('Evaluate model.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--sort_table', action='store_true')
    parser.add_argument('--overwrite_valid', action='store_true')
    parser.add_argument('--overwrite_cm', action='store_true')
    parser.add_argument('--rename_dir', action='store_true')

    args = parser.parse_args()

    
    if args.module == "rnn":
        Module = TC_RNN_Module
    elif args.module == "rcnn":
        Module = TC_RCNN_Module
    else:
        raise NotImplementedError(args.module)

    checkpoints = []
    if Path(args.ckpt).is_file():
        checkpoints.append(args.ckpt)
        parent = Path(args.ckpt).parent
    elif Path(args.ckpt).is_dir():
        checkpoints = [ckpt.as_posix() for ckpt in Path(args.ckpt).glob("**/*") if ckpt.suffix == ".ckpt"]
        parent = Path(args.ckpt)
    else:
        raise ValueError(args.ckpt)

    valid_files = []
    trainer = pl.Trainer(gpus=1)
    for i, ckpt in enumerate(checkpoints):
        print("Validating checkpoint {}".format(ckpt))
        mid = "+".join([header[int(c)] for c in torch.load(ckpt)["hyper_parameters"]['cols'].replace("[", "").replace("]", "").split(",")])
        target = Path(ckpt).parents[2]/mid
        if args.rename_dir:
            new_path = Path(ckpt).parents[1].rename(target)
            ckpt = (new_path/"checkpoints"/Path(ckpt).name).as_posix()
        val_file = Path(ckpt).parent/"val.json"
        cm_file = Path(ckpt).parent/"cm.pdf"
        pred_file = Path(ckpt).parent/"preds.npy"
        gt_file = Path(ckpt).parent/"gt.npy"
        valid_files.append(val_file)
    
        if all([val_file.exists(), cm_file.exists()]) and not args.overwrite_valid and not args.overwrite_cm: continue
        module = None
        if not val_file.exists() or args.overwrite_valid:
            print("Validating ckpt: ", ckpt)
            module = Module.load_from_checkpoint(ckpt, scale=7, path=args.path, dataset="thermal_comfort")
            val_result = trainer.validate(model=module)
            with open(val_file.as_posix(), "w") as jsonfile:
                json.dump(val_result, jsonfile, indent=4)
        
        if not cm_file.exists() or args.overwrite_cm:
            print("Create Confusion Matrix for ckpt: ", ckpt)
            if not pred_file.exists():
                if module is None:
                    module = Module.load_from_checkpoint(ckpt, scale=7, path=args.path, dataset="thermal_comfort")
                result = trainer.predict(model=module)
                pred = np.array([torch.argmax(p).item() for (p,_) in result], dtype=int)
                gt   = np.array([torch.argmax(g).item() for (_,g) in result], dtype=int)

                np.save(pred_file.as_posix(), pred)
                np.save(gt_file.as_posix(), gt)
            else:
                pred = np.load(pred_file.as_posix())
                gt   = np.load(gt_file.as_posix())

            fig, ax = plt.subplots()
            ax.spines["left"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["top"].set_color("white")
            ax.spines["bottom"].set_color("white")

            ConfusionMatrixDisplay.from_predictions(gt, pred, normalize='true', display_labels=["cold", "cool", "slightly cool", "comfortable", "slightly warm", "warm", "hot"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
            plt.savefig(cm_file.as_posix(),bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots()
            ax.spines["left"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["top"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ConfusionMatrixDisplay.from_predictions(class7To3(gt), class7To3(pred), normalize='true', display_labels=["cool", "comfortable", "warm"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
            plt.savefig(cm_file.as_posix().replace("cm.pdf", "cm3.pdf"),bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots()
            ax.spines["left"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.spines["top"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ConfusionMatrixDisplay.from_predictions(class7To2(gt), class7To2(pred), normalize='true', display_labels=["uncomfortable","comfortable"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
            plt.savefig(cm_file.as_posix().replace("cm.pdf", "cm2.pdf"),bbox_inches='tight', pad_inches=0)
    

    #if args.rename_dir:
    #    for i, ckpt in enumerate(checkpoints):
    #        src  = Path(ckpt).parents[1]
    #        hparams = torch.load(ckpt)
    #        tgt = "+".join([header[int(c)] for c in hparams['hyper_parameters']['cols'].replace("]", "").replace("[", "").split(",")][0:-1])
    #        tgt = Path(Path(ckpt).parents[2], tgt)
    #        print("Renaming ckpt {} to {}".format(src, tgt)) 
    #        src.rename(tgt)
    
    data = {}
    for valid_file in valid_files:
        with open(valid_file, "r") as jsonfile:
            vdata = json.load(jsonfile)[0]
            data[valid_file.parents[1].name] = vdata
    
    items = data.items()
    if args.sort_table:
        items = sorted(data.items(), key=lambda x:x[1]['val_accuracy'], reverse=True)
    
    table = ["{BT} & {PCE} & {HR} & {GSR} & {AT} & {AH} & {RT} & {WS} & {val_acc:.1f}\% & {val_3_acc:.1f}\% & {val_2_acc:.1f}\% & {mse:.3f} & {l1:.3f}\\\\\n".format(
        GSR="X" if "GSR" in key else "", 
        AT="X" if "Ambient_Temperature" in key else "",
        AH="X" if "Ambient_Humidity" in key else "",
        RT="X" if "Radiation-Temp" in key else "",
        WS="X" if "Wrist_Skin_Temperature" in key else "",
        BT="X" if "Bodytemp" in key else "",
        PCE="X" if "PCE-Ambient-Temp" in key else "",
        HR="X" if "Heart_Rate" in key else "",
        val_acc  =100*round(value['val_accuracy'], 3),
        val_3_acc=100*round(value['val_accuracy3'],3),
        val_2_acc=100*round(value['val_accuracy2'],3),
        mse      =    round(value['val_mse'],3),
        l1       =    round(value["val_l1"],3)
        ) for (key, value) in items]

    #for row in table:
    #    print(row)
    
    with open((parent/"table.tex").as_posix(), "w") as txtfile:
        txtfile.writelines(table)
    
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

    #print(table)

    table = "{val_acc:.1f}\% & {val_3_acc:.1f}\% & {val_2_acc:.1f}\% & {mse:.3f} & {l1:.3f}\\\\".format(
        val_acc  =100*round(std_val_acc, 3),
        val_3_acc=100*round(std_val_acc3,3),
        val_2_acc=100*round(std_val_acc2,3),
        mse      =    round(std_val_mse,3),
        l1       =    round(std_val_l1,3)
        )

    #print(table)
    