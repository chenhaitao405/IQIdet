#!/usr/bin/env python3
"""Validate F-Clip with count precision.
Usage:
    valid.py [options] <model-config> <ckpt>
    valid.py (-h | --help )

Arguments:
   <model-config>                  Path to model yaml config
   <ckpt>                          Path to ckpt

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   --params <file>                 Params yaml file [default: params.yaml]
   --output_dir <dir>              Output directory for logs/vis [default: logs/valid]
   --vis                           Save visualization results.
   --threshold <t>                 Heatmap threshold [default: 0.4]
"""

import os
from docopt import docopt
import torch

from FClip.config import C, M
from FClip.datasets import collate
from FClip.datasets import LineDataset as WireframeDataset
from FClip.config_loader import load_configs
from FClip.infer_utils import build_infer_model
from FClip.valid_utils import evaluate_count_precision


def main():
    args = docopt(__doc__)
    config_file = args["<model-config>"]
    ckpt = args["<ckpt>"]
    params_file = args["--params"]
    load_configs(model_yaml=config_file, params_yaml=params_file, ckpt=ckpt)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(C.io.datadir, split="valid", dataset=C.io.dataname),
        batch_size=C.model.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=C.io.num_workers,
        pin_memory=True,
    )

    model = build_infer_model(device)

    out_dir = args["--output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    vis = bool(args["--vis"])
    vis_dir = os.path.join(out_dir, "vis")
    threshold = float(args["--threshold"])

    image_getter = val_loader.dataset._get_im_name
    precision = evaluate_count_precision(
        model=model,
        val_loader=val_loader,
        device=device,
        threshold=threshold,
        vis=vis,
        vis_dir=vis_dir if vis else None,
        image_getter=image_getter if vis else None,
    )

    with open(os.path.join(out_dir, "precision.txt"), "w") as f:
        f.write(f"{precision:.6f}\n")
    print(f"count precision: {precision:.6f}")


if __name__ == "__main__":
    main()
