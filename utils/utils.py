import argparse
import math
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def weights_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Namespace: Namespace with arguments.
    """
    # fmt: off
    parser = argparse.ArgumentParser("CIL Road Segmentation")

    # Model setup
    parser.add_argument(
        "--model", 
        type=str, required=True, help="Model name"
    )
    parser.add_argument(
        "--backbone",
        type=str, help="Backbone name"
    )

    # Training setup
    parser.add_argument(
        "--val_split", 
        type=float, default=0.2, help="Validation split"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--augmentation", 
        type=int, default=0, help="Whether to use augmentations."
    )
    parser.add_argument(
        "--lr", 
        type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, default=6, help="Batch size"
    )
    parser.add_argument(
        "--grad_clip",
        type=float, default=0.1, help="Gradient clipping"
    )

    # Loss
    parser.add_argument(
        "--weight_miou", 
        type=float, default=1, help="Weight for road loss"
    )
    parser.add_argument(
        "--weight_vec", 
        type=float, default=1, help="Weight for orientation loss"
    )
    parser.add_argument(
        "--weight_topo",
        type=float, default=0, help="Weight for topo loss"
    )
    parser.add_argument(
        "--weight_bce",
        type=float, default=0, help="Weight for topo loss"
    )
    parser.add_argument(
        "--weight_dice",
        type=float, default=0, help="Weight for topo loss"
    )
    parser.add_argument(
        "--weight_focal",
        type=float, default=0, help="Weight for topo loss"
    )

    # Dataset setup
    parser.add_argument(
        "--data_path",
        type=str, default="./data/big-dataset", help="Path to csv file"
    )
    parser.add_argument(
        "--datasets",
        type=str, choices=["all", "cil", "cil-mrd", "cil-dg", "dg"], default="cil", help="Datasets to use"
    )
    parser.add_argument(
        "--min_pixels",
        type=int, default=None, help="Cut off samples with small pixel count"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, default=6, help="Number of workers"
    )

    # General setup
    parser.add_argument(
        "--device",
        type=str, default="cpu", help="Device to use"
    )
    parser.add_argument(
        "--seed", 
        type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--resume", 
        type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--start_epoch", 
        type=int, default=0, help="Resume from epoch"
    )
    parser.add_argument(
        "--save_dir",
        type=str, default="./checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, default="./logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--multi_scale_pred",
        default=True,
        type=str2bool,
        help="perform multi-scale prediction (default: True)",
    )
    # fmt: on
    return parser.parse_args()
