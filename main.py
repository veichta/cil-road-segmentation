import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchsummary import summary
from tqdm import tqdm

from models.base_unet import accuracy_fn, patch_accuracy_fn
from utils.utils import parse_arguments


def load_data_info_with_split(args):
    """Apply validation split to dataset.

    Args:
        args: Arguments.

    Returns:
        pd.Dataframe: Dataset info.
    """
    dataset_info = pd.read_csv(os.path.join(args.data_path, "dataset.csv"))

    cil_data_info = dataset_info[dataset_info["dataset"] == "CIL"]
    # set splits for CIL dataset
    for idx, row in cil_data_info.iterrows():
        if np.random.rand() <= args.val_split:
            row["split"] = "val"

    if args.datasets == "all":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
    elif args.datasets == "cil":
        dataset_info = cil_data_info

    train_df = dataset_info[dataset_info["split"] == "train"]
    val_df = dataset_info[dataset_info["split"] == "val"]
    return train_df, val_df


def main(args):
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(vars(args))

    train_df, val_df = load_data_info_with_split(args)

    if args.model == "unet":
        from datasets.base_dataset import BaseDataset
        from models.base_unet import UNet, evaluate_model, train_one_epoch

        # create train and val datasets
        train_dataset = BaseDataset(
            dataframe=train_df, use_patches=False, target_size=(384, 384), args=args
        )
        val_dataset = BaseDataset(
            dataframe=val_df, use_patches=False, target_size=(384, 384), args=args
        )

        # create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # create model
        model = UNet().to(args.device)
        summary(model, input_size=(args.batch_size, 384, 384))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == "spin":
        from models.hourglas_spin import HourglassNet, evaluate_model, train_one_epoch

        # from datasets ...
        from utils.loss import CrossEntropyLoss2d, mIoULoss

        criterion = [mIoULoss(torch.ones(2), 2), CrossEntropyLoss2d(torch.ones(37))]

        # init dataloader, model, optimizer and metrics

    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}

    history = {}
    for epoch in range(args.num_epochs):
        metrics = train_one_epoch(
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metric_fns=metric_fns,
            epoch=epoch,
            args=args,
        )
        history = evaluate_model(
            val_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            history=history,
            epoch=epoch,
            metrics=metrics,
        )


if __name__ == "__main__":
    args = parse_arguments()
    # setup logging
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
        filename=f"./{args.log_dir}/{args.model}-{timestamp}.log",
    )

    main(args)
