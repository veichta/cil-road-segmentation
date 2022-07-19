import logging
import os
import time
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchsummary import summary

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
    total_start = timer()

    train_df, val_df = load_data_info_with_split(args)

    if args.model == "unet":
        logging.info("Using UNet.")
        from datasets.base_dataset import BaseDataset
        from models.base_unet import UNet, evaluate_model, train_one_epoch

        # create train and val datasets
        train_dataset = BaseDataset(
            dataframe=train_df, use_patches=False, target_size=(384, 384), args=args
        )
        val_dataset = BaseDataset(
            dataframe=val_df, use_patches=False, target_size=(384, 384), args=args
        )

        # create model
        model = UNet().to(args.device)
        summary(model, input_size=(args.batch_size, 384, 384))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == "spin":
        logging.info("Using Spin Model.")
        from datasets import road_dataset
        from models.hourglas_spin import HourglassNet, evaluate_model, train_one_epoch
        from utils.loss import CrossEntropyLoss2d, mIoULoss
        from utils.utils import weights_init

        target_size = (400, 400)

        model = HourglassNet().to(args.device)
        weights_init(model, args.seed)
        summary(model, input_size=(3, target_size[0], target_size[1]))

        train_dataset = road_dataset.RoadDataset(
            dataframe=train_df,
            target_size=target_size,
            multi_scale_pred=args.multi_scale_pred,
            is_train=True,
            args=args,
        )
        val_dataset = road_dataset.RoadDataset(
            dataframe=val_df,
            target_size=target_size,
            multi_scale_pred=args.multi_scale_pred,
            is_train=False,
            args=args,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        weights_angles = torch.ones(37).to(args.device)
        weights = torch.ones(2).to(args.device)
        angle_loss = CrossEntropyLoss2d(
            weight=weights_angles, size_average=True, ignore_index=255, reduce=True
        ).to(args.device)
        road_loss = mIoULoss(weight=weights, n_classes=2).to(args.device)

        loss_fn = [road_loss, angle_loss]

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

    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}

    history = {}
    for epoch in range(args.num_epochs):
        start = timer()
        logging.info(f"--------- Training epoch {epoch + 1} / {args.num_epochs} ---------")
        metrics = train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            metric_fns=metric_fns,
            epoch=epoch,
            args=args,
        )
        # FIXME: for spin
        metrics = evaluate_model(
            val_loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            epoch=epoch,
            metrics=metrics,
            args=args,
        )
        history[f"{epoch}"] = metrics
        end = timer()
        logging.info(f"\tEpoch {epoch + 1} took {timedelta(seconds=end - start)}")

    logging.info("--------- Training finished. ---------")

    logging.info("Loss history:")
    for key, value in history.items():
        logging.info(
            f"\tEpoch {key}: Train Loss: {sum(metrics['loss']) / len(metrics['loss']):.4f}, Val Loss: {sum(metrics['val_loss']) / len(metrics['val_loss']):.4f}"
        )

    logging.info("Acc history:")
    for key, value in history.items():
        logging.info(
            f"\tEpoch {key}: Train Acc: {sum(metrics['acc']) / len(metrics['acc']):.4f}, Val Acc: {sum(metrics['val_acc']) / len(metrics['val_acc']):.4f}"
        )

    logging.info("Patch Acc history:")
    for key, value in history.items():
        logging.info(
            f"\tEpoch {key}: Train Patch Acc: {sum(metrics['patch_acc']) / len(metrics['patch_acc']):.4f}, Val Patch Acc: {sum(metrics['val_patch_acc']) / len(metrics['val_patch_acc']):.4f}"
        )

    total_end = timer()
    logging.info(f"Total time: {timedelta(seconds=total_end - total_start)}")


if __name__ == "__main__":
    args = parse_arguments()
    # setup logging
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
        # filename=f"./{args.log_dir}/{args.model}-{timestamp}.log",
    )

    main(args)
