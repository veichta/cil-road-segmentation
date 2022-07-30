import logging
import os
import time
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch

from utils.evaluate_utils import save_and_plot_history
from utils.loss import CrossEntropyLoss2d, dice_loss, focal_loss, getTopoLoss, mIoULoss
from utils.metrics import accuracy_fn, patch_accuracy_fn
from utils.topo_loss import soft_cldice
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
            cil_data_info.loc[idx] = row

    if args.datasets == "dg":
        dg_data_info = dataset_info[dataset_info["dataset"] == "DeepGlobe"]
        # set splits for CIL dataset
        for idx, row in dg_data_info.iterrows():
            if np.random.rand() <= args.val_split:
                row["split"] = "val"
                dg_data_info.loc[idx] = row

    if args.datasets == "all":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
    elif args.datasets == "cil":
        dataset_info = cil_data_info
    elif args.datasets == "cil-mrd":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
        dataset_info = dataset_info[dataset_info["dataset"] != "DeepGlobe"]
    elif args.datasets == "cil-dg":
        dataset_info[dataset_info["dataset"] == "CIL"] = cil_data_info
        dataset_info = dataset_info[dataset_info["dataset"] != "MRD"]
    elif args.datasets == "dg":
        dataset_info = dg_data_info
    else:
        raise ValueError(f"Unknown dataset: {args.datasets}")

    if args.min_pixels is not None:
        for idx, row in dataset_info.iterrows():
            if row["n_pixels"] < args.min_pixels and row["dataset"] != "CIL":
                row["split"] = "none"
                dataset_info.loc[idx] = row

    train_df = dataset_info[dataset_info["split"] == "train"]
    val_df = dataset_info[dataset_info["split"] == "val"]
    out = dataset_info[dataset_info["split"] == "none"]

    logging.info("Dataset info:")
    logging.info(f"\tTrain dataset: {train_df.shape[0]} images")
    logging.info(f"\tValidation dataset: {val_df.shape[0]} images")
    logging.info(f"\tIgnoring: {out.shape[0]} images")
    return train_df, val_df


def main(checkpoint_dir, args):
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(vars(args))

    logging.info("Loss Weigths:")
    logging.info(f"\tRoad weight: {args.weight_miou}")
    logging.info(f"\tOrientation weight: {args.weight_vec}")
    logging.info(f"\tTopo weight: {args.weight_topo}")

    total_start = timer()

    train_df, val_df = load_data_info_with_split(args)

    # store dataset info
    train_df.to_csv(os.path.join(checkpoint_dir, "train_dataset.csv"), index=False)
    val_df.to_csv(os.path.join(checkpoint_dir, "val_dataset.csv"), index=False)

    if args.model == "unet":
        logging.info("Using UNet.")
        from datasets.base_dataset import BaseDataset
        from models.base_unet import UNet, evaluate_model, train_one_epoch

        # create train and val datasets
        train_dataset = BaseDataset(
            dataframe=train_df, use_patches=False, target_size=(400, 400), is_train=True, args=args
        )
        val_dataset = BaseDataset(
            dataframe=val_df, use_patches=False, target_size=(400, 400), is_train=False, args=args
        )

        model = UNet().to(args.device)

    elif args.model == "spin":
        logging.info("Using Spin Model.")
        from datasets import road_dataset
        from models.hourglas_spin import HourglassNet, evaluate_model, train_one_epoch
        from utils.utils import weights_init

        target_size = (400, 400)

        model = HourglassNet().to(args.device)
        weights_init(model, args.seed)

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

    # multi step lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[120, 150, 180],
        gamma=0.1,
    )

    weights_angles = torch.ones(37).to(args.device)
    angle_loss = CrossEntropyLoss2d(
        weight=weights_angles, size_average=True, ignore_index=255, reduce=True
    ).to(args.device)

    weights = torch.ones(2).to(args.device)
    road_loss = mIoULoss(weight=weights, n_classes=2).to(args.device)

    loss_fn = [road_loss, angle_loss, soft_cldice(iter_=20), dice_loss, focal_loss]

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
        for _ in range(args.start_epoch):
            lr_scheduler.step()

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
    best_acc = 0
    for epoch in range(args.start_epoch, args.num_epochs):
        start = timer()
        logging.info(f"--------- Training epoch {epoch + 1} / {args.num_epochs} ---------")
        metrics = train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
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
            checkpoint_path=checkpoint_dir,
            args=args,
        )

        history[f"{epoch}"] = metrics

        score = sum(metrics["val_patch_acc"]) / len(metrics["val_patch_acc"])
        if score > best_acc:
            logging.info("\tNew best model.")
            best_acc = score
            torch.save(model.state_dict(), f"{checkpoint_dir}/models/best_model.pth")

        if (epoch + 1) % 20 == 0 and epoch >= 100:
            torch.save(model.state_dict(), f"{checkpoint_dir}/models/model_{epoch + 1}.pth")

        end = timer()
        logging.info(f"\tEpoch {epoch + 1} took {timedelta(seconds=end - start)}")

    logging.info("--------- Training finished. ---------")
    logging.info(f"Best model has {best_acc:.4f} patch accuracy.")

    save_and_plot_history(checkpoint_dir, total_start, history)
    total_end = timer()
    logging.info(f"Total time: {timedelta(seconds=total_end - total_start)}")


if __name__ == "__main__":
    args = parse_arguments()

    # setup logging
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = f"./checkpoints/{args.model}_{timestamp}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(os.path.join(checkpoint_dir, "models"))
        os.makedirs(os.path.join(checkpoint_dir, "plots"))

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO,
        filename=f"{checkpoint_dir}/log.txt",
    )

    main(checkpoint_dir, args)
