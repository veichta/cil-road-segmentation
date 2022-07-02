import logging
import os
import time
from re import I

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchsummary import summary
from tqdm import tqdm

from datasets.base_dataset import BaseDataset
from models.base_unet import UNet, accuracy_fn, patch_accuracy_fn
from utils.utils import parse_arguments, show_image_segmentation


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
    model = UNet(chs=(3, 64, 128)).to(args.device)
    summary(model, input_size=(args.batch_size, 384, 384))
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}

    history = {}
    for epoch in range(args.num_epochs):
        metrics = train_one_epoch(args, train_loader, model, loss_fn, optimizer, metric_fns, epoch)
        history = evaluate_model(val_loader, model, loss_fn, metric_fns, history, epoch, metrics)


def evaluate_model(val_loader, model, loss_fn, metric_fns, history, epoch, metrics):
    model.eval()
    with torch.no_grad():  # do not keep track of gradients
        show = True
        for (x, y) in tqdm(val_loader):
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)

            if show:
                print_predictions(epoch, x, y, y_hat)
                show = False

                # log partial metrics
            metrics["val_loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
    history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
    print(" ".join([f"- {str(k)} = {str(v)}" + "\n " for (k, v) in history[epoch].items()]))
    return history


def print_predictions(epoch, x, y, y_hat):
    idx = 1
    img = (x[idx] * 255).to(torch.uint8).permute(1, 2, 0)
    mask = (y[idx] * 255).to(torch.uint8)

    img = img.cpu().numpy()
    mask = mask.cpu().numpy()

    img = Image.fromarray(img)
    mask = Image.fromarray(mask * 255)
    show_image_segmentation(img, mask, f"Ground Truth - Epoch {epoch+1}")

    pred_mask = (y_hat[idx] * 255).to(torch.uint8).cpu().numpy()
    pred_mask = Image.fromarray(pred_mask)
    show_image_segmentation(img, pred_mask, f"Prediction - Epoch {epoch+1}")


def train_one_epoch(args, train_loader, model, loss_fn, optimizer, metric_fns, epoch):
    metrics = {"loss": [], "val_loss": []}
    for k in metric_fns:
        metrics[k] = []
        metrics["val_" + k] = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    pbar.set_postfix({k: 0 for k in metrics})

    model.train()
    for (x, y) in pbar:
        train_step(args, model, loss_fn, optimizer, metric_fns, metrics, x, y)

        pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

    return metrics


def train_step(args, model, loss_fn, optimizer, metric_fns, metrics, x, y):
    x = x.to(args.device)
    y = y.to(args.device)

    optimizer.zero_grad()  # zero out gradients
    y_hat = model(x)  # forward pass
    loss = loss_fn(y_hat, y)
    loss.backward()  # backward pass
    optimizer.step()  # optimize weights

    # log partial metrics
    metrics["loss"].append(loss.item())
    for k, fn in metric_fns.items():
        metrics[k].append(fn(y_hat, y).item())


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
