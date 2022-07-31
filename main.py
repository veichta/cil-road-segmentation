import logging
import os
import time
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import torch
<<<<<<< HEAD
from torch.nn import BCELoss

from losses.road_loss import CrossEntropyLoss2d, mIoULoss
from losses.topo_loss import soft_cldice, soft_dice
from utils.data_utils import load_data_info_with_split
from utils.evaluate_utils import save_and_plot_history
from utils.metrics import accuracy_fn, patch_accuracy_fn
=======
from torch import nn
# from torchsummary import summary

from models.base_unet import accuracy_fn, patch_accuracy_fn, f1_fn, patch_F1_fn
>>>>>>> master
from utils.utils import parse_arguments

from topoloss.topoloss_pytorch import getTopoLoss


<<<<<<< HEAD
def main(checkpoint_dir, args):
=======
def load_data_info_with_split(args):
    """Apply validation split to dataset.

    Args:
        args: Arguments.

    Returns:
        pd.Dataframe: Dataset info.
    """
    dataset_info = pd.read_csv(os.path.join(args.data_path, "dataset.csv"))

    cil_data_info = dataset_info[dataset_info["dataset"] == "CIL"]
    
    if args.inference:
        return cil_data_info[cil_data_info["split"] == "test"]
    else:
        cil_data_info = cil_data_info[cil_data_info["split"] != "test"]
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
>>>>>>> master
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

    target_size = (400, 400)
    if args.model == "unet":
        from datasets.base_dataset import BaseDataset
        from models.base_unet import UNet, evaluate_model, train_one_epoch

        logging.info("Using UNet.")

        train_dataset = BaseDataset(
            dataframe=train_df, use_patches=False, target_size=target_size, is_train=True, args=args
        )
        val_dataset = BaseDataset(
            dataframe=val_df, use_patches=False, target_size=target_size, is_train=False, args=args
        )

        model = UNet().to(args.device)
<<<<<<< HEAD
=======
        # summary(model, input_size=(args.batch_size, 384, 384))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
>>>>>>> master

    elif args.model == "spin":
        from datasets import road_dataset
        from models.hourglas_spin import HourglassNet, evaluate_model, train_one_epoch
        from utils.utils import weights_init

<<<<<<< HEAD
        logging.info("Using Spin Model.")
=======
        target_size = (400, 400)

        model = HourglassNet().to(args.device)
        start_epoch = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.resume:
            checkpoint = torch.load(os.path.join("checkpoints", args.resume))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            # loss = checkpoint['loss']
        else:
            weights_init(model, args.seed)
        # summary(model, input_size=(3, target_size[0], target_size[1]))
>>>>>>> master

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

<<<<<<< HEAD
        if args.backbone == "resnet":
            logging.info("Using ResNet backbone.")
            from models.hourglas_spin import BasicResnetBlock
            block = BasicResnetBlock
        else:
            logging.info("Using ResNeXt backbone.")
            from models.hourglas_spin import ResNeXtBottleneck
            block = ResNeXtBottleneck
    
        model = HourglassNet(block=block).to(args.device)
        weights_init(model, args.seed)
=======
>>>>>>> master

    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

<<<<<<< HEAD
    # Dataloaders
=======
        loss_fn = [road_loss, angle_loss, getTopoLoss]

    # create dataloaders
>>>>>>> master
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

<<<<<<< HEAD
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[120, 150, 180],
        gamma=0.1,
    )

    # Losses and metrics
    road_loss = mIoULoss(weight=torch.ones(2).to(args.device), n_classes=2).to(args.device)
    angle_loss = CrossEntropyLoss2d(
        weight=torch.ones(37).to(args.device), size_average=True, ignore_index=255, reduce=True
    ).to(args.device)

    loss_fn = [road_loss, angle_loss, BCELoss(), soft_cldice(iter_=20), soft_dice]

    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
=======
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn, "f1": f1_fn, "patch_f1": patch_F1_fn}
>>>>>>> master

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
        for _ in range(args.start_epoch):
            lr_scheduler.step()

    # Training and validation
    history = {}
<<<<<<< HEAD
    best_acc = 0
    for epoch in range(args.start_epoch, args.num_epochs):
=======
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
>>>>>>> master
        start = timer()
        logging.info(f"--------- Training epoch {args.start_epoch + epoch + 1} / {args.start_epoch + args.num_epochs} ---------")
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
<<<<<<< HEAD
=======

>>>>>>> master
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

        torch.save(model.state_dict(), f"{checkpoint_dir}/models/last_model.pth")

        end = timer()
        logging.info(f"\tEpoch {epoch + 1} took {timedelta(seconds=end - start)}")

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics,
            }, os.path.join("checkpoints", args.model_name + "_checkpoint_epoch" + str(epoch)))

    logging.info("--------- Training finished. ---------")
    logging.info(f"Best model has {best_acc:.4f} patch accuracy.")

    save_and_plot_history(checkpoint_dir, history)
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
