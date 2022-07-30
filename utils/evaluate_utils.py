import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def denormalize_img(img):
    # channel_mean = [0.510, 0.521, 0.518]
    # channel_std = [0.239, 0.218, 0.209]

    # img = np.array(img, dtype=np.float32)
    # for c in range(3):
    #     img[:, :, c] = (img[:, :, c] * channel_std[c]) + channel_mean[c]

    # # plt.imshow(np.array(img, dtype=np.uint8))
    # # plt.show()
    img = img * 255.0

    return np.array(img, dtype=np.uint8)


def plot_predictions(images, masks, pred_masks, path, epoch):
    fig, ax = plt.subplots(len(images), 4, figsize=(20, len(images) * 5))

    images = [img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] for img in images]
    images = [denormalize_img(img) for img in images]

    masks = [mask.cpu().numpy() for mask in masks]
    if len(pred_masks.shape) > 3:
        pred_masks = [pred_mask.argmax(dim=0).cpu().numpy() for pred_mask in pred_masks]
    else:
        pred_masks = [pred_mask.sigmoid().round().cpu().numpy() for pred_mask in pred_masks]

    masks = [(mask * 255).astype(np.uint8) for mask in masks]
    pred_masks = [(pred_mask * 255).astype(np.uint8) for pred_mask in pred_masks]

    for row in range(len(images)):
        mask = masks[row]
        pred_mask = pred_masks[row]

        img = images[row].copy()
        img[mask < 255] = img[mask < 255] * 0.5
        ax[row, 0].imshow(img)

        ax[row, 1].imshow(mask, cmap="gray")
        ax[row, 2].imshow(pred_mask, cmap="gray")

        img = images[row].copy()
        img[pred_mask < 255] = img[pred_mask < 255] * 0.5
        ax[row, 3].imshow(img)

    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Mask")
    ax[0, 2].set_title("Predicted Mask")
    ax[0, 3].set_title("Image with predicted mask")
    fig.suptitle(f"Epoch {epoch}")

    plt.savefig(path)
    plt.close()


def save_and_plot_history(checkpoint_dir, total_start, history):
    losses = [sum(metrics["loss"]) / len(metrics["loss"]) for _, metrics in history.items()]
    val_losses = [
        sum(metrics["val_loss"]) / len(metrics["val_loss"]) for _, metrics in history.items()
    ]

    road_losses = [
        sum(metrics["road_loss"]) / len(metrics["road_loss"]) for _, metrics in history.items()
    ]
    val_road_losses = [
        sum(metrics["val_road_loss"]) / len(metrics["val_road_loss"])
        for _, metrics in history.items()
    ]

    angle_losses = [
        sum(metrics["angle_loss"]) / len(metrics["angle_loss"]) for _, metrics in history.items()
    ]
    val_angle_losses = [
        sum(metrics["val_angle_loss"]) / len(metrics["val_angle_loss"])
        for _, metrics in history.items()
    ]

    topo_losses = [
        sum(metrics["topo_loss"]) / len(metrics["topo_loss"]) for _, metrics in history.items()
    ]
    val_topo_losses = [
        sum(metrics["val_topo_loss"]) / len(metrics["val_topo_loss"])
        for _, metrics in history.items()
    ]

    bce_losses = [
        sum(metrics["bce_loss"]) / len(metrics["bce_loss"]) for _, metrics in history.items()
    ]
    val_bce_losses = [
        sum(metrics["val_bce_loss"]) / len(metrics["val_bce_loss"])
        for _, metrics in history.items()
    ]

    dice_losses = [
        sum(metrics["dice_loss"]) / len(metrics["dice_loss"]) for _, metrics in history.items()
    ]
    val_dice_losses = [
        sum(metrics["val_dice_loss"]) / len(metrics["val_dice_loss"])
        for _, metrics in history.items()
    ]

    focal_losses = [
        sum(metrics["focal_loss"]) / len(metrics["focal_loss"]) for _, metrics in history.items()
    ]
    val_focal_losses = [
        sum(metrics["val_focal_loss"]) / len(metrics["val_focal_loss"])
        for _, metrics in history.items()
    ]

    accs = [sum(metrics["acc"]) / len(metrics["acc"]) for _, metrics in history.items()]
    val_accs = [sum(metrics["val_acc"]) / len(metrics["val_acc"]) for _, metrics in history.items()]
    patch_accs = [
        sum(metrics["patch_acc"]) / len(metrics["patch_acc"]) for _, metrics in history.items()
    ]
    val_patch_accs = [
        sum(metrics["val_patch_acc"]) / len(metrics["val_patch_acc"])
        for _, metrics in history.items()
    ]

    plt.plot(losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overall Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_loss.pdf")
    plt.close()

    plt.plot(road_losses, label="Train Road Loss")
    plt.plot(val_road_losses, label="Val Road Loss")
    plt.ylim(np.min(road_losses) - 1, np.max(road_losses) + 1)
    plt.xlabel("Epoch")
    plt.ylabel("Road Loss")
    plt.title("Road Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_road_loss.pdf")
    plt.close()

    plt.plot(angle_losses, label="Train Angle Loss")
    plt.plot(val_angle_losses, label="Val Angle Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Angle Loss")
    plt.title("Angle Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_angle_loss.pdf")
    plt.close()

    plt.plot(topo_losses, label="Train Topo Loss")
    plt.plot(val_topo_losses, label="Val Topo Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Topo Loss")
    plt.title("Topo Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_topo_loss.pdf")
    plt.close()

    plt.plot(bce_losses, label="Train BCE Loss")
    plt.plot(val_bce_losses, label="Val BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("BCE Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_mse_loss.pdf")
    plt.close()

    plt.plot(dice_losses, label="Train Dice Loss")
    plt.plot(val_dice_losses, label="Val Dice Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.title("Dice Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_dice_loss.pdf")
    plt.close()

    plt.plot(focal_losses, label="Train Focal Loss")
    plt.plot(val_focal_losses, label="Val Focal Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.title("Focal Loss")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_focal_loss.pdf")
    plt.close()

    plt.plot(accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Overall Accuracy")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_acc.pdf")
    plt.close()

    plt.plot(patch_accs, label="Train Patch Acc")
    plt.plot(val_patch_accs, label="Val Patch Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Patch Acc")
    plt.title("Patch Accuracy")
    plt.legend()
    plt.savefig(f"{checkpoint_dir}/plots/history_patch_acc.pdf")
    plt.close()

    df_hist = [
        (
            losses[epoch],
            val_losses[epoch],
            road_losses[epoch],
            val_road_losses[epoch],
            angle_losses[epoch],
            val_angle_losses[epoch],
            topo_losses[epoch],
            val_topo_losses[epoch],
            bce_losses[epoch],
            val_bce_losses[epoch],
            dice_losses[epoch],
            val_dice_losses[epoch],
            focal_losses[epoch],
            val_focal_losses[epoch],
            accs[epoch],
            val_accs[epoch],
            patch_accs[epoch],
            val_patch_accs[epoch],
        )
        for epoch in range(len(losses))
    ]

    df_hist = pd.DataFrame(
        df_hist,
        columns=[
            "train_loss",
            "val_loss",
            "train_road_loss",
            "val_road_loss",
            "train_angle_loss",
            "val_angle_loss",
            "train_topo_loss",
            "val_topo_loss",
            "train_mse_loss",
            "val_mse_loss",
            "train_dice_loss",
            "val_dice_loss",
            "train_focal_loss",
            "val_focal_loss",
            "train_acc",
            "val_acc",
            "train_patch_acc",
            "val_patch_acc",
        ],
    ).to_csv(f"{checkpoint_dir}/history.csv")


def log_metrics(metrics):
    train_loss = np.mean(metrics["loss"])
    val_loss = np.mean(metrics["val_loss"])

    train_road_loss = np.mean(metrics["road_loss"])
    val_road_loss = np.mean(metrics["val_road_loss"])

    train_angle_loss = np.mean(metrics["angle_loss"])
    val_angle_loss = np.mean(metrics["val_angle_loss"])

    train_topo_loss = np.mean(metrics["topo_loss"])
    val_topo_loss = np.mean(metrics["val_topo_loss"])

    train_bce = np.mean(metrics["bce_loss"])
    val_bce = np.mean(metrics["val_bce_loss"])

    train_dice_loss = np.mean(metrics["dice_loss"])
    val_dice_loss = np.mean(metrics["val_dice_loss"])

    train_focal_loss = np.mean(metrics["focal_loss"])
    val_focal_loss = np.mean(metrics["val_focal_loss"])

    train_acc = np.mean(metrics["acc"])
    val_acc = np.mean(metrics["val_acc"])

    train_patch_acc = np.mean(metrics["patch_acc"])
    val_patch_acc = np.mean(metrics["val_patch_acc"])

    logging.info("")
    logging.info("\t            train\tval")
    logging.info(f"\tLoss:       {train_loss:.4f}\t({val_loss:.4f})")
    logging.info(f"\tRoad loss:  {train_road_loss:.4f}\t({val_road_loss:.4f})")
    logging.info(f"\tAngle loss: {train_angle_loss:.4f}\t({val_angle_loss:.4f})")
    logging.info(f"\tTopo loss:  {train_topo_loss:.4f}\t({val_topo_loss:.4f})")
    logging.info(f"\tBCE:        {train_bce:.4f}\t({val_bce:.4f})")
    logging.info(f"\tDice loss:  {train_dice_loss:.4f}\t({val_dice_loss:.4f})")
    logging.info(f"\tFocal loss: {train_focal_loss:.4f}\t({val_focal_loss:.4f})")
    logging.info(f"\tAcc:        {train_acc:.4f}\t({val_acc:.4f})")
    logging.info(f"\tPatch acc:  {train_patch_acc:.4f}\t({val_patch_acc:.4f})")
    logging.info("")
