import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def denormalize_img(img):
    """Denormalize an image."""
    return np.array(img * 255, dtype=np.uint8)


def plot_predictions(images, masks, pred_masks, path, epoch):
    """Plot predictions of the current epoch."""
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


def save_and_plot_history(checkpoint_dir, history):
    """Save and plot the history of the training."""
    losses = [np.mean(metrics["loss"]) for _, metrics in history.items()]
    val_losses = [np.mean(metrics["val_loss"]) for _, metrics in history.items()]

    road_losses = [np.mean(metrics["road_loss"]) for _, metrics in history.items()]
    val_road_losses = [np.mean(metrics["val_road_loss"]) for _, metrics in history.items()]

    angle_losses = [np.mean(metrics["angle_loss"]) for _, metrics in history.items()]
    val_angle_losses = [np.mean(metrics["val_angle_loss"]) for _, metrics in history.items()]

    topo_losses = [np.mean(metrics["topo_loss"]) for _, metrics in history.items()]
    val_topo_losses = [np.mean(metrics["val_topo_loss"]) for _, metrics in history.items()]

    bce_losses = [np.mean(metrics["bce_loss"]) for _, metrics in history.items()]
    val_bce_losses = [np.mean(metrics["val_bce_loss"]) for _, metrics in history.items()]

    dice_losses = [np.mean(metrics["dice_loss"]) for _, metrics in history.items()]
    val_dice_losses = [np.mean(metrics["val_dice_loss"]) for _, metrics in history.items()]

    accs = [np.mean(metrics["acc"]) for _, metrics in history.items()]
    val_accs = [np.mean(metrics["val_acc"]) for _, metrics in history.items()]

    patch_accs = [np.mean(metrics["patch_acc"]) for _, metrics in history.items()]
    val_patch_accs = [np.mean(metrics["val_patch_acc"]) for _, metrics in history.items()]

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax[0, 0].plot(range(len(losses)), losses, label="Training Loss")
    ax[0, 0].plot(range(len(val_losses)), val_losses, label="Validation Loss")
    ax[0, 0].set_title("Loss")
    ax[0, 0].legend()

    ax[0, 1].plot(range(len(road_losses)), road_losses, label="Training Road Loss")
    ax[0, 1].plot(range(len(val_road_losses)), val_road_losses, label="Validation Road Loss")
    ax[0, 1].set_title("Road Loss")
    ax[0, 1].legend()

    ax[0, 2].plot(range(len(angle_losses)), angle_losses, label="Training Angle Loss")
    ax[0, 2].plot(range(len(val_angle_losses)), val_angle_losses, label="Validation Angle Loss")
    ax[0, 2].set_title("Angle Loss")
    ax[0, 2].legend()

    ax[0, 3].plot(range(len(topo_losses)), topo_losses, label="Training Topo Loss")
    ax[0, 3].plot(range(len(val_topo_losses)), val_topo_losses, label="Validation Topo Loss")
    ax[0, 3].set_title("Topo Loss")
    ax[0, 3].legend()

    ax[1, 0].plot(range(len(bce_losses)), bce_losses, label="Training BCE Loss")
    ax[1, 0].plot(range(len(val_bce_losses)), val_bce_losses, label="Validation BCE Loss")
    ax[1, 0].set_title("BCE Loss")
    ax[1, 0].legend()

    ax[1, 1].plot(range(len(dice_losses)), dice_losses, label="Training Dice Loss")
    ax[1, 1].plot(range(len(val_dice_losses)), val_dice_losses, label="Validation Dice Loss")
    ax[1, 1].set_title("Dice Loss")
    ax[1, 1].legend()

    ax[1, 2].plot(range(len(accs)), accs, label="Training Accuracy")
    ax[1, 2].plot(range(len(val_accs)), val_accs, label="Validation Accuracy")
    ax[1, 2].set_title("Accuracy")
    ax[1, 2].legend()

    ax[1, 3].plot(range(len(patch_accs)), patch_accs, label="Training Patch Accuracy")
    ax[1, 3].plot(range(len(val_patch_accs)), val_patch_accs, label="Validation Patch Accuracy")
    ax[1, 3].set_title("Patch Accuracy")
    ax[1, 3].legend()

    plt.savefig(f"{checkpoint_dir}/plots/history.pdf")
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
            "train_acc",
            "val_acc",
            "train_patch_acc",
            "val_patch_acc",
        ],
    ).to_csv(f"{checkpoint_dir}/history.csv")


def log_metrics(metrics):
    """Log metrics of the current epoch."""
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
    logging.info(f"\tAcc:        {train_acc:.4f}\t({val_acc:.4f})")
    logging.info(f"\tPatch acc:  {train_patch_acc:.4f}\t({val_patch_acc:.4f})")
    logging.info("")
