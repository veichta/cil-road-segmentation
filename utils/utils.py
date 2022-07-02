import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
        "--lr", 
        type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, default=0, help="Weight decay"
    )
    parser.add_argument(
        "--dropout", 
        type=float, default=0, help="Dropout"    
    )

    # Dataset setup
    parser.add_argument(
        "--data_path",
        type=str, default="./data/big-dataset", help="Path to csv file"
    )
    parser.add_argument(
        "--datasets",
        type=str, choices=["all", "cil"], default="all", help="Datasets to use"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, default=4, help="Number of workers"
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
        type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--save_dir",
        type=str, default="./checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, default="./logs", help="Directory to save logs"
    )
    # fmt: on
    return parser.parse_args()


def apply_mask(img, mask):
    """Apply mask to image.

    Args:
        img: Image to apply mask to.
        mask: Mask to apply.

    Returns:
        PIL.Image: Image with mask applied.
    """
    img = np.array(img)
    mask = np.array(mask)

    img[mask > 255 * 0.5] = 255
    return Image.fromarray(img)


def show_image_segmentation(img, mask, title):
    """Show image and mask.

    Args:
        img: Image to show.
        mask: Mask to show.
        title: Title of image.
    """
    seg = apply_mask(img, mask)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    ax[2].imshow(seg)
    ax[2].set_title("Segmentation applied to image")
    ax[2].axis("off")

    plt.suptitle(title)
    plt.savefig(f"checkpoints/{title}.png")
