import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


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
    # plt.show()
    plt.savefig(f"checkpoints/{title}.png")

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
