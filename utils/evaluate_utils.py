import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(images, masks, pred_masks, path, epoch):
    fig, ax = plt.subplots(len(images), 4, figsize=(20, len(images) * 5))

    for row in range(len(images)):
        ax[row, 0].imshow(images[row])
        ax[row, 1].imshow(masks[row], cmap="gray")
        ax[row, 2].imshow(pred_masks[row], cmap="gray")
        pred_mask = np.array(pred_masks[row])
        img = np.array(images[row])
        img[pred_mask > 0.5 * 255] = [255]
        ax[row, 3].imshow(img)

    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Mask")
    ax[0, 2].set_title("Predicted Mask")
    ax[0, 3].set_title("Image with predicted mask")
    fig.suptitle(f"Epoch {epoch}")

    plt.savefig(path)
    plt.close()
