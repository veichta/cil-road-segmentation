import matplotlib.pyplot as plt
import numpy as np


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
    pred_masks = [pred_mask.argmax(dim=0).cpu().numpy() for pred_mask in pred_masks]

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
