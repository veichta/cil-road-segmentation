import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    """Function to erode an image.

    Args:
        img (torch.tensor): Image to be eroded with shape (N, C, H, W).

    Returns:
        torch.tensor: eroded image.
    """

    if len(img.shape) != 4:
        raise ValueError(f"Invalid input shape: {img.shape}")

    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    """Function to dilate an image.

    Args:
        img (torch.tensor): Image to be dilated with shape (N, C, H, W).

    Returns:
        torch.tensor: eroded image.
    """

    if len(img.shape) != 4:
        raise ValueError(f"Invalid input shape: {img.shape}")

    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    """Function to open an image."""
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    """Function to perform soft skeletonization.

    Args:
        img (torch.tensor): Image to be skeletonized with shape (N, C, H, W).
        iter_ (int): Number of iterations.

    Returns:
        torch.tensor: skeletonized image.
    """

    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class soft_cldice(nn.Module):
    def __init__(self, iter_=100, smooth=1.0):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        """Function to compute the soft topo loss.

        Args:
            y_true (torch.tensor): Ground truth with shape (N, C, H, W).
            y_pred (torch.tensor): Predicted image with shape (N, C, H, W).

        Returns:
            torch.tensor: soft topo loss.
        """
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_pred[:, 1:, ...]) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_true[:, 1:, ...]) + self.smooth
        )
        return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def soft_dice(y_true, y_pred):
    """Function to compute soft dice loss.

    Args:
        y_true (torch.tensor): Ground truth with shape (N, C, H, W).
        y_pred (torch.tensor): Predicted image with shape (N, C, H, W).

    Returns:
        torch.tensor: Soft dice loss.
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:, 1:, ...])
    coeff = (2.0 * intersection + smooth) / (
        torch.sum(y_true[:, 1:, ...]) + torch.sum(y_pred[:, 1:, ...]) + smooth
    )
    return 1.0 - coeff


class soft_dice_cldice(soft_cldice):
    def __init__(self, iter_=0, alpha=0.5, smooth=1.0):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        """Function to compute soft dice loss and topo loss.

        Args:
            y_true (torch.tensor): Ground truth with shape (N, C, H, W).
            y_pred (torch.tensor): Predicted image with shape (N, C, H, W).

        Returns:
            torch.tensor: Weighted sum of dice loss and topo loss.
        """
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_pred[:, 1:, ...]) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (
            torch.sum(skel_true[:, 1:, ...]) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
