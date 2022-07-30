"""
Implementation of the road dataset. 
The code is taken from the following repository and changed to our needs:
https://github.com/wgcban/SPIN_RoadMapper
"""
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utils import affinity_utils


class RoadDataset(torch.utils.data.Dataset):
    """Road Dataset used for SPIN Model"""

    def __init__(self, dataframe, target_size, args, multi_scale_pred=True, is_train=True):
        self.args = args

        self.dir = self.args.data_path
        self.dataframe = dataframe

        self.multi_scale_pred = multi_scale_pred
        self.crop_size = [target_size[0], target_size[1]]
        self.angle_theta = 10

        # augmentations
        self.is_train = is_train
        self.do_augmentations = args.augmentation

        self.augment = torch.nn.Sequential(
            transforms.RandomAdjustSharpness(3),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        )

        # to avoid Deadloack between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.images = []
        self.masks = []
        for index in tqdm(range(len(self.dataframe))):
            image_data = self.dataframe.iloc[index, :]
            # load image
            img_path = os.path.join(self.dir, image_data["fpath"])
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.imread(img_path).astype(float)

            # load mask
            mask_path = os.path.join(self.dir, image_data["mpath"])
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Image not found: {mask_path}")
            gt = cv2.imread(mask_path, 0).astype(float)

            self.images.append(image)
            self.masks.append(gt)

    def __len__(self):
        return len(self.dataframe)

    def getRoadData(self, index):
        """Take Image and Mask and augment them.

        Args:
            index (int): Index of the image and mask in the dataframe.

        Returns:
            image (torch.tensor): Image in the shape of (c, h, w).
            gt (torch.tensor): Mask in the shape of (h, w).
        """
        image = self.images[index]
        gt = self.masks[index]

        h, w, c = image.shape
        if self.is_train:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1) / 255.0

        if self.do_augmentations and self.is_train:
            image = self.augment(image)

        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        """Create Orientation Ground Truth

        Args:
            keypoints (list): List of keypoints.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            vecmap_angle (torch.tensor): Orientation Ground Truth in the shape of (h, w).
        """
        vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def __getitem__(self, index):
        """Create images, ground truth and orientation ground truth for multiple scales.

        Args:
            index (int): Index of the image and mask in the dataframe.

        Returns:
            image (list(torch.tensor)): List of images in the shape of (c, h, w) for per scale.
            gt (list(torch.tensor)): List of ground truth in the shape of (h, w) for per scale.
            vecmap_angle (list(torch.tensor)): List of orientation ground truth in the shape of (h, w) for per scale.
        """
        image, gt = self.getRoadData(index)
        c, h, w = image.shape

        labels = []
        vecmap_angles = []
        if self.multi_scale_pred:
            smoothness = [1, 2, 4]
            scale = [4, 2, 1]
        else:
            smoothness = [4]
            scale = [1]

        for i, val in enumerate(scale):
            if val != 1:
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
            gt_orig /= 255.0

            gt_orig_tens = torch.tensor(gt_orig, dtype=torch.float32)
            labels.append(gt_orig_tens)

            # Create Orientation Ground Truth
            keypoints = affinity_utils.getKeypoints(
                gt_orig, is_gaussian=False, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=int(math.ceil(h / (val * 1.0))),
                width=int(math.ceil(w / (val * 1.0))),
            )
            vecmap_angles.append(vecmap_angle)

        return image, labels, vecmap_angles
