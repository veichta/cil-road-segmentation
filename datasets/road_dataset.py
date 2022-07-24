import math
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from utils import affinity_utils


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_size, args, multi_scale_pred=True, is_train=True):

        self.dataframe = dataframe
        self.args = args
        # paths
        self.dir = self.args.data_path

        # augmentations
        self.augmentation = is_train
        self.crop_size = [target_size[0], target_size[1]]
        self.multi_scale_pred = multi_scale_pred

        self.channel_mean = [125.78279375, 130.15193705, 130.92051354]
        self.channel_std = [52.71067801, 49.9758017, 48.39758796]

        self.normalize = torch.nn.Sequential(
            transforms.Normalize(self.channel_mean, self.channel_std),
        )

        # preprocess
        self.angle_theta = 10  # TODO: change to args

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
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
        image = self.images[index]
        gt = self.masks[index]

        h, w, c = image.shape
        if self.augmentation:  # TODO: change to troch transforms
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))

        # image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        image = self.normalize(image)

        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def __getitem__(self, index):

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
