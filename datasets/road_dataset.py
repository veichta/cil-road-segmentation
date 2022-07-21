import math
import os

import cv2
import numpy as np
import torch
from utils import affinity_utils


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target_size, args, multi_scale_pred=True, is_train=True):

        self.dataframe = dataframe
        self.args = args
        # paths
        self.dir = self.args.data_path

        # list of all images
        # self.images = [line.rstrip("\n") for line in open(self.image_list)]

        # augmentations
        self.augmentation = is_train
        self.crop_size = [target_size[0], target_size[1]]
        self.multi_scale_pred = multi_scale_pred

        # preprocess
        self.angle_theta = 10  # TODO: change to args

        self.is_train = is_train

        # FIXME: fix for train and validation
        self.mean_bgr = np.array([129.98980851, 132.72141085, 132.08936675])
        self.deviation_bgr = np.array([53.78613508, 51.27242953, 50.02140993])

        # to avoid Deadlock between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.dataframe)

    def getRoadData(self, index):
        image_data = self.dataframe.iloc[index, :]

        # load image
        img_path = os.path.join(self.dir, image_data["fpath"])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(img_path).astype(float)
        h, w, c = image.shape

        if self.args.inference:
            if self.augmentation:  # TODO: change to troch transforms
                flip = np.random.choice(2) * 2 - 1
                image = np.ascontiguousarray(image[:, ::flip, :])
                rotation = np.random.randint(4) * 90
                M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
                image = cv2.warpAffine(image, M, (w, h))

            # image = image / 255.0
            image = self.reshape(image)
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            return image, []

        # load mask
        mask_path = os.path.join(self.dir, image_data["mpath"])
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Image not found: {mask_path}")

        gt = cv2.imread(mask_path, 0).astype(float)

        if self.augmentation:  # TODO: change to troch transforms
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))
            
        # image = image / 255.0
        image = self.reshape(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def __getitem__(self, index):

        image, gt = self.getRoadData(index)

        if self.args.inference:
            return image, [], []

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
                    (int(math.ceil(h / (val * 1.0))),
                     int(math.ceil(w / (val * 1.0)))),
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

    def random_crop(self, image, gt, size):

        w, h, _ = image.shape
        crop_h, crop_w = size

        start_x = np.random.randint(0, w - crop_w) if w - crop_w > 0 else 0
        start_y = np.random.randint(0, h - crop_h) if h - crop_h > 0 else 0

        image = image[start_x: start_x + crop_w, start_y: start_y + crop_h, :]
        gt = gt[start_x: start_x + crop_w, start_y: start_y + crop_h]

        return image, gt

    def reshape(self, image):

        if self.args.normalize_type == "Std":
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.args.normalize_type == "MinMax":
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.args.normalize_type == "Mean":
            image -= self.mean_bgr
        else:
            image = (image / 255.0) * 2 - 1
        return image
