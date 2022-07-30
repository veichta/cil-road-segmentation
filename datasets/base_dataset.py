import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, use_patches, target_size, is_train, args):
        super().__init__()
        self.dataframe = dataframe
        self.data_path = args.data_path
        self.device = args.device
        self.patches = use_patches
        self.target_size = target_size
        self.is_train = is_train
        self.do_augmentations = args.augmentation

        self.augment = torch.nn.Sequential(
            transforms.RandomAdjustSharpness(3),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        )

        self.images = []
        self.masks = []
        for index in tqdm(range(len(self.dataframe))):
            image_data = self.dataframe.iloc[index, :]
            # load image
            img_path = os.path.join(self.data_path, image_data["fpath"])
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = cv2.imread(img_path).astype(float)

            # load mask
            mask_path = os.path.join(self.data_path, image_data["mpath"])
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Image not found: {mask_path}")

            gt = cv2.imread(mask_path, 0).astype(float)

            self.images.append(image)
            self.masks.append(gt)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        img = cv2.resize(img, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        img = np.array(img) / 255.0
        mask = np.array(mask) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.is_train and self.do_augmentations:
            img = self.augment(img)

        return img, mask
