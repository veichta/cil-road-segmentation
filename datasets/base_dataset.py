import os

import numpy as np
import torch
from PIL import Image

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


def image_to_patches(image, masks=None):
    h, w = image.shape[0], image.shape[1]  # shape of images
    assert (h % PATCH_SIZE) + (w % PATCH_SIZE) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE

    patches = image.reshape((h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1))
    patches = np.moveaxis(patches, 1, 2)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 1, 2)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, use_patches, target_size, args):
        super().__init__()
        self.dataframe = dataframe
        self.data_path = args.data_path
        self.device = args.device
        self.patches = use_patches
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.dataframe.iloc[idx]["fpath"]))
        mask = Image.open(os.path.join(self.data_path, self.dataframe.iloc[idx]["mpath"]))
        img = img.resize(self.target_size)
        mask = mask.resize(self.target_size)

        img = np.array(img) / 255
        mask = np.array(mask)
        mask[mask > 0] = 1
        
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.patches:
            img, mask = image_to_patches(img, mask)
            img = torch.tensor(img, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            img = img.permute(0, 3, 1, 2)

            return img, mask

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32)
        # print(img.device, mask.device)
        return img, mask
