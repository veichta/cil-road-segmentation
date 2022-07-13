#!/usr/bin/env python2

"""
create_crops.py: script to crops for training and validation.
                It will save the crop images and mask in the format: 
                <image_name>_<x>_<y>.<suffix>
                where x = [0, (Image_Height - crop_size) / stride]
                      y = [0, (Image_Width - crop_size) / stride] 

It will create following directory structure:
    base_dir
        |   train_crops.txt   # created by script
        |   val_crops.txt     # created by script
        |
        └───train_crops       # created by script
        │   └───gt
        │   └───images
        └───val_crops         # created by script
        │   └───gt
        │   └───images
"""

from __future__ import print_function

import argparse
import os
import mmap
import cv2
import time
import numpy as np
from skimage import io
from tqdm import tqdm
tqdm.monitor_interval = 0



def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def CreatCrops(base_dir, crop_type, size, stride, image_suffix, gt_suffix, image_prefix, gt_prefix):

    crops = os.path.join(base_dir, '{}_crops'.format(crop_type))
    crops_file = open(os.path.join(base_dir,'{}_crops.txt'.format(crop_type)),'w')

    full_file_path = os.path.join(base_dir,'{}.txt'.format(crop_type))
    full_file = open(full_file_path,'r')

    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    failure_images = []
    for name in tqdm(full_file, ncols=100, desc="{}_crops".format(crop_type), 
                            total=get_num_lines(full_file_path)):
        
        name = name.strip("\n")
        image_file = os.path.join(base_dir,'{}/images'.format(crop_type),image_prefix+name+image_suffix)
        gt_file = os.path.join(base_dir,'{}/gt'.format(crop_type),gt_prefix+name+gt_suffix)

        if not verify_image(image_file):
            failure_images.append(image_file)
            continue

        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file,0)
        
        if image is None:
            failure_images.append(image_file)
            continue

        if gt is None:
            failure_images.append(image_file)
            continue

        H,W,C = image.shape
        maxx = (H-size)//stride
        maxy = (W-size)//stride
        for x in range(maxx+1):
            for y in range(maxy+1):
                im_ = image[x*stride:x*stride + size,y*stride:y*stride + size,:]
                gt_ = gt[x*stride:x*stride + size,y*stride:y*stride + size]
                crops_file.write('{}_{}_{}\n'.format(name,x,y))
                # print(crops+'/images/{}_{}_{}.png'.format(name,x,y))
                # print(im_)
                im_check = cv2.imwrite(crops+'/images/{}_{}_{}.png'.format(name,x,y),  im_)
                gt_check = cv2.imwrite(crops+'/gt/{}_{}_{}.png'.format(name,x,y), gt_)
                assert(im_check and gt_check)
    
    crops_file.close()
    full_file.close()
    if len(failure_images) > 0:
        print("Unable to process {} images : {}".format(len(failure_images), failure_images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, required=True, 
        help='Base directory for Spacenent Dataset.')
    parser.add_argument('--crop_size', type=int, required=True, 
        help='Crop Size of Image')
    parser.add_argument('--crop_overlap', type=int, required=True, 
        help='Crop overlap Size of Image')
    parser.add_argument('--im_suffix', type=str, required=True, 
        help='Dataset specific image suffix.')
    parser.add_argument('--gt_suffix', type=str, required=True, 
        help='Dataset specific gt suffix.')
    parser.add_argument('--im_prefix', type=str, required=True, 
        help='Dataset specific image prefix.')
    parser.add_argument('--gt_prefix', type=str, required=True, 
        help='Dataset specific gt preifx.')

    args = parser.parse_args()

    start = 0
    ## Create overlapping Crops for training
    CreatCrops(args.base_dir, 
                crop_type='train', 
                size=args.crop_size, 
                stride=args.crop_overlap,
                image_suffix=args.im_suffix, 
                gt_suffix=args.gt_suffix,
                image_prefix=args.im_prefix,
                gt_prefix=args.gt_prefix)

    ## Create non-overlapping Crops for validation
    CreatCrops(args.base_dir, 
                crop_type='val', 
                size=args.crop_size, 
                stride=args.crop_size,  ## Non-overlapping
                image_suffix=args.im_suffix, 
                gt_suffix=args.gt_suffix,
                image_prefix=args.im_prefix,
                gt_prefix=args.gt_prefix)

    end = 0
    print('Finished Creating crops, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()