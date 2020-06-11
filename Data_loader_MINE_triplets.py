from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import cv2
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

window_coef_dictionary = {'1_Cathode_1_CBS':8,
                '2_Cathode_1_CBS':8,
                '2_Cathode_1_TLD':8,
                '2_Cathode_2_TLD':8,
                '2__UCL_Li-battery_DCTC_bin2_ROI1_RUN1':8,
                '3__UCL_Li-battery_cath_disc_side__Run1':8,
                '5__UCL_Li-battery_cath_disc_side__Run3':8,
                'RON_Oxygen':6,
                'RON_Xenon':6,
                'St_Jude':3,
                'TIQ':6,
                'eds_weld':12,
                'Bosch_EBSD_SEM':3,
                'T1_19x24x20nm':8,
                'IN718_CBS':6}


class Custom_Dataloader(Dataset):

    def __init__(self, train, n_epochs=None, current_epoch=None, train_root_dir=None, val_root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.transform = transform
        self.img_paths =[]
        self.n_epochs = n_epochs
        self.current_epoch = current_epoch


        if self.train:
            for subfol in os.walk(train_root_dir):
                for file_path in glob(os.path.join(subfol[0], '*.tif')):
                    self.img_paths.append(file_path)     
            
            print("Number of training images", len(self.img_paths))

        else:    
            for subfol in os.walk(val_root_dir):
                for file_path in glob(os.path.join(subfol[0], '*.tif')):
                    self.img_paths.append(file_path)
            print("Number of validation images", len(self.img_paths))

        self.img_paths = sorted(self.img_paths)

    def __getitem__(self, idx):

        subfolder_path = os.path.dirname(self.img_paths[idx])
        subfolder_name = os.path.basename(subfolder_path)

        window_coef = window_coef_dictionary[subfolder_name]

        image_anchor = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        direction = np.random.choice(np.asarray(['forward']), 1)

        positive_idx = int(idx+self.current_epoch/self.n_epochs*window_coef+1)

        negative_idx = positive_idx+1

        if direction == 'forward':
            positive_image_path = self.img_paths[np.clip(idx+positive_idx, 0, len(self.img_paths)-1)]

            if os.path.basename((os.path.dirname(positive_image_path))) != subfolder_name:
                positive_image_path = self.img_paths[idx]

            positive_image_path = self.img_paths[np.clip(idx+positive_idx, 0, len(self.img_paths)-1)]
            negative_image_path = self.img_paths[np.clip(idx+negative_idx, 0, len(self.img_paths)-1)]

            image_positive = cv2.imread(positive_image_path, cv2.IMREAD_GRAYSCALE)
  
            image_negative = cv2.imread(negative_image_path,cv2.IMREAD_GRAYSCALE)

        
        if direction == 'backward':
            positive_image_path = self.img_paths[np.clip(idx-positive_idx, 0, len(self.img_paths)-1)]
            negative_image_path = self.img_paths[np.clip(idx-negative_idx, 0, len(self.img_paths)-1)]

            image_positive = cv2.imread(positive_image_path, cv2.IMREAD_GRAYSCALE)

            image_negative = cv2.imread(negative_image_path,cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            image_anchor, image_positive, image_negative = self.transform((image_anchor, image_positive, image_negative))

        target = [1,1,0]
        
        return (image_anchor/255, image_positive/255, image_negative/255), target

    def __len__(self):
        return len(self.img_paths)
        # return 100


class RandomCrop(object):
    """Crop the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        n_patches = 64
        batch_of_patches_anchor = np.zeros((n_patches, self.output_size[0], self.output_size[1],3))
        batch_of_patches_positive = np.zeros((n_patches, self.output_size[0], self.output_size[1],3))
        batch_of_patches_negative = np.zeros((n_patches, self.output_size[0], self.output_size[1],3))

        mse_positions_list = []
        (image_anchor, image_positive, image_negative) = sample
        h, w = min(image_anchor.shape[0],image_positive.shape[0], image_negative.shape[0]), min(image_anchor.shape[1],image_positive.shape[1], image_negative.shape[1])
        new_h, new_w = self.output_size

        for i in range(0, h-new_h, new_h):
            for j in range(0, w-new_w, new_w):
                patch_anchor = image_anchor[i:i+new_h, j:j+new_w]
                patch_negative = image_negative[i:i+new_h, j:j+new_w]

                patch_MSE = (np.square(patch_anchor - patch_negative)).mean(axis=None)
                mse_positions_list.append([patch_MSE, i, j])

        mse_positions_list = sorted(mse_positions_list, reverse = True)[0::int(len(mse_positions_list)/n_patches+1)]

        for patch_idx, patch in enumerate(mse_positions_list):
            patch_an = image_anchor[patch[1]:patch[1]+new_h, patch[2]:patch[2]+new_w]
            patch_pos = image_positive[patch[1]:patch[1]+new_h, patch[2]:patch[2]+new_w]
            patch_neg = image_negative[patch[1]:patch[1]+new_h, patch[2]:patch[2]+new_w]
            
            for channel in range(3):
                batch_of_patches_anchor[patch_idx,:,:,channel] = patch_an
                batch_of_patches_positive[patch_idx,:,:,channel] = patch_pos
                batch_of_patches_negative[patch_idx,:,:,channel] = patch_neg

        return (batch_of_patches_anchor, batch_of_patches_positive, batch_of_patches_negative)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        (image_anchor, image_positive, image_negative) = sample

        h, w = image_anchor.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_anchor = transform.resize(image_anchor, (new_h, new_w))
        image_positive = transform.resize(image_positive, (new_h, new_w))
        image_negative = transform.resize(image_negative, (new_h, new_w))

        return (image_anchor*255, image_positive*255, image_negative*255)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        (image_anchor, image_positive, image_negative)= sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_anchor = image_anchor.transpose((0, 3, 1, 2))
        image_positive = image_positive.transpose((0, 3, 1, 2))
        image_negative = image_negative.transpose((0, 3, 1, 2))

        return (image_anchor, image_positive, image_negative)


