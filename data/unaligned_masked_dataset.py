import os.path
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class TransformWithMask:
    def __init__(self, opt, grayscale, method=Image.BICUBIC):
        self.grayscale = transforms.Grayscale(1) if grayscale else None
        self.resize = transforms.Resize((opt.load_size, opt.load_size), method)
        self.to_tensor = transforms.ToTensor()
        self.normalize = (
            transforms.Normalize((0.5,), (0.5,))
            if grayscale
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        self.crop_size = opt.crop_size
        self.load_size = opt.load_size

    def random_crop(self, img, mask):
        x = random.randint(0, np.maximum(0, self.load_size - self.crop_size))
        y = random.randint(0, np.maximum(0, self.load_size - self.crop_size))
        cropped_img = img[:, x : x + self.crop_size, y : y + self.crop_size]
        cropped_mask = mask[:, x : x + self.crop_size, y : y + self.crop_size]
        return cropped_img, cropped_mask

    def __call__(self, img, mask):
        if self.grayscale is not None:
            img = self.grayscale(img)
            mask = self.grayscale(mask)
        img = self.to_tensor(self.resize(img))
        img = self.normalize(img)
        mask = self.to_tensor(self.resize(mask))
        # convert float to binary
        mask = (mask >= 0.5).float()
        img, mask = self.random_crop(img, mask)
        return img, mask


class UnalignedMaskedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(
            opt.dataroot, opt.phase + "A"
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, opt.phase + "B"
        )  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = TransformWithMask(self.opt, grayscale=(input_nc == 1))
        self.transform_B = TransformWithMask(self.opt, grayscale=(output_nc == 1))

        # No need
        # self.norm_AB = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # self.norm_AB_mask = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # load mask data
        A_mask_path = (
            "/".join(A_path.split("/")[:-1]) + "_mask/" + A_path.split("/")[-1]
        )
        B_mask_path = (
            "/".join(B_path.split("/")[:-1]) + "_mask/" + B_path.split("/")[-1]
        )

        A_mask_img = Image.open(A_mask_path).convert("RGB")
        B_mask_img = Image.open(B_mask_path).convert("RGB")

        # apply image transformation
        A, A_mask = self.transform_A(A_img, A_mask_img)
        B, B_mask = self.transform_B(B_img, B_mask_img)

        # No need
        # A_mask = self.norm_AB_mask(A)
        # B_mask = self.norm_AB_mask(B)

        # A = self.norm_AB(A)
        # B = self.norm_AB(B)s

        return {
            "A": A,
            "B": B,
            "mask_A": A_mask,
            "mask_B": B_mask,
            "A_paths": A_path,
            "B_paths": B_path,
        }
        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
