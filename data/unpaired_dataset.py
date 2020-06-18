import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch
import torchvision.transforms as transforms
from datetime import datetime


def generateMask(image, threshold):
    """
    image: tensor(C,H,W)
    threhols: (-1,1)
    
    """
    mask = (image>threshold).to(dtype=torch.float32)
    return mask


class UnpairedDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'train' + 'A')
        self.dir_B = os.path.join(opt.dataroot, 'train' + 'B')
        self.dir_len = len(self.dir_A) + 1

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        
    
    def __getitem__(self, index):
        
        A_path = self.A_paths[index % self.A_size]
        A_name = A_path[self.dir_len:]
        
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
            
        B_path = self.B_paths[index_B]
        
        A_mask_path = A_path.replace('trainA','maskA')
        B_mask_path = B_path.replace('trainB','maskB')
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_mask = Image.open(A_mask_path).convert('RGB')
        B_mask = Image.open(B_mask_path).convert('RGB')
    
        
        A = self.transform(A_img)
        B = self.transform(B_img)
        
        A_mask = self.transform(A_mask)
        B_mask = self.transform(B_mask)
    
        tmp = A_mask[0, ...] * 0.299 + A_mask[1, ...] * 0.587 + A_mask[2, ...] * 0.114
        A_mask = tmp.unsqueeze(0)
        tmp = B_mask[0, ...] * 0.299 + B_mask[1, ...] * 0.587 + B_mask[2, ...] * 0.114
        B_mask = tmp.unsqueeze(0)
        

        if self.opt.input_nc == 1:  # RGB to gray
            if A.shape[0] != 1:
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

        if self.opt.output_nc == 1:  # RGB to gray
            if B.shape[0] != 1:
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)
        
        A_mask = generateMask(A_mask, -0.8)
        B_mask = generateMask(B_mask, -0.8)

        return {'A': A, 'B': B,'A_mask':A_mask, 'B_mask': B_mask, 'A_paths': A_path, 'B_paths':B_path,'A_name':A_name }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnpairedDataset'
