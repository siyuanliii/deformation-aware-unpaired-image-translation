import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import random
import cv2
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch
import torchvision.transforms as transforms
from datetime import datetime
import pickle
from util.util import *


def random_flip(img, pose_keypoints, H_prob = 0, V_prob=0, H=1, W=1):
    '''
    img: PIL image
    keypoints: same with generate heatmap
    H_prob: horizontal flip probablity
    V_prob: vertical flip probablity
    
    '''
    keypoints = pose_keypoints.copy()
    
    
    if type(img) != type(None):
        if random.random()<H_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            keypoints[:,0] = W - keypoints[:,0]
        if random.random()<V_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            keypoints[:,1] = H - keypoints[:,1]
        return img, keypoints
    else:
        if random.random()<H_prob:
            keypoints[:,0] = W - keypoints[:,0]
        if random.random()<V_prob:
            keypoints[:,1] = H - keypoints[:,1]
        return keypoints



def get_transform_pose(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """
    Transform pixel location to different reference
    """
    t = get_transform_pose(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.pad(pt,((0,0),(0,1)),mode = 'constant', constant_values = 1).T
    new_pt = np.dot(t, new_pt)
    return new_pt

def generateMask(image, threshold):
    """
    image: tensor(C,H,W)
    threhols: (-1,1)
    
    """
    mask = (image>threshold).to(dtype=torch.float32)
    return mask



class PoseDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = opt.pose_img_dir
        self.dir_len = len(self.dir_A) + 1
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.transform = get_transform(opt)
        
        # whether visulize the prediction during training
        if opt.use_vis:
            self.dir_B = opt.val_img_dir
            self.B_paths = make_dataset(self.dir_B)
            self.B_paths = sorted(self.B_paths)
            self.B_size = len(self.B_paths)
 
        


        self.anno = pickle.load(open(self.opt.target_anno_dir,'rb'))

        self.A_size = len(self.anno)
        self.index_list = np.arange(self.A_size)
        np.random.shuffle(self.index_list)
        
#         size of the heatmap should be input size / 4
        self.Hsize = self.opt.fineSize / 4
        
        
        
              
    

    def __getitem__(self, index):
        
        index_i = self.index_list[index]
        A_name = list(self.anno.keys())[index_i]
        A_path = os.path.join(self.dir_A, A_name)
        
        A_img = Image.open(A_path).convert('RGB')

        if A_img.size[0] != A_img.size[1]:
            A_img = A_img.resize((128, 128))
        A_img, keypoints = random_flip(A_img, self.anno[A_name])
                
        
        
        angle = random.randint(-self.opt.rot, self.opt.rot)
        A_img = A_img.rotate(angle)
        transformed_keypoints = transform(keypoints,(1/2, 1/2), 1,(1, 1),rot= angle).T
#         tt = t.copy()
#         tt[:,[0,1]] = t[:,[1,0]]
        heatmap = genertate_heatmap((self.Hsize,self.Hsize), transformed_keypoints * self.Hsize, 0.5 * self.opt.fineSize / 128)
        heatmap_tensor = torch.from_numpy(heatmap).to(dtype=torch.float32)
        
        skeleton = heatmap_tensor[-1,:,:].unsqueeze(0)
        
        A = self.transform(A_img)
        
        

        if self.opt.input_nc == 1:  # RGB to gray
            if A.shape[0] != 1:
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)
                
                
        if self.opt.use_vis:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform(B_img)
            if self.opt.input_nc == 1: 
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)
                
            return {'A': A, 'B': B,'A_pose':heatmap_tensor,'A_sk': skeleton,'A_name':A_name,'A_paths': A_path}
        
        return {'A': A,'A_pose':heatmap_tensor,'A_sk': skeleton,'A_name':A_name,'A_paths': A_path}

    def __len__(self):
        return max(self.A_size, 0)

    def name(self):
        return 'PoseDataset'
