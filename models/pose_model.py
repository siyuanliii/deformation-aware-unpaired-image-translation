import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import copy


class PoseModel(BaseModel):
    def name(self):
        return 'PoseModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['pose']
     
        if self.isTrain:
            # specify the images you want to save/display. The program will call base_model.get_current_visuals
            
            visual_names_A = ['real_A','A_sk','pose'] 
            if self.opt.use_vis:
                visual_names_B = ['real_B','test_output']
            else:
                visual_names_B = []
            self.visual_names = visual_names_A + visual_names_B
            
            self.model_names = ['G_P']
            self.optimizer_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G_P']
            
            self.visual_names = ['real_A', 'render_pose']
            self.target_anno = {}

        # load/define networks
                
        self.netG_P = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'pose', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                        use_sigmoid =True)
            
            

        if self.isTrain:
            

            self.criterion = networks.JointsMSELoss(use_target_weight = False, use_switch = False)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_P.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)



    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        if self.opt.use_vis:
            self.real_B = input['B'].to(self.device)
        self.A_pose = input['A_pose'].to(self.device)
        self.A_sk = input['A_sk'].to(self.device)
        self.A_name = input['A_name'][0]
        self.image_paths = input['A_paths']
        

    def forward(self):
        
        self.output = self.netG_P(self.real_A)
        
        if self.isTrain:
            if self.opt.use_vis:
                self.test = self.netG_P(self.real_B)
            
            if type(self.output) == list:  
                self.pose = self.output[-1][:,-1,:,:].unsqueeze(1)
                if self.opt.use_vis:
                    self.test_output = self.test[-1][:,-1,:,:].unsqueeze(1)
                self.final_output = self.output[-1]
            else:
                self.pose = self.output[:,-1,:,:].unsqueeze(1)
                self.final_output = self.output
        
        if not self.isTrain:
            self.pred_heatmap = self.output[-1]
            H = float(self.pred_heatmap.shape[-2])
            W = float(self.pred_heatmap.shape[-1])
            self.keypoints = np.zeros((self.opt.output_nc,2))
            for j in range(self.opt.output_nc):
                index = self.pred_heatmap[0,j,:,:].argmax()
                self.keypoints[j,1] = float(index / H) / H
                self.keypoints[j,0] = float(index % W) / W
            self.target_anno[self.A_name] = self.keypoints
            self.render_pose = copy.deepcopy(self.real_A)
        

    def backward_G(self):
    
        with torch.autograd.set_detect_anomaly(True):
            self.loss_pose = 0
            if type(self.output) == list:  # multiple output
                for o in self.output:
                    self.loss_pose += self.criterion(o, self.A_pose[:,:,:,:])

            self.loss_pose.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
