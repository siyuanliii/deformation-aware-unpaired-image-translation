import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import pickle
import pandas as pd
from util.util import *
import copy


def getBaseGrid(N=64, normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N-1), (N), 2).to(dtype = torch.float32)
    if normalize:
        a = a/(N-1.0)
    x = a.repeat(N,1)
    y = x.t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid


class BinActive(torch.autograd.Function):

    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        
        res = ((input)>0.35).to(dtype = torch.float32)
#         ctx.save_for_backward(res)
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
            
        grad_input = grad_output.clone()
        return grad_input


class DeformTransferModel(BaseModel):
    def name(self):
        return 'DeformTransferModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_I', type=float, default=1e-5, help='weight for image synthesis gan loss')
            parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for shape deform gan loss')
            parser.add_argument('--lambda_identity', type=float, default=0, help='weight for reconstruction loss')
           
            parser.add_argument('--lambda_TV', type=float, default=0.01, help='weight for regloss')
            parser.add_argument('--lambda_bias', type=float, default=0.01, help='weight for regloss')
            parser.add_argument('--lambda_affine', type=float, default=0.01, help='weight for regloss')
            parser.add_argument('--updateThreshold', type=int, default=1, help='how many times G updates before update D')
            
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.init_mode:
            #self.loss_names = ['TV','bias','affine']
            self.loss_names = ['reg']
            visual_names_A = ['real_A', 'A_mask', 'transformed_Amask'] # 'transformed_Amask_D','B_mask_D'
            visual_names_B = []
            self.model_names = [ 'STN','G_S']
            self.optimizer_names = ['GS']
        else:
            self.loss_names = ['GS','D_S' ,'reg',
                          'D_I','rec','GI']
            self.optimizer_names = ['GS','GI','DS','DI']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
            if self.isTrain:
                visual_names_A = ['real_A','A_mask','transformed_Amask','fake_B','rec_B'] 
                visual_names_B = ['real_B','B_mask']
        # specify the models you want to save to the disk. 
                self.model_names = [ 'STN','G_S','D_S','G_I','D_I']
            else:
                visual_names_A = ['real_A','fake_B']
                if opt.render_pose:
                    visual_names_B = ['render_pose']
                else:
                    visual_names_B = []
                self.model_names = ['STN','G_S','G_I']
               
                
        self.visual_names = visual_names_A + visual_names_B

        #specify whether to transfer annotations across domain
        
        if not self.isTrain and self.opt.transfer_anno:
            self.source_anno = pickle.load(open(self.opt.source_anno_dir,'rb'))
            self.target_anno = {}
            self.keypoints_num = len(list(self.source_anno.values())[0])
            
        

        # load/define networks
        
        ##### generate_grid ##########
        self.base_grid = getBaseGrid(N=opt.fineSize,getbatch = False, batchSize = opt.batch_size).unsqueeze(0).to(self.device)
        
        ###### define network ########
        self.netSTN = networks.AffineNet(opt.input_nc).to(self.device)
        self.netG_S = networks.DeformationNet(opt.input_nc, 2, lb = 0, ub =0.1,norm_layer = nn.InstanceNorm2d).to(self.device)
        #self.sample_layer = networks.waspWarper(opt.batch_size, opt.fineSize)
        self.warp_integral = networks.SpatialGradientIntegration(opt.fineSize)
            
        self.thresh = BinActive.apply
        self.counter = 0
#         self.loss_D_S = 0
        
        if self.opt.init_mode or not self.isTrain:
            networks.init_weights(self.netSTN, 'normal',gain=0.002)
            networks.init_weights(self.netG_S, 'normal',gain=0.002)
        else:
            networks.load_network_weights(self.netSTN, opt.init_dir + '/latest_net_STN.pth')
            networks.load_network_weights(self.netG_S, opt.init_dir + '/latest_net_G_S.pth')
            
                
        self.netG_I = networks.define_G(1, opt.output_nc, opt.ngf, 'unet_128', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids 
                                        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
            self.netD_S = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids,dilation = opt.dilation)
        
            self.netD_I = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            1, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.opt.init_mode or not self.isTrain:
            networks.init_weights(self.netSTN, 'normal',gain=0.002)
            networks.init_weights(self.netG_S, 'normal',gain=0.002)
        else:
            networks.load_network_weights(self.netSTN, opt.init_dir + '/latest_net_STN.pth')
            networks.load_network_weights(self.netG_S, opt.init_dir + '/latest_net_G_S.pth') 
            
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_B_mask_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            self.criterionReg = networks.RegLoss(opt.lambda_TV, opt.lambda_bias, opt.lambda_affine) 

            # initialize optimizers
            self.optimizers = []
            if self.opt.init_mode:
                self.optimizer_GS = torch.optim.Adam([
                {'params': self.netG_S.parameters(), 'lr': opt.lr},
                {'params': self.netSTN.parameters(), 'lr': opt.lr},
                ],lr=opt.lr, betas=(opt.beta1, 0.999))
                
            else:
                self.optimizer_GS = torch.optim.Adam([
                {'params': self.netG_S.parameters(), 'lr': opt.lr_GS},
                {'params': self.netSTN.parameters(), 'lr': opt.lr_GS},
                ], 
                 lr=opt.lr, betas=(opt.beta1, 0.999))
                
                self.optimizer_GI = torch.optim.Adam([
                    {'params': self.netG_I.parameters(),'lr': opt.lr_GI},
                ], 
                 lr=opt.lr, betas=(opt.beta1, 0.999))
                
                self.optimizers.append(self.optimizer_GI)

            
            
                self.optimizer_DI = torch.optim.Adam([
                    {'params': self.netD_I.parameters(),'lr': opt.lr_DI},
            
                ], lr=opt.lr, betas=(opt.beta1, 0.999))
                
                self.optimizers.append(self.optimizer_DI)
                
                self.optimizer_DS = torch.optim.Adam([
                {'params': self.netD_S.parameters(), 'lr': opt.lr_DS},
                ], lr=opt.lr, betas=(opt.beta1, 0.999))
                
          
                self.optimizers.append(self.optimizer_DS)
            
            
            self.optimizers.append(self.optimizer_GS)
    


    def set_input(self, input):
        #AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.A_mask = input['A_mask'].to(self.device)
        self.B_mask = input['B_mask'].to(self.device)
        self.A_name = input['A_name'][0]
        self.image_paths = input['A_paths']
        if len(self.real_A.shape) == 3:
            self.real_A = self.real_A.unsqueeze(0)
            self.real_B = self.real_B.unsqueeze(0)
            self.A_mask = self.A_mask.unsqueeze(0)
            self.B_mask = self.B_mask.unsqueeze(0)
        
        
    def grid_sample(self, input, grid, integral = False):
        
        if not integral:
            output = F.grid_sample(input, grid,  padding_mode='border')
            return output
        else:
            warp = grid.permute(0,2,3,1)
            output = F.grid_sample(input, warp)
            return output
        
        

    def forward(self):
        
        
        batch_size = self.real_A.size(0)
    
        self.affine_grid = self.netSTN(self.real_A)
        self.real_A_affine = self.grid_sample(self.real_A, self.affine_grid)
        self.real_Amask_affine = self.grid_sample(self.A_mask, self.affine_grid)

        # the constant number can be changed or cancelled. (values used here for better integration reason.)
        self.diff_grid = self.netG_S(self.real_A_affine) * 50 / 128
        self.grid = self.warp_integral(self.diff_grid) - 1.01

        self.transformed_Amask = self.grid_sample(self.real_Amask_affine, self.grid, integral = True)
        self.transformed_A = self.grid_sample(self.real_A_affine, self.grid, integral = True)

        self.grid_vis = self.grid.permute(0,2,3,1)

        self.transformed_Amask = self.thresh(self.transformed_Amask)
        
        if self.opt.init_mode:
            pass
        else:
            self.rec_B = self.netG_I(self.B_mask)
            self.fake_B = self.netG_I(self.transformed_Amask.detach())


        if not self.isTrain and self.opt.transfer_anno:
            self.joints_map = genertate_heatmap((self.opt.fineSize,self.opt.fineSize), self.source_anno[self.A_name]*self.opt.fineSize, 1)
            self.joint_affine = self.grid_sample(torch.Tensor(self.joints_map).unsqueeze(0).to(self.device), self.affine_grid)
            self.joints_deform = self.grid_sample(self.joint_affine, self.grid, integral = True)
            self.keypoints = np.zeros((self.keypoints_num,2))
            for i in range(self.keypoints_num):
                index = self.joints_deform[0,i,:,:].argmax()
                self.keypoints[i,1] = float(index / self.opt.fineSize) / float(self.opt.fineSize)
                self.keypoints[i,0] = float(index % self.opt.fineSize ) / float(self.opt.fineSize)
            self.target_anno[self.A_name[:-4]+'_'+'fake_B.png'] = self.keypoints
            self.render_pose = copy.deepcopy(self.fake_B)
        
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
#         loss_D.backward()
        return loss_D

    def backward_DI(self):
#         fake_B = self.fake_B_pool.query(torch.cat((self.fake_B,self.transformed_Amask.detach()),1))
#         self.loss_D_I = self.backward_D_basic(self.netD_I, torch.cat((self.real_B,self.B_mask),1), fake_B)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_I = self.backward_D_basic(self.netD_I, self.real_B, fake_B)
        self.loss_D_I.backward()

        
    def backward_DS(self):
        fake_B_mask = self.fake_B_mask_pool.query(self.transformed_Amask)
        self.loss_D_S = self.backward_D_basic(self.netD_S, self.B_mask, fake_B_mask)
        self.loss_D_S.backward()
        

    def backward_GS(self):
        self.loss_G_S = self.criterionGAN(self.netD_S(self.transformed_Amask), True) * self.opt.lambda_S
        
        self.loss_reg = self.criterionReg(self.grid, self.affine_grid.permute(0,3,1,2), self.base_grid)
        
        self.loss_GS = self.loss_G_S + self.loss_reg
        self.loss_GS.backward()

    def backward_GI(self):

        self.loss_G_I = self.criterionGAN(self.netD_I(self.fake_B), True) * self.opt.lambda_I
        self.loss_rec = self.criterionL1(self.rec_B, self.real_B) * self.opt.lambda_identity

        self.loss_GI = self.loss_G_I + self.loss_rec
        self.loss_GI.backward()
           

    def optimize_parameters(self):
        self.counter+=1
        # forward
        self.forward()
        # G_I and G_S
        if self.opt.init_mode:
            self.set_requires_grad([self.netD_S, self.netD_I], False)
            self.optimizer_GS.zero_grad()
            self.backward_GS()
            self.optimizer_GS.step()
        else:
            self.set_requires_grad([self.netD_S, self.netD_I], False)
            self.optimizer_GS.zero_grad()
            self.backward_GS()
            self.optimizer_GS.step()
            
            self.optimizer_GI.zero_grad()
            self.backward_GI()
            self.optimizer_GI.step()
            
        if self.counter % self.opt.updateThreshold ==0 and not self.opt.init_mode:
        # D_I and D_S
            self.set_requires_grad([self.netD_S, self.netD_I], True)
            self.optimizer_DS.zero_grad()
            self.backward_DS()
            self.optimizer_DS.step()
            self.optimizer_DI.zero_grad()
            self.backward_DI()
            self.optimizer_DI.step()