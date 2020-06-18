import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import pickle
from util.util import *
import random
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM




if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    
    ## variable  ###
    total_steps = 0
    avg_dist = 0
    avg_pck = 0
    best_pck = 0
    best_dist = 9999
#     ssim_mask = {}
#     ssim_image = {}
#     best_ssim_mask = 0
#     best_ssim_img = 0
#     best_mask_epoch = 0
#     best_img_epoch = 0
    best_pose_epoch = 0
    

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
#         ssim_mask[opt.epoch_count] = 0
#         ssim_image[opt.epoch_count] = 0
        

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                print(total_steps)
                save_result = total_steps % opt.update_html_freq == 0
                
                # whether in pose estimation mode
                if opt.pose_mode:
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                else:
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, model.grid_vis.detach().cpu())

            if total_steps % opt.print_freq == 0:
        
                losses = model.get_current_losses()
                if opt.use_val and opt.epoch_count != epoch:
                    losses['val_rmse'] = avg_dist
                else:
                    losses['val_rmse'] = 0
                    
                    
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data, best_pose_epoch)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()


            
        
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        # Whether use validation for pose estimation. Due to limited available annotations, this option is turned off by default. 
        # You can annotation more images for validation purpose. Put val images and annotations in val_img_dir and val_anno_dir
        if opt.use_val:
            val_path = opt.val_img_dir
            val_anno = pickle.load(open(opt.val_anno_dir,'rb'))
            val_img_names = os.listdir(val_path)
            if opt.use_ssim:
                img_path = os.path.join(opt.dataroot, 'trainB')
                mask_path = os.path.join(opt.dataroot, 'maskB')
                val_mask_names = os.listdir(mask_path)
            
            model.netG_P.eval()
            all_dist = 0
            all_pck = 0
            for key in val_img_names:
                val_image = Image.open(os.path.join(val_path , key)).convert('RGB')
                if opt.input_nc == 1:
                    B = dataset.dataset.transform(val_image)
                    tmp_b = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                    B = tmp_b.unsqueeze(0).unsqueeze(0).to(model.device) 
                if opt.which_animal == 'fly':    
                    real_joints = np.array( val_anno['test'][key][:15]) * opt.fineSize 
                    real_joints = swapxy(real_joints)
                else:
                    real_joints = np.array( val_anno[key]) * opt.fineSize
                    
                with torch.no_grad():
                    heatmap_pred = model.netG_P(B)[-1]
                pred_joints = np.zeros((opt.output_nc,2))
                for j in range(opt.output_nc):
                    index = heatmap_pred[0,j,:,:].argmax()
                    pred_joints[j,0] = index/(opt.fineSize /4)
                    pred_joints[j,1] = index% int(opt.fineSize/4)
                pred_joints = pred_joints/(opt.fineSize /4) * opt.fineSize
                
                pck, d = single_animal_statistic(real_joints, pred_joints,thres =5, animal = opt.which_animal,compute_pck = False)
                
                all_dist +=d 
                all_pck += pck
            avg_dist = all_dist/len(val_img_names)
#             avg_pck = all_pck/len(val_img_names)
            if avg_dist <= best_dist:
                best_dist = avg_dist
                best_pose_epoch = epoch
                print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
                model.save_networks('best')
                 
            model.netG_P.train()
            
            

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        
        
        
