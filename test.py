"""General-purpose test script for unpaired image translation and pose estimation


Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from data import CreateDataLoader
import pickle
import numpy as np
from util.util import *

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
     # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
   

     # create a website
    
    if opt.pose_mode: # if specified, test and save results for pose estimation
           
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, 'pose_' + opt.epoch))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    else:
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch_GS + '_' + opt.epoch_GI))  
        print('creating web directory', web_dir)        
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase,  opt.epoch_GS + '_' + opt.epoch_GI))
        
    # test with eval mode. This only affects layers like batchnorm and dropout.

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if not opt.pose_mode and i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, render = opt.render_pose, keypoints = model.keypoints, animal = opt.which_animal)
    
    # whether save the prediction annotation
    if opt.pose_mode and opt.transfer_anno:
        pickle.dump(model.target_anno,open(os.path.join(web_dir,'pred_anno.pth'),'wb')) 
    else:
        pickle.dump(model.target_anno,open(os.path.join(web_dir,'target_anno.pth'),'wb'))  
        
    # whether running evaluation (need annotations for test images)
    if opt.evaluate:
        test_anno = pickle.load(open(opt.target_anno_dir,'rb'))
        res = {'pck':[],'rmse': 0}
        anno_keys = list(model.target_anno.keys())
        for thres in range(opt.pck_range):
            all_pck = 0
            all_dist = 0
            for key in anno_keys:
                joints = np.array( test_anno[key]) * opt.fineSize
                p,d = single_animal_statistic(joints, model.target_anno[key]* opt.fineSize, thres = thres, animal = opt.which_animal)
                all_dist +=d
                all_pck += p
            res['pck'].append(all_pck / len(anno_keys))    
        res['rmse'] = all_dist / len(anno_keys)
        pickle.dump(res,open(os.path.join(web_dir,'eval_results.pth'),'wb'))   
    
    webpage.save()  # save the HTML
