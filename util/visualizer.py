import numpy as np
import os
import copy 
import sys
import ntpath
import time
from . import util
from . import html
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import torch

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError
    
def visualize_patchGAN(source_img, feature_map, color = 'hot', alpha = 0.4):
    '''
    img: PIL image
    feature_map: the output of patch discriminator (h,w)
    color: the color map
    
    '''
    img = copy.deepcopy(source_img)

    H,W,_ = feature_map.shape
    step_h = img.size[1]/H
    step_w = img.size[0]/W
    drw = ImageDraw.Draw(img, 'RGBA')
    for i in range(H):
        for j in range(W):
            drw.rectangle([(j*step_h,i*step_w), ((j+1)*step_h, (i+1)*step_w)], tuple(feature_map[i,j,:]))
    del drw
    
    return img

def draw_deformation(source_image, grid, grid_size = 12):
    """
    source_image: PIL image object
    sample_grid: the sampling grid
    grid_size: the size of drawing grid
    """
    im = copy.deepcopy(source_image)
    d = ImageDraw.Draw(im)
    H,W = source_image.size
    dist =int(H/grid_size)
    for i in range(grid_size):
        step = int(dist*i)
        d.line(list(zip((grid[0,step,:,0].numpy()+1)/2*H, (grid[0,step,:,1].numpy()+1)/2*H)),fill = 255,width=1)
        d.line(list(zip((grid[0,:,step,0].numpy()+1)/2*H, (grid[0,:,step,1].numpy()+1)/2*H)),fill = 255,width=1)
    return im    
    

# save image to the disk
def save_images(webpage, visuals, image_path,aspect_ratio=1.0, width=256, render = False, animal = 'fly',keypoints = None):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
#         im_data=im_data[:,0,:,:].unsqueeze(0)

        im = util.tensor2im(im_data)
        
        image_name = '%s_%s.png' % (name, label)
        

        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = resize(im, (h, int(w * aspect_ratio)))
        if aspect_ratio < 1.0:
            im = resize(im, (int(h / aspect_ratio), w))
        
        if render:
            if label == 'render_pose':
                if animal == 'fly':
                    im = util.draw_skeleton(Image.fromarray(im), keypoints * h)
                    im = np.array(im)
                else:
                    im = util.draw_keypoints(Image.fromarray(im), keypoints * h)
                    im = np.array(im)
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


    
class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env, raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, sample_grid1 = None):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
#                 print("hw: ",str(h),str(w))
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                
                for label, image in visuals.items():
#                     image=image[:,0,:,:].unsqueeze(0)

                    image_numpy = util.tensor2im(image)
                    if(image_numpy.shape[0]!=128):
                        im_ = np.zeros((128,128,3))
                        im_[:image_numpy.shape[0],:image_numpy.shape[0],:] += image_numpy
                        image_numpy = im_
                    
                

                    ###### if the real images show the deformation######
                    if (type(sample_grid1)!= type(None) and label == 'real_A'):
                        image_pil = Image.fromarray(image_numpy)
                        image_withgrid = draw_deformation(image_pil, sample_grid1)
                        image_numpy = np.array(image_withgrid)
                        
                        
                    
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
#                     print(image_numpy.transpose([2, 0, 1]).shape)
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)

                if (type(sample_grid1)!= type(None) and label == 'real_A'):
                    image_pil = Image.fromarray(image_numpy)
                    image_withgrid = draw_deformation(image_pil, sample_grid1)
                    image_numpy = np.array(image_withgrid)

    
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data, best_pose_epoch = 0):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
#         message += ' best_mask_epoch: %d'%(best_mask_epoch)
#         message += ' best_img_epoch: %d'%(best_img_epoch)
        message += ' best_pose_epoch: %d'%(best_pose_epoch)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
