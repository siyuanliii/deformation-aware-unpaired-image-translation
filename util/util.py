from __future__ import print_function
import torch
import numpy as np
import os
import copy
from PIL import Image, ImageDraw, ImageFont



# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))+1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
        
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def genertate_heatmap(imSize, keypoints, sigma):
    ######## keypoints represent (x, y) thus, (column, row)

    heatmap = np.zeros((keypoints.shape[0]+1, int(imSize[1]), int(imSize[0])))
    for i in range(keypoints.shape[0]):
        center = keypoints[i, :2]
        gaussian_map = heatmap[i,:, :]
        heatmap[i,:, :] = putGaussianMaps(
            center, gaussian_map, 1.3*sigma,imSize)

    heatmap[-1, :, :] = np.maximum(1 - np.max(heatmap[:keypoints.shape[0] ,:, :], axis=0), 0.)
    
    return heatmap

def putGaussianMaps(center, accumulate_confid_map,sigma, im_size):

    grid_y = im_size[1]
    grid_x = im_size[0]
    stride = 1
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    return accumulate_confid_map


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def draw_skeleton(img, keypoints, width = 1 , r = 1):
    color = ['#A64AC9','#FCCD04','#FFB48F','F5E6CC','17E9E0','86C232' ]
    im = copy.deepcopy(img)
    d = ImageDraw.Draw(im)
    num_legs = int(len(keypoints)/5)
    for i in range(num_legs):
        x = keypoints[np.arange(0+i*5,(i+1)*5,1).tolist(),1]
        y = keypoints[np.arange(0+i*5,(i+1)*5,1).tolist(),0]
        d.line(list(zip(y,x)), fill = hex_to_rgb(color[i]),width= width)
        for j in range(5):
            d.ellipse((y[j]-r, x[j]-r, y[j]+r, x[j]+r), fill = 'red', outline ='red')
    return im

def swapxy(joints):
    joints[:,[0,1]] = joints[:,[1,0]]
    return joints


def compute_PCK(leg, leg_f, thres = 25 ,r = [0,3,1], animal = 'fly'):
    if animal == 'fly':
        correct = 0
        for i in range(len(leg)):
            for j in range(5):
                if (point_dist(leg[i][j,:],leg_f[r[i]][j,:]) < thres):
                    correct+=1
        return correct/15
    else:
        correct = 0
        for i in range(len(leg)):
            if (point_dist(leg[i],leg_f[r[i]]) < thres):
                correct+=1
        return correct/len(leg)
    

def dist(a, b):
    res = np.sum(np.sqrt(np.sum(np.square(a - b), axis=1)))
    return res


def get_fly_legs(joints, real = False):
    
    '''
    type_ : 'real': 3 legs real fly annotation
            'fake': 6 legs fake fly annotation
    joints: np.array [n,2] n is number of joints
    '''
    leg =[]
#     if real:
    num_legs = int(len(joints)/5)
    for k in range(num_legs):
        leg.append(joints[np.arange(0+k*5,(k+1)*5,1).tolist(),:])
#     else:
#         for k in range(6):
#             leg.append(joints[np.arange(0+k*5,(k+1)*5,1).tolist(),:])
            
    return leg

def point_dist (x,y):
    return np.sqrt(np.sum(np.square(x-y)))

def single_animal_statistic(real_joints, predict_joints, thres=5, animal='fly',compute_pck=True):
    if animal == 'fly':
        leg_real = get_fly_legs(real_joints)
        leg_pred = get_fly_legs(predict_joints)
    else:
        leg_real = real_joints
        leg_pred = predict_joints
        
    avg_dist = 0
    avg_pck = 0
    index_ =[]
    
    for j in range(len(leg_real)):
        min_index = 0
        min_dist = 9999
        for m in range(len(leg_pred)):
            if animal == 'fly':
                d = dist(leg_pred[m], leg_real[j])
            else:
                d = point_dist(leg_pred[m], leg_real[j])
            if d < min_dist and m not in index_:
                min_dist = d
                min_index = m
        avg_dist += min_dist
        index_.append(min_index)
    if compute_pck:
        avg_pck = compute_PCK(leg_real, leg_pred, thres = thres ,r = index_, animal = animal)
    if animal == 'fly':
        avg_dist = avg_dist/15
    else:
        avg_dist = avg_dist/len(leg_real)
    return avg_pck, avg_dist

def draw_keypoints(img, keypoints):
    im = copy.deepcopy(img)
    d = ImageDraw.Draw(im)
    for j in range(len(keypoints)):
        d.ellipse((keypoints[j,0]-1, keypoints[j,1]-1, keypoints[j,0]+1, keypoints[j,1]+1), fill = 'red', outline ='red')
    return im

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