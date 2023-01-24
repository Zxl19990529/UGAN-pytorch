import torch
import torch.nn as nn 
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
import cv2
import numpy as np
from dataset import get_transforms
def tensor2img(one_tensor):# [b,c,h,w] [-1,1]
    tensor = one_tensor.squeeze(0) #[c,h,w] [0,1]
    tensor = (tensor*0.5 + 0.5)*255 # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu,dtype=np.uint8)
    img = np.transpose(img,(1,2,0))
    return img
def img2tensor(np_img):# [h,w,c]
    tensor = get_transforms()(np_img).cuda() # [c,h,w] [-1,1]
    tensor = tensor.unsqueeze(0) # [b,c,h,w] [-1,1]
    return tensor


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(module.weight.data,0.0,0.02)

def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x**2))/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y**2))/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl


def calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ]
    ).cuda()
    x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=3, padding=(1, 1)
    )
    return result


def calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=3, padding=(1, 1)
    )
    return result

def loss_igdl( correct_images, generated_images): # taken from https://github.com/Arquestro/ugan-pytorch/blob/master/ops/loss_modules.py
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=1)
    distances_x_gradient = pairwise_p_distance(
        correct_images_gradient_x, generated_images_gradient_x
    )
    distances_y_gradient = pairwise_p_distance(
        correct_images_gradient_y, generated_images_gradient_y
    )
    loss_x_gradient = torch.mean(distances_x_gradient)
    loss_y_gradient = torch.mean(distances_y_gradient)
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss
