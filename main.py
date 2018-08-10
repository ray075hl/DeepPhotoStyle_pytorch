import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import matplotlib.pyplot as plt

import argparse

# ------custom module----
import config
import utils

sys.path.append('seg')
from seg.segmentation import *
from model import *
from merge_index import *


def gen_mask(image_path):
    """
    Generate semantic mask
    """
    seg_result = segmentation(image_path).squeeze(0)
    channel, height_, width_ = seg_result.size()

    for classes in merge_classes:
        for index, each_class in enumerate(classes):
            if index == 0:
                zeros_index = each_class
                base_map = seg_result[each_class, :, :].clone()
            else:
                base_map = base_map | seg_result[each_class, :, :]
        seg_result[zeros_index, :, :] = base_map

    return seg_result, height_, width_

if __name__ == '__main__':
    
    #----------init------------
    ap = argparse.ArgumentParser()

    ap.add_argument("-s", "--style_image", required=True, 
        help="path of the style image")

    ap.add_argument("-c", "--content_image", required=True,
        help="path of the content image")

    args = vars(ap.parse_args())

    style_image_path = args["style_image"]
    content_image_path = args["content_image"]

    #-------------------------
    print('Computing Laplacian matrix of content image')
    L = utils.compute_lap(content_image_path)
    print()
    #-------------------------
    print('Merge the similar semantic mask')
    style_mask_origin, height_, width_ = gen_mask(style_image_path)
    content_mask_origin, height2, width2 = gen_mask(content_image_path)

    merged_style_mask = np.zeros((117, height_, width_), dtype='int')
    merged_content_mask = np.zeros((117, height2, width2), dtype='int')
    print()
    #--------------------------
    count = 0
    for i in range(150):
        temp = style_mask_origin[i, :, :].numpy()
        if i not in del_classed and np.sum(temp)>50:
            # print(count, np.sum(temp))
            merged_style_mask[count, :, :] = temp
            merged_content_mask[count, :, :] = content_mask_origin[i, :, :].numpy()
            count += 1
        else:
            pass
    print('Total semantic classes in style image: {}'.format(count))
    style_mask_tensor = torch.from_numpy(merged_style_mask[:count, :, :]).float().to(config.device0)
    content_mask_tensor = torch.from_numpy(merged_content_mask[:count, :, :]).float().to(config.device0)
    #--------------------------
    print('Save each mask as an image for debugging')
    for i in range(count):
        utils.save_pic( torch.stack([style_mask_tensor[i, :, :], style_mask_tensor[i, :, :], style_mask_tensor[i, :, :]], dim=0), 
                                    'style_mask_' + str(i) )
        utils.save_pic( torch.stack([content_mask_tensor[i, :, :], content_mask_tensor[i, :, :], content_mask_tensor[i, :, :]], dim=0), 
                                    'content_mask_' + str(i) )
    
    # Using GPU or CPU
    device = torch.device(config.device0)


    style_img = utils.load_image(style_image_path, None)
    content_img = utils.load_image(content_image_path, None)
    width_s, height_s = style_img.size
    width_c, height_c = content_img.size
    
    # print(height_s, width_s)
    # print(height_c, width_c)
    
    style_img = utils.image_to_tensor(style_img).unsqueeze(0)
    content_img = utils.image_to_tensor(content_img).unsqueeze(0)

    style_img = style_img.to(device, torch.float)
    content_img = content_img.to(device, torch.float)
    
    # print('content_img size: ', content_img.size())
    # utils.show_pic(style_img, 'style image')
    # utils.show_pic(content_img, 'content image')

    # -------------------------
    # Eval() means the parameters of cnn are frozen.
    cnn = models.vgg19(pretrained=True).features.to(config.device0).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(config.device0)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(config.device0)
    
    # Two different initialization ways
    input_img = torch.randn(1, 3, height_c, width_c).to(config.device0)
    # input_img = content_img.clone()
    # print('input_img size: ', input_img.size())
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                                style_mask_tensor, content_mask_tensor, L)
    print('Style transfer completed')
    utils.save_pic(output, 'deep_style_tranfer')
    print()

    #--------------------------
    print('Postprocessing......')
    utils.post_process(output, content_image_path)
    print('Done!')




