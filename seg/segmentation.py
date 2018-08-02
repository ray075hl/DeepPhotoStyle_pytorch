# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from seg.dataset import TestDataset
from seg.models import ModelBuilder, SegmentationModule
from seg.utils import colorEncode
from seg.lib.nn import user_scattered_collate, async_copy_to
from seg.lib.utils import as_numpy, mark_volatile
import seg.lib.utils.data as torchdata

from PIL import Image
import torchvision.transforms as transforms


def segmentation(image_path):
    torch.cuda.set_device(0)
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50_dilated8',
        fc_dim=2048,
        weights='seg_checkpoint/encoder_epoch_20.pth')

    net_decoder = builder.build_decoder(
        arch='ppm_bilinear_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='seg_checkpoint/decoder_epoch_20.pth',
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()

    segmentation_module.eval()
    list_test = [{'fpath_img': image_path}]
    opt = {}
    opt['imgSize'] = [300, 400, 500, 600]
    opt['imgMaxSize'] = 1000
    opt['padding_constant'] = 8
    opt['segm_downsampling_rate'] = 8
    dataset_val = TestDataset(
        list_test, opt, max_sample=-1)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    for i, batch_data in enumerate(loader_val):
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])

        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            pred = torch.zeros(1, 150, segSize[0], segSize[1])

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img

                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, 0)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                pred = pred + pred_tmp.cpu() / 4
                '''
                for i in range(150):
                    pred_tmp
                    if torch.max(pred_tmp[0, i, :, :]) > 0.5:
                        print('pred_tmp:', i)
                '''
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0
    return pred.type(torch.IntTensor)


    
