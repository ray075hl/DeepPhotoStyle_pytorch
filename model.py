import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
import torchvision.models as models
import copy

import config
import cv2
import utils
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        #print('*************: ', input.size(), self.target.size())
        if input.size() != self.target.size():
            pass
        else:
            channel, height, width = input.size()[1:4]
            self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a = batch size (=1)
    # b = number of feature maps
    # (c, d) = dimensions of a f. map (N=c*d)

    features = input.view(a*b, c*d)

    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c *d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask, content_mask):
        super(StyleLoss, self).__init__()

        self.style_mask = style_mask.detach()
        self.content_mask = content_mask.detach()

        #print(target_feature.type(), mask.type())
        _, channel_f, height, width = target_feature.size()
        channel = self.style_mask.size()[0]
        
        # ********
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2) 
        grid = grid.unsqueeze_(0).to(config.device0)
        mask_ = F.grid_sample(self.style_mask.unsqueeze(0), grid).squeeze(0)
        # ********       
        target_feature_3d = target_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        target_feature_masked = torch.zeros(size_of_mask, dtype=torch.float).to(config.device0)
        for i in range(channel):
            target_feature_masked[i, :, :, :] = mask_[i, :, :] * target_feature_3d

        self.targets = list()
        for i in range(channel):
            if torch.mean(mask_[i, :, :]) > 0.0:
                temp = target_feature_masked[i, :, :, :]
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach()/torch.mean(mask_[i, :, :]) )
            else:
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach())
    def forward(self, input_feature):
        self.loss = 0
        _, channel_f, height, width = input_feature.size()
        #channel = self.content_mask.size()[0]
        channel = len(self.targets)
        # ****
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).to(config.device0)
        mask = F.grid_sample(self.content_mask.unsqueeze(0), grid).squeeze(0)
        # ****
        #mask = self.content_mask.data.resize_(channel, height, width).clone()
        input_feature_3d = input_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        input_feature_masked = torch.zeros(size_of_mask, dtype=torch.float32).to(config.device0)
        for i in range(channel):
            input_feature_masked[i, :, :, :] = mask[i, :, :] * input_feature_3d
        
        inputs_G = list()
        for i in range(channel):
            temp = input_feature_masked[i, :, :, :]
            mask_mean = torch.mean(mask[i, :, :])
            if mask_mean > 0.0:
                inputs_G.append( gram_matrix(temp.unsqueeze(0))/mask_mean)
            else:
                inputs_G.append( gram_matrix(temp.unsqueeze(0)))
        for i in range(channel):
            mask_mean = torch.mean(mask[i, :, :])
            self.loss += F.mse_loss(inputs_G[i], self.targets[i]) * mask_mean
        
        return input_feature

class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()
        self.ky = np.array([
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]]
        ])
        self.kx = np.array([
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]]
        ])
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(torch.from_numpy(self.kx).float().unsqueeze(0).to(config.device0),
                                          requires_grad=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(self.ky).float().unsqueeze(0).to(config.device0),
                                          requires_grad=False)

    def forward(self, input):
        height, width = input.size()[2:4]
        gx = self.conv_x(input)
        gy = self.conv_y(input)

        # gy = gy.squeeze(0).squeeze(0)
        # cv2.imwrite('gy.png', (gy*255.0).to('cpu').numpy().astype('uint8'))
        # exit()

        self.loss = torch.sum(gx**2 + gy**2)/2.0
        return input

class RealLoss(nn.Module):
    
    def __init__(self, laplacian_m):
        super(RealLoss, self).__init__()
        self.L = Variable(laplacian_m.detach(), requires_grad=False)

    def forward(self, input):
        channel, height, width = input.size()[1:4]
        self.loss = 0
        for i in range(channel):
            temp = input[0, i, :, :]
            temp = torch.reshape(temp, (1, height*width))
            r = torch.mm(self.L, temp.t())
            self.loss += torch.mm(temp , r)
       
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses:
content_layers_default = ['conv4_2'] 
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_mask, content_mask, laplacian_m,
                               content_layer= content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(config.device0)

    # just in order to have an iterable access to or list of content.style losses
    content_losses = []
    style_losses = []
    tv_losses = []
    #real_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn. Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    tv_loss = TVLoss()
    model.add_module("tv_loss_{}".format(0), tv_loss)
    tv_losses.append(tv_loss)
    num_pool = 1
    num_conv = 0
    content_num = 0
    style_num = 0
    for layer in cnn.children():          # cnn feature without fully connected layers
        if isinstance(layer, nn.Conv2d):
            num_conv += 1
            name = 'conv{}_{}'.format(num_pool, num_conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(num_pool, num_conv)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(num_pool)
            num_pool += 1
            num_conv = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(num_pool, num_conv)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layer:
            # add content loss
            print('xixi: ', content_img.size())
            target = model(content_img).detach()
            #print('content target size: ', target.size())
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(content_num), content_loss)
            content_losses.append(content_loss)
            content_num += 1
        if name in style_layers:
            # add style loss:
            #print('style_:', style_img.type())
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, style_mask.detach(), content_mask.detach())
            model.add_module("style_loss_{}".format(style_num), style_loss)
            style_losses.append(style_loss)
            style_num += 1

    # now we trim off the layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses, tv_losses#, real_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    # optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer
'''
def manual_grad(image, laplacian_m):
    img = image.squeeze(0)
    channel, height, width = img.size() 
    
    loss = 0
    temp = img.reshape(3, -1)
    grad = torch.mm(laplacian_m, temp.t())
    
    loss += (grad * temp.t()).sum()
    return loss, None #2.*grad.reshape(img.size())
'''
def realistic_loss_grad(image, laplacian_m):
    img = image.squeeze(0)
    channel, height, width = img.size()
    loss = 0
    grads = list()
    for i in range(channel):
        grad = torch.mm(laplacian_m, img[i, :, :].reshape(-1, 1))
        loss += torch.mm(img[i, :, :].reshape(1, -1), grad)
        grads.append(grad.reshape((height, width)))
    gradient = torch.stack(grads, dim=0).unsqueeze(0)
    return loss, 2.*gradient


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, style_mask, content_mask, laplacian_m,
                       num_steps=3000,
                       style_weight=1000000, content_weight=100, tv_weight=0.0001, rl_weight=1):

    """Run the style transfer."""
    print("Buliding the style transfer model..")
    model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, style_mask, content_mask, laplacian_m)
    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")
    print('*'*20)
    print("Style_weith: {} Content_weighti: {} \
           TV_loss_weight: {} Realistic_loss_weight: {}".format \
           (style_weight, content_weight, tv_weight, rl_weight))
    print('*'*20)
    run = [0]
    
    best_loss = 1e10    
    best_input = input_img.data 

    while run[0] <= num_steps:

        def closure(): 
            nonlocal best_loss
            nonlocal input_img
            nonlocal best_input

            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0
            tv_score = 0
            
            
            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            for tl in tv_losses:
                tv_score += tl.loss
            
            style_score *= style_weight
            content_score *= content_weight
            tv_score *= tv_weight    

            # Two stage optimaztion pipline    
            if run[0] > num_steps // 2:
                # Realistic loss relate sparse matrix computing, 
                # which do not support autogard in pytorch, so we compute it separately.
                rl_score, part_grid = realistic_loss_grad(input_img, laplacian_m)
                rl_score *= rl_weight

                loss = style_score + content_score + tv_score + rl_score

                # Store the best result for outputing
                if loss < best_loss:
                    # print(best_loss)
                    best_loss = loss
                    best_input = input_img.data.clone()
            else:
                loss = style_score + content_score + tv_score

                rl_score = torch.zeros(1) # Just to print

                if loss < best_loss and run[0] > 0:
                    # print(best_loss)
                    best_loss = loss
                    best_input = input_img.data

                if run[0] == num_steps // 2:
                    # Store the best temp result to initialize second stage input
                    input_img.data = best_input
                    best_loss = 1e10

            loss.backward()
            
            # Gradient cliping deal with gradient exploding
            clip_grad_norm(model.parameters(), 15.0)
          
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}/{}:".format(run, num_steps))
        
                print('Style Loss: {:4f} Content Loss: {:4f} TV Loss: {:4f} real loss: {:4f}'.format(
                   style_score.item(), content_score.item(), tv_score.item(), rl_score.item()))

                print('Total Loss: ', loss.item())
              
                saved_img = input_img.clone() 
                saved_img.data.clamp_(0, 1)
                utils.save_pic(saved_img, run[0])
            return loss

        optimizer.step(closure)
              
    # a last corrention...
    input_img.data = best_input
    input_img.data.clamp_(0, 1)
    
    return input_img


