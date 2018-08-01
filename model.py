import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import copy

import config
import cv2
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
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

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

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

        self.loss = torch.sum(gx**2 + gy**2)/(height * width)
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
content_layers_default = ['conv_2'] # ['conv_2']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layer= content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(config.device0)

    # just in order to have an iterable access to or list of content.style losses
    content_losses = []
    style_losses = []
    tv_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn. Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    tv_loss = TVLoss()
    model.add_module("tv_loss_{}".format(0), tv_loss)
    tv_losses.append(tv_loss)
    i = 0 # increment every time we see a conv layer
    for layer in cnn.children():          # cnn feature  是没有 全连接层的
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layer:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses, tv_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=50, tv_weight=30):

    """Run the style transfer."""
    print("Buliding the style transfer model..")
    model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
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

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss: {:4f} Content Loss: {:4f} TV Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), tv_score.item()
                ))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last corrention...
    input_img.data.clamp_(0, 1)

    return input_img



