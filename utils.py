from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_image(path, size):

    image = Image.open(path)
    if size is None:
        pass
    else:
        image = image.resize((size, size), Image.BICUBIC)

    return image


def image_to_tensor(img):
    transform_ = transforms.Compose([transforms.ToTensor()])
    return transform_(img)


def show_pic(tensor, title=None):
    plt.figure()
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.title(title)

def save_pic(tensor, i):
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save("final_result{}.png".format(i), "PNG")

import torch

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def bilinear_interpolate_torch(im, x, y):
    print(im.size())
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
    
    
    return torch.t(torch.t(Ia)*wa) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

