from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matting import *
import config
import scipy.ndimage as spi
from wls_filter import wls_filter


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
    image.save("temp_result_{}.png".format(i), "PNG")

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

def nearest_interpolate(array, height, width):
    channel, ori_h, ori_w = array.shape
    ratio_h = ori_h / height
    ratio_w = ori_w / width
    # target_array = torch.zeros((channel, height, width))
    target_array = torch.cuda.FloatTensor(channel, height, width).fill_(0)
    for i in range(height):
        for j in range(width):
            th = int(i * ratio_h)
            tw = int(j * ratio_w)
            target_array[:, i, j] = array[:, th, tw]

    return target_array    


def compute_lap(path_img):
    '''
    input: image path
    output: laplacian matrix of the input image, format is sparse matrix of pytorch in gpu
    '''
    image = cv2.imread(path_img, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = 1.0 * image / 255.0
    h, w, _ = image.shape
    const_size = np.zeros(shape=(h, w))
    M = compute_laplacian(image)    
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().cuda()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cuda'))
    return Ms

def post_process(tensor, origin_image_path):
    unloader = transforms.ToPILImage() # tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)  # PIL image RGB, range[0 ~ 255]
    
    image = np.asarray(image)
    
    image = np.flip(image, 2)  # RGB2BGR
    
    guide_image = cv2.imread(origin_image_path, -1)
    guide_image = guide_image[:, :, :3]  # if alpha channel remove it
    
    # To 0.0 ~ 1.0
    image = 1.0 * image / 255.0
    guide_image = 1.0 * guide_image / 255.0

    result = wls_filter(image, guide_image, alpha=1.2, Lambda=1.0) + \
                guide_image - wls_filter(guide_image, guide_image, alpha=1.2, Lambda=1.0)
    result = np.clip(result, 0.0, 1.0)
    cv2.imwrite('final_result.png', (result*255.0).astype('uint8'))


