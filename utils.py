from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matting import *
import config
import scipy.ndimage as spi
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

def compute_laplacian2(i_arr, consts, epsilon=1e-5, win_rad=1):
    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    a_sparse = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = scipy.sparse.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse

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




