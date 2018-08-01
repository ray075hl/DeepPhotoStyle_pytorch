from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_image(path, size):

    image = Image.open(path)
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
