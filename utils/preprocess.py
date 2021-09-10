import random
import numpy
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

random.seed(2)
numpy.random.seed(2)
os.environ['PYTHONHASHSEED'] = str(2)

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'variance': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': tf.Variable([0.2175, 0.0188, 0.0045]),
    'eigvec': tf.Variable([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    processing = tf.keras.Sequential(
        [
            preprocessing.CenterCrop(input_size),
            preprocessing.Normalize(**normalize),
        ])
    if scale_size != input_size:
        processing.add(preprocessing.Rescaling(scale_size))

    return processing


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    processing = tf.keras.Sequential(
        [
            preprocessing.RandomCrop(input_size[1], input_size[2]),
            preprocessing.Normalization(**normalize),
        ])
    if scale_size != input_size:
        processing.add(preprocessing.Rescaling(scale_size))

    return processing


def get_transform(input_size=None, scale_size=None, normalize=None, augment=True):
    normalize = normalize or __imagenet_stats

    input_size = input_size or 32
    #if augment:
    #    scale_size = scale_size or 40
    #    return pad_random_crop(input_size, scale_size=scale_size,
    #                           normalize=normalize)
    #else:
    scale_size = scale_size or 32
    return scale_crop(input_size=input_size,
                      scale_size=scale_size, normalize=normalize)


# class Lighting(object):
#     """Lighting noise(AlexNet - style PCA - based noise)"""
#
#     def __init__(self, alphastd, eigval, eigvec):
#         self.alphastd = alphastd
#         self.eigval = eigval
#         self.eigvec = eigvec
#
#     def __call__(self, img):
#         if self.alphastd == 0:
#             return img
#
#         alpha = img.new().resize_(3).normal_(0, self.alphastd)
#         rgb = self.eigvec.type_as(img).clone()\
#             .mul(alpha.view(1, 3).expand(3, 3))\
#             .mul(self.eigval.view(1, 3).expand(3, 3))\
#             .sum(1).squeeze()
#
#         return img.add(rgb.view(3, 1, 1).expand_as(img))
#
#
# class Grayscale(object):
#
#     def __call__(self, img):
#         gs = img.clone()
#         gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
#         gs[1].copy_(gs[0])
#         gs[2].copy_(gs[0])
#         return gs
#
#
# class Saturation(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = Grayscale()(img)
#         alpha = random.uniform(0, self.var)
#         return img.lerp(gs, alpha)
#
#
# class Brightness(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = img.new().resize_as_(img).zero_()
#         alpha = random.uniform(0, self.var)
#         return img.lerp(gs, alpha)
#
#
# class Contrast(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = Grayscale()(img)
#         gs.fill_(gs.mean())
#         alpha = random.uniform(0, self.var)
#         return img.lerp(gs, alpha)
#
#
# class RandomOrder(object):
#     """ Composes several transforms together in random order.
#     """
#
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, img):
#         if self.transforms is None:
#             return img
#         order = torch.randperm(len(self.transforms))
#         for i in order:
#             img = self.transforms[i](img)
#         return img
#
#
# class ColorJitter(RandomOrder):
#
#     def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
#         self.transforms = []
#         if brightness != 0:
#             self.transforms.append(Brightness(brightness))
#         if contrast != 0:
#             self.transforms.append(Contrast(contrast))
#         if saturation != 0:
#             self.transforms.append(Saturation(saturation))
