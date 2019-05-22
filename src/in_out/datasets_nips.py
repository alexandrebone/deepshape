### Base ###
import fnmatch
import os

### Core ###
import numpy as np
import torch

### IMPORTS ###
from in_out.data_nips import *
from torchvision import datasets
from scipy.ndimage import zoom


def resize_32_32(image):
    a, b = image.shape
    return zoom(image, zoom=(32*1./a, 32*1./b))


def load_mnist(number_of_images_train, number_of_images_test, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    number_of_images = number_of_images_train + number_of_images_test + 1
    mnist_data = datasets.MNIST('./', train=True, download=True)

    digit = 2
    mnist_data = mnist_data.train_data[mnist_data.train_labels == digit]

    assert number_of_images <= mnist_data.shape[0], \
        'Too many required files. A maximum of %d are available' % mnist_data.shape[0]

    mnist_data__rdm = mnist_data[np.random.choice(mnist_data.shape[0], size=number_of_images, replace=None)]

    intensities = []
    for k, mnist_datum in enumerate(mnist_data__rdm):
        img = mnist_datum
        resized_img = resize_32_32(img)
        intensities.append(torch.from_numpy(resized_img).float())

    intensities = torch.stack(intensities)

    intensities_train = intensities[:number_of_images_train].unsqueeze(1)
    intensities_test = intensities[number_of_images_train:number_of_images_train + number_of_images_test].unsqueeze(1)
    intensities_template = intensities[-1].unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train - intensities_mean) / intensities_std
    intensities_test = (intensities_test - intensities_mean) / intensities_std
    intensities_template = (intensities_template - intensities_mean) / intensities_std

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std
