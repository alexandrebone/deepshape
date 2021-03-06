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
import nibabel as nib


def resize_32_32(image):
    a, b = image.shape
    return zoom(image, zoom=(32 * 1. / a, 32 * 1. / b))


def load_mnist(number_of_images_train, number_of_images_test, digit=2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    number_of_images = number_of_images_train + number_of_images_test + 1
    mnist_data = datasets.MNIST('./', train=True, download=True)

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


def create_cross_sectional_brains_dataset__64(number_of_images_train, number_of_images_test, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/data'))
    path_to_template = os.path.join(path_to_data, 'colin27.nii')

    # CN.
    number_of_datum = number_of_images_train + number_of_images_test
    files = fnmatch.filter(os.listdir(path_to_data), 's*.nii')
    files = np.array(sorted(files))
    assert number_of_datum <= files.shape[0], \
        'Too many required CN brains. A maximum of %d are available' % files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

    intensities_cn = []
    for k, fl in enumerate(files__rdm):
        path_to_datum = os.path.join(path_to_data, fl)
        intensities_cn.append(torch.from_numpy(nib.load(path_to_datum).get_data()).float())
    intensities = torch.stack(intensities_cn).unsqueeze(1)

    intensities_train = intensities[:number_of_images_train]
    intensities_test = intensities[number_of_images_train:]
    intensities_template = torch.from_numpy(nib.load(path_to_template).get_data()).float().unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train[:, :, :, :, 32] - intensities_mean) / intensities_std
    intensities_test = (intensities_test[:, :, :, :, 32] - intensities_mean) / intensities_std
    intensities_template = (intensities_template[:, :, :, 32] - intensities_mean) / intensities_std

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


def create_cross_sectional_brains_dataset__128(number_of_datum_train, number_of_datum_test, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    path_to_data = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../examples/brains/data_128'))
    path_to_template = os.path.join(path_to_data, 'colin27.npy')

    number_of_datum_train_ = (number_of_datum_train // 3 + number_of_datum_train % 3,
                              number_of_datum_train // 3, number_of_datum_train // 3)
    number_of_datum_test_ = (number_of_datum_test // 3 + number_of_datum_test % 3,
                             number_of_datum_test // 3, number_of_datum_test // 3)

    print('>> TRAIN: %d CN ; %d AD ; %d MCI' %
          (number_of_datum_train_[0], number_of_datum_train_[1], number_of_datum_train_[2]))
    print('>> TEST : %d CN ; %d AD ; %d MCI' %
          (number_of_datum_test_[0], number_of_datum_test_[1], number_of_datum_test_[2]))

    # CN.
    number_of_datum = number_of_datum_train_[0] + number_of_datum_test_[0]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'cn')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required CN brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_cn = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            intensities_cn.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_cn = torch.stack(intensities_cn)

    # AD.
    number_of_datum = number_of_datum_train_[1] + number_of_datum_test_[1]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'ad')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required AD brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_ad = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            intensities_ad.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_ad = torch.stack(intensities_ad)

    # MCI.
    number_of_datum = number_of_datum_train_[2] + number_of_datum_test_[2]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'mci')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required MCI brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_mci = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            intensities_mci.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_mci = torch.stack(intensities_mci)

    intensities_train = torch.cat((intensities_cn[:number_of_datum_train_[0]],
                                   intensities_ad[:number_of_datum_train_[1]],
                                   intensities_mci[:number_of_datum_train_[2]]), dim=0).unsqueeze(1)
    intensities_test = torch.cat((intensities_cn[number_of_datum_train_[0]:],
                                  intensities_ad[number_of_datum_train_[1]:],
                                  intensities_mci[number_of_datum_train_[2]:]), dim=0).unsqueeze(1)
    intensities_template = torch.from_numpy(np.load(path_to_template)).float().unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train[:, :, :, :, 60] - intensities_mean) / intensities_std
    intensities_test = (intensities_test[:, :, :, :, 60] - intensities_mean) / intensities_std
    intensities_template = (intensities_template[:, :, :, 60] - intensities_mean) / intensities_std

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std
