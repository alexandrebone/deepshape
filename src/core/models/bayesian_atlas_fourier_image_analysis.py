### Base ###
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import fnmatch
from torch.utils.data import TensorDataset, DataLoader
import itertools
import math
from sklearn.decomposition import PCA
import nibabel as nib

# ### Visualization ###
# #import seaborn as sns
# #sns.set(color_codes=True)
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', **{'family':'serif','serif':['Palatino']})

### Deformetrica ###
import sys
path_to_deformetrica = '/home/alexandre.bone/Softwares/deformetrica'
sys.path.append(os.path.join(path_to_deformetrica, 'src'))
from in_out.xml_parameters import XmlParameters
from core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from in_out.deformable_object_reader import DeformableObjectReader
from in_out.dataset_functions import create_dataset
from in_out.array_readers_and_writers import *
from core.model_tools.deformations.exponential import Exponential

from api.deformetrica import Deformetrica
from deformetrica import get_dataset_specifications, get_estimator_options, get_model_options
from core.models.longitudinal_atlas import LongitudinalAtlas
import support.kernels as kernel_factory


def read_nii_image(path):
    img = nib.load(path)
    img_data = img.get_data()
    return torch.from_numpy(img_data).float()

def convolve(x, y, p, kernel_width):
    sq = squared_distances(x, y)
    return torch.mm(torch.exp(-sq / (kernel_width ** 2)), p)

def squared_distances(x, y):
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)

def scalar_product(x, y, p, q, kernel_width):
    return torch.sum(p * convolve(x, y, q, kernel_width))


def get_deformed_intensities(deformed_points, intensities, downsampling_factor):
    image_shape = intensities.shape
    deformed_voxels = deformed_points

    if not downsampling_factor == 1:
        shape = deformed_points.shape
        deformed_voxels = torch.nn.Upsample(size=intensities.shape, mode='trilinear', align_corners=True)(
            deformed_voxels.permute(3, 0, 1, 2).contiguous().view(
                1, shape[3], shape[0], shape[1], shape[2]))[0].permute(1, 2, 3, 0).contiguous()

    u, v, w = deformed_voxels.view(-1, 3)[:, 0], \
              deformed_voxels.view(-1, 3)[:, 1], \
              deformed_voxels.view(-1, 3)[:, 2]

    u1 = torch.floor(u.detach())
    v1 = torch.floor(v.detach())
    w1 = torch.floor(w.detach())

    u1 = torch.clamp(u1, 0, image_shape[0] - 1)
    v1 = torch.clamp(v1, 0, image_shape[1] - 1)
    w1 = torch.clamp(w1, 0, image_shape[2] - 1)
    u2 = torch.clamp(u1 + 1, 0, image_shape[0] - 1)
    v2 = torch.clamp(v1 + 1, 0, image_shape[1] - 1)
    w2 = torch.clamp(w1 + 1, 0, image_shape[2] - 1)

    fu = u - u1
    fv = v - v1
    fw = w - w1
    gu = u1 + 1 - u
    gv = v1 + 1 - v
    gw = w1 + 1 - w

    u1 = u1.long()
    v1 = v1.long()
    w1 = w1.long()
    u2 = u2.long()
    v2 = v2.long()
    w2 = w2.long()

    deformed_intensities = (intensities[u1, v1, w1] * gu * gv * gw +
                            intensities[u1, v1, w2] * gu * gv * fw +
                            intensities[u1, v2, w1] * gu * fv * gw +
                            intensities[u1, v2, w2] * gu * fv * fw +
                            intensities[u2, v1, w1] * fu * gv * gw +
                            intensities[u2, v1, w2] * fu * gv * fw +
                            intensities[u2, v2, w1] * fu * fv * gw +
                            intensities[u2, v2, w2] * fu * fv * fw).view(image_shape)
    return deformed_intensities


path_to_results = '/home/alexandre.bone/IPMI_2019/2_deformetrica/14_brain_pga__160_subjects/output'

template_i__pga = read_nii_image(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__Template_brain.nii'))
control_points__pga = read_3D_array(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__ControlPoints.txt'))
latent_positions__pga = read_3D_array(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))
principal_directions__pga = read_3D_array(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__PrincipalDirections.txt'))

downsampling_factor = 3
deformation_kernel_width = 5

z = torch.from_numpy(latent_positions__pga).float()
for d in range(3):
    cp = torch.from_numpy(control_points__pga).float()
    mom = torch.from_numpy(principal_directions__pga[d].reshape(control_points__pga.shape)).float()
    sp = scalar_product(cp, cp, mom, mom, deformation_kernel_width)
    z[:, d] *= sp
    print(math.sqrt(sp))

print(100 * torch.sum(z ** 2, dim=0) / torch.sum(z**2))




# Corner points
corner_points = np.zeros((8, 3))
umax, vmax, wmax = np.subtract(template_i__pga.shape, (1, 1, 1))
corner_points[0] = np.array([0, 0, 0])
corner_points[1] = np.array([umax, 0, 0])
corner_points[2] = np.array([0, vmax, 0])
corner_points[3] = np.array([umax, vmax, 0])
corner_points[4] = np.array([0, 0, wmax])
corner_points[5] = np.array([umax, 0, wmax])
corner_points[6] = np.array([0, vmax, wmax])
corner_points[7] = np.array([umax, vmax, wmax])

# Image points
image_shape = template_i__pga.shape
axes = []
for d in range(3):
    axe = np.linspace(corner_points[0, d], corner_points[2 ** d, d], image_shape[d] // downsampling_factor)
    axes.append(axe)

global_image_points = np.array(np.meshgrid(*axes, indexing='ij')[:])
for d in range(3):
    global_image_points = np.swapaxes(global_image_points, d, d + 1)

global_image_points = torch.from_numpy(global_image_points).float().cuda()

mult = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]

deformation_kernel_width = 5
number_of_time_points = 16
indexes = [0, 5, 10, 15]

images = []
for d in range(1):
    print('d = %d' % d)
    images_d = []

    latent_position = np.zeros((1, 3))
    latent_position[d] = 1
    momentum = np.dot(latent_position, principal_directions__pga).reshape(control_points__pga.shape)

    b_exponential = Exponential(
        kernel=kernel_factory.factory('keops', deformation_kernel_width),
        number_of_time_points=number_of_time_points,
        initial_control_points=torch.from_numpy(control_points__pga).float().cuda(),
        initial_momenta=torch.from_numpy(- momentum).float().cuda(),
        initial_template_points={'image_points': global_image_points})
    b_exponential.update()

    f_exponential = Exponential(
        kernel=kernel_factory.factory('keops', deformation_kernel_width),
        number_of_time_points=number_of_time_points,
        initial_control_points=torch.from_numpy(control_points__pga).float().cuda(),
        initial_momenta=torch.from_numpy(momentum).float().cuda(),
        initial_template_points={'image_points': global_image_points})
    f_exponential.update()

    images_d.append(
        get_deformed_intensities(b_exponential.get_template_points(15)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(b_exponential.get_template_points(10)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(b_exponential.get_template_points(5)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(b_exponential.get_template_points(0)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(f_exponential.get_template_points(5)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(f_exponential.get_template_points(10)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())
    images_d.append(
        get_deformed_intensities(f_exponential.get_template_points(15)['image_points'], template_i__pga.cuda(),
                                 downsampling_factor).detach().cpu().numpy())

    images.append(images_d)


print('done')
print('done')
print('done')



# ###########
# ### PGA ###
# ###########
#
# path_to_results = '/home/alexandre.bone/IPMI_2019/2_deformetrica/12_brains_pga_64/output'
# path_to_results_bis = '/home/alexandre.bone/IPMI_2019/2_deformetrica/12_brains_pga_registration_train/output'
#
# template_i__pga = read_nii_image(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__Template_brain.nii'))
# control_points__pga = read_3D_array(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__ControlPoints.txt'))
# latent_positions__pga = read_3D_array(os.path.join(path_to_results_bis, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))
# principal_directions__pga = read_3D_array(os.path.join(path_to_results, 'PrincipalGeodesicAnalysis__EstimatedParameters__PrincipalDirections.txt'))
#
# downsampling_factor = 2
#
# # Corner points
# corner_points = np.zeros((8, 3))
# umax, vmax, wmax = np.subtract(template_i__pga.shape, (1, 1, 1))
# corner_points[0] = np.array([0, 0, 0])
# corner_points[1] = np.array([umax, 0, 0])
# corner_points[2] = np.array([0, vmax, 0])
# corner_points[3] = np.array([umax, vmax, 0])
# corner_points[4] = np.array([0, 0, wmax])
# corner_points[5] = np.array([umax, 0, wmax])
# corner_points[6] = np.array([0, vmax, wmax])
# corner_points[7] = np.array([umax, vmax, wmax])
#
# # Image points
# image_shape = template_i__pga.shape
# axes = []
# for d in range(3):
#     axe = np.linspace(corner_points[0, d], corner_points[2 ** d, d], image_shape[d] // downsampling_factor)
#     axes.append(axe)
#
# global_image_points = np.array(np.meshgrid(*axes, indexing='ij')[:])
# for d in range(3):
#     global_image_points = np.swapaxes(global_image_points, d, d + 1)
#
# global_image_points = torch.from_numpy(global_image_points).float()
#
# # Load TRAIN and TEST data
# np.random.seed(42)
# number_of_brains = 50
# path_to_dataset = '/home/alexandre.bone/IPMI_2019/1_datasets/9_brains_64'
# prefix = 's'
#
# files_all = fnmatch.filter(os.listdir(path_to_dataset), '%s*' % prefix)
# files_all = np.array(sorted(files_all))
# files_train = files_all[np.random.choice(files_all.shape[0], size=number_of_brains, replace=None)]
# files_test = []
# for fl in files_all:
#     if fl not in files_train:
#         files_test.append(fl)
#
# # Train
# train_i = []
# for fl in files_train:
#     path_to_datum = os.path.join(path_to_dataset, fl)
#     i = read_nii_image(path_to_datum)
#     train_i.append(i)
# train_i = torch.stack(train_i)
#
# # Test
# test_i = []
# for fl in files_test:
#     path_to_datum = os.path.join(path_to_dataset, fl)
#     i = read_nii_image(path_to_datum)
#     test_i.append(i)
# test_i = torch.stack(test_i)
#
#
# # Reconstruction TRAIN
# path_to_residuals = '22_residuals_train_pga.txt'
# if os.path.isfile(path_to_residuals):
#     residuals_train__pga = np.loadtxt(path_to_residuals)
#
# else:
#     momenta_ = np.dot(latent_positions__pga, principal_directions__pga).reshape(
#         (latent_positions__pga.shape[0],) + control_points__pga.shape)
#
#     residuals_train__pga = []
#     for k, momentum in enumerate(momenta_):
#         print(k)
#         deformation_kernel_width = 4
#         number_of_time_points = 11
#
#         exponential = Exponential(
#             kernel=kernel_factory.factory('torch', deformation_kernel_width),
#             number_of_time_points=number_of_time_points,
#             initial_control_points=torch.from_numpy(control_points__pga).float(),
#             initial_momenta=torch.from_numpy(momentum).float(),
#             initial_template_points={'image_points': global_image_points})
#         exponential.update()
#         deformed_points = exponential.get_template_points()['image_points']
#         deformed_intensities = get_deformed_intensities(deformed_points, template_i__pga.cuda(), downsampling_factor)
#
#         residual = math.sqrt(float(torch.sum((deformed_intensities - train_i[k]) ** 2).detach().cpu().numpy()))
#         residuals_train__pga.append(residual)
#
#     residuals_train__pga = np.array(residuals_train__pga)
#
# print('residuals card   = %d' % len(residuals_train__pga))
# print('residuals mean   = %.3f' % np.mean(residuals_train__pga))
# print('residuals std    = %.3f' % np.std(residuals_train__pga))
#
# # Reconstruction TEST
# path_to_residuals = '22_residuals_test_pga.txt'
# if os.path.isfile(path_to_residuals):
#     residuals_train__pga = np.loadtxt(path_to_residuals)
#
# else:
#     momenta_ = np.dot(latent_positions__pga, principal_directions__pga).reshape(
#         (latent_positions__pga.shape[0],) + control_points__pga.shape)
#
#     residuals_test__pga = []
#     for k, momentum in enumerate(momenta_):
#         print(k)
#         deformation_kernel_width = 4
#         number_of_time_points = 11
#
#         exponential = Exponential(
#             kernel=kernel_factory.factory('torch', deformation_kernel_width),
#             number_of_time_points=number_of_time_points,
#             initial_control_points=torch.from_numpy(control_points__pga).float().cuda(),
#             initial_momenta=torch.from_numpy(momentum).float().cuda(),
#             initial_template_points={'image_points': global_image_points.cuda()})
#         exponential.update()
#         deformed_points = exponential.get_template_points()['image_points']
#         deformed_intensities = get_deformed_intensities(deformed_points, template_i__pga.cuda(), downsampling_factor)
#
#         residual = math.sqrt(float(torch.sum((deformed_intensities - test_i[k].cuda()) ** 2).detach().cpu().numpy()))
#         residuals_test__pga.append(residual)
#     residuals_test__pga = np.array(residuals_test__pga)
#     np.savetxt(path_to_residuals, residuals_test__pga)
#
# print('residuals card   = %d' % len(residuals_test__pga))
# print('residuals mean   = %.3f' % np.mean(residuals_test__pga))
# print('residuals std    = %.3f' % np.std(residuals_test__pga))


