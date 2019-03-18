### Core ###
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

### Keops ###
from pykeops.torch import Genred

### VTK ###
from vtk import vtkPolyData, vtkPoints, vtkDoubleArray, vtkPolyDataWriter


def cprint(str):
    print(str)
    return str + '\n'


def batched_vector_smoothing(vector, sigma, scaled=True):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    """
    kernel_size = int(5. * sigma + .5)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    dim = vector.size(1)
    if dim == 2:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / (2 * variance))
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv2d(2, 2, kernel_size, groups=2, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size).repeat(2, 1, 1, 1)

    elif dim == 3:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / (2 * variance))
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv3d(3, 3, kernel_size, groups=3, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(3, 1, 1, 1, 1)

    else:
        assert False, 'Impossible dimension.'

    filter.weight.data.requires_grad_(False)
    padded_vector = nn.functional.pad(vector, tuple([int(mean) for k in range(dim * 2)]), mode='reflect')
    return filter(padded_vector)


def compute_bounding_box(points):
    dimension = points.size(1)
    bounding_box = torch.zeros((dimension, 2))
    for d in range(dimension):
        bounding_box[d, 0] = torch.min(points[:, d])
        bounding_box[d, 1] = torch.max(points[:, d])
    return bounding_box


def compute_grid(bounding_box, margin=0.1, grid_size=64):
    bounding_box = bounding_box.numpy()
    dimension = bounding_box.shape[0]

    axes = []
    for d in range(dimension):
        mi = bounding_box[d, 0]
        ma = bounding_box[d, 1]
        length = ma - mi
        assert length > 0
        offset = margin * length
        axes.append(np.linspace(mi - offset, ma + offset, num=grid_size))

    grid = np.array(np.meshgrid(*axes, indexing='ij'))
    for d in range(dimension):
        grid = np.swapaxes(grid, d, d + 1)

    return torch.from_numpy(grid).float()


def check_and_adapt_kernel_widths(splatting_kernel_width, deformation_kernel_width,
                                  splatting_grid_size, deformation_grid_size,
                                  bounding_box):

    # Splatting.
    dx_splatting = np.max(bounding_box[:, 1] - bounding_box[:, 0]) / float(splatting_grid_size - 1)
    if dx_splatting <= splatting_kernel_width / 3.:
        print('>> [OK].      splatting_grid_step = %.2f %% splatting_kernel_width' %
              (100. * dx_splatting / splatting_kernel_width))
    elif dx_splatting > splatting_kernel_width:
        print('>> [OULALA].  splatting_grid_step = %.2f %% splatting_kernel_width' %
              (100. * dx_splatting / splatting_kernel_width))
    else:
        print('>> [WARNING]. splatting_grid_step = %.2f %% splatting_kernel_width' %
              (100. * dx_splatting / splatting_kernel_width))

    # Deformation.
    dx_deformation = np.max(bounding_box[:, 1] - bounding_box[:, 0]) / float(deformation_grid_size - 1)
    if dx_deformation <= deformation_kernel_width / 3.:
        print('>> [OK].      deformation_grid_step = %.2f %% deformation_kernel_width' %
              (100. * dx_splatting / deformation_kernel_width))
    elif dx_deformation > deformation_kernel_width:
        print('>> [OULALA].  deformation_grid_step = %.2f %% deformation_kernel_width' %
              (100. * dx_splatting / deformation_kernel_width))
    else:
        print('>> [WARNING]. deformation_grid_step = %.2f %% deformation_kernel_width' %
              (100. * dx_splatting / deformation_kernel_width))

    return splatting_kernel_width, deformation_kernel_width / dx_deformation



def bilinear_interpolation(velocity, points, bounding_box, grid_size):
    nb_of_points = points.size(0)
    dimension = points.size(1)

    x = points[:, 0]
    y = points[:, 1]

    u = (x - bounding_box[0, 0]) / (bounding_box[0, 1] - bounding_box[0, 0]) * (grid_size - 1)
    v = (y - bounding_box[1, 0]) / (bounding_box[1, 1] - bounding_box[1, 0]) * (grid_size - 1)

    u1 = torch.floor(u.detach())
    v1 = torch.floor(v.detach())

    u1 = torch.clamp(u1, 0, grid_size - 1)
    v1 = torch.clamp(v1, 0, grid_size - 1)
    u2 = torch.clamp(u1 + 1, 0, grid_size - 1)
    v2 = torch.clamp(v1 + 1, 0, grid_size - 1)

    fu = (u - u1).view(nb_of_points, 1).expand(nb_of_points, dimension)
    fv = (v - v1).view(nb_of_points, 1).expand(nb_of_points, dimension)
    gu = (u1 + 1 - u).view(nb_of_points, 1).expand(nb_of_points, dimension)
    gv = (v1 + 1 - v).view(nb_of_points, 1).expand(nb_of_points, dimension)

    u1 = u1.long()
    v1 = v1.long()
    u2 = u2.long()
    v2 = v2.long()

    velocity_on_points = (velocity[u1, v1] * gu * gv +
                          velocity[u1, v2] * gu * fv +
                          velocity[u2, v1] * fu * gv +
                          velocity[u2, v2] * fu * fv)
    return velocity_on_points


def batch_index_select(input, dim, index):
    """
    batch_index_select
    :param input: B x * x ... x *
    :param dim: 0 < scalar
    :param index: B x M
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def batched_bilinear_interpolation(velocity, points, bounding_box, grid_size):
    bts = points.size(0)
    nbp = points.size(1)
    dim = points.size(2)

    if dim == 2:
        velocity = velocity.permute(0, 2, 3, 1).view(bts, -1, 2)

        x = points[:, :, 0]
        y = points[:, :, 1]

        u = (x - bounding_box[0, 0]) / (bounding_box[0, 1] - bounding_box[0, 0]) * (grid_size - 1)
        v = (y - bounding_box[1, 0]) / (bounding_box[1, 1] - bounding_box[1, 0]) * (grid_size - 1)

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())

        u1 = torch.clamp(u1, 0, grid_size - 1)
        v1 = torch.clamp(v1, 0, grid_size - 1)
        u2 = torch.clamp(u1 + 1, 0, grid_size - 1)
        v2 = torch.clamp(v1 + 1, 0, grid_size - 1)

        fu = (u - u1).view(bts, nbp, 1).expand(bts, nbp, dim)
        fv = (v - v1).view(bts, nbp, 1).expand(bts, nbp, dim)
        gu = (u1 + 1 - u).view(bts, nbp, 1).expand(bts, nbp, dim)
        gv = (v1 + 1 - v).view(bts, nbp, 1).expand(bts, nbp, dim)

        u1 = u1.long()
        v1 = v1.long()
        u2 = u2.long()
        v2 = v2.long()

        velocity_on_points = (batch_index_select(velocity, 1, u1 * grid_size + v1) * gu * gv +
                              batch_index_select(velocity, 1, u1 * grid_size + v2) * gu * fv +
                              batch_index_select(velocity, 1, u2 * grid_size + v1) * fu * gv +
                              batch_index_select(velocity, 1, u2 * grid_size + v2) * fu * fv)

        return velocity_on_points

    elif dim == 3:
        velocity = velocity.permute(0, 2, 3, 4, 1).view(bts, -1, 3)

        x = points[:, :, 0]
        y = points[:, :, 1]
        z = points[:, :, 2]

        u = (x - bounding_box[0, 0]) / (bounding_box[0, 1] - bounding_box[0, 0]) * (grid_size - 1)
        v = (y - bounding_box[1, 0]) / (bounding_box[1, 1] - bounding_box[1, 0]) * (grid_size - 1)
        w = (z - bounding_box[2, 0]) / (bounding_box[2, 1] - bounding_box[2, 0]) * (grid_size - 1)

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())
        w1 = torch.floor(w.detach())

        u1 = torch.clamp(u1, 0, grid_size - 1)
        v1 = torch.clamp(v1, 0, grid_size - 1)
        w1 = torch.clamp(w1, 0, grid_size - 1)
        u2 = torch.clamp(u1 + 1, 0, grid_size - 1)
        v2 = torch.clamp(v1 + 1, 0, grid_size - 1)
        w2 = torch.clamp(w1 + 1, 0, grid_size - 1)

        fu = (u - u1).view(bts, nbp, 1).expand(bts, nbp, dim)
        fv = (v - v1).view(bts, nbp, 1).expand(bts, nbp, dim)
        fw = (w - w1).view(bts, nbp, 1).expand(bts, nbp, dim)
        gu = (u1 + 1 - u).view(bts, nbp, 1).expand(bts, nbp, dim)
        gv = (v1 + 1 - v).view(bts, nbp, 1).expand(bts, nbp, dim)
        gw = (w1 + 1 - w).view(bts, nbp, 1).expand(bts, nbp, dim)

        u1 = u1.long()
        v1 = v1.long()
        w1 = w1.long()
        u2 = u2.long()
        v2 = v2.long()
        w2 = w2.long()

        velocity_on_points = (
                batch_index_select(velocity, 1, u1 * grid_size ** 2 + v1 * grid_size + w1) * gu * gv * gw +
                batch_index_select(velocity, 1, u1 * grid_size ** 2 + v1 * grid_size + w2) * gu * gv * fw +
                batch_index_select(velocity, 1, u1 * grid_size ** 2 + v2 * grid_size + w1) * gu * fv * gw +
                batch_index_select(velocity, 1, u1 * grid_size ** 2 + v2 * grid_size + w2) * gu * fv * fw +
                batch_index_select(velocity, 1, u2 * grid_size ** 2 + v1 * grid_size + w1) * fu * gv * gw +
                batch_index_select(velocity, 1, u2 * grid_size ** 2 + v1 * grid_size + w2) * fu * gv * fw +
                batch_index_select(velocity, 1, u2 * grid_size ** 2 + v2 * grid_size + w1) * fu * fv * gw +
                batch_index_select(velocity, 1, u2 * grid_size ** 2 + v2 * grid_size + w2) * fu * fv * fw)

        return velocity_on_points

    else:
        raise RuntimeError


def convolutive_interpolation(velocity, points, deformation_grid, kernel_width):
    dimension = points.size(1)
    velocity_on_points = convolve(points, deformation_grid.contiguous().view(-1, dimension),
                                  velocity.view(-1, dimension), kernel_width)
    return velocity_on_points
