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
    pads = [[(kernel_size - 1) // 2, kernel_size // 2] for k in range(dim)]
    pads = [item for sublist in pads for item in sublist]

    if dim == 2:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / variance)
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv2d(2, 2, kernel_size, groups=2, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size).repeat(2, 1, 1, 1)
        # padded_vector = nn.functional.pad(vector, tuple(pads), mode='reflect')
        padded_vector = nn.functional.pad(vector, tuple(pads), mode='constant', value=0)
        # padded_vector = nn.functional.pad(vector, tuple(pads), mode='replicate', value=0)

    elif dim == 3:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / variance)
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv3d(3, 3, kernel_size, groups=3, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(3, 1, 1, 1, 1)
        padded_vector = nn.functional.pad(vector, tuple(pads), mode='constant', value=0)

    else:
        assert False, 'Impossible dimension.'

    filter.weight.data.requires_grad_(False)
    return filter(padded_vector)


def batched_scalar_smoothing(vector, sigma, scaled=True):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    """
    kernel_size = int(5. * sigma + .5)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    dim = len(vector.size()) - 2
    pads = [[(kernel_size - 1) // 2, kernel_size // 2] for k in range(dim)]
    pads = [item for sublist in pads for item in sublist]

    if dim == 2:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / variance)
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv2d(1, 1, kernel_size, groups=1, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        # padded_vector = nn.functional.pad(vector, tuple(pads), mode='reflect')
        padded_vector = nn.functional.pad(vector, tuple(pads), mode='constant', value=0)
        # padded_vector = nn.functional.pad(vector, tuple(pads), mode='replicate', value=0)

    elif dim == 3:
        grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                           torch.arange(kernel_size),
                                           torch.arange(kernel_size)]), dim=-1).float().type(str(vector.type()))
        weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / variance)
        if scaled:
            weights /= torch.sum(weights)
        filter = nn.Conv3d(1, 1, kernel_size, groups=1, bias=False)
        filter.weight.data = weights.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(1, 1, 1, 1, 1)
        padded_vector = nn.functional.pad(vector, tuple(pads), mode='constant', value=0)

    else:
        assert False, 'Impossible dimension.'

    filter.weight.data.requires_grad_(False)
    return filter(padded_vector)


def batched_vector_interpolation(vector, points, downsampling_factor=1):
    bts = points.size(0)
    dim = points.size(1)
    dgs = points.size(2)
    nbp = dgs ** dim

    if dim == 2:

        points = points.permute(0, 2, 3, 1).view(bts, -1, 2)
        vector = vector.permute(0, 2, 3, 1).view(bts, -1, 2)

        x = points[:, :, 0]
        y = points[:, :, 1]

        u = (x + 1.0) / float(downsampling_factor) - 1.0
        v = (y + 1.0) / float(downsampling_factor) - 1.0

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())

        u1 = torch.clamp(u1, 0.0, dgs - 1.0)
        v1 = torch.clamp(v1, 0.0, dgs - 1.0)
        u2 = torch.clamp(u1 + 1, 0.0, dgs - 1.0)
        v2 = torch.clamp(v1 + 1, 0.0, dgs - 1.0)

        fu = (u - u1).view(bts, nbp, 1).expand(bts, nbp, dim)
        fv = (v - v1).view(bts, nbp, 1).expand(bts, nbp, dim)
        gu = (u1 + 1 - u).view(bts, nbp, 1).expand(bts, nbp, dim)
        gv = (v1 + 1 - v).view(bts, nbp, 1).expand(bts, nbp, dim)

        u1 = u1.long()
        v1 = v1.long()
        u2 = u2.long()
        v2 = v2.long()

        vector_on_grid = (
                batch_index_select(vector, 1, u1 * dgs + v1) * gu * gv +
                batch_index_select(vector, 1, u1 * dgs + v2) * gu * fv +
                batch_index_select(vector, 1, u2 * dgs + v1) * fu * gv +
                batch_index_select(vector, 1, u2 * dgs + v2) * fu * fv)
        vector_on_grid = vector_on_grid.view(bts, dgs, dgs, dim).permute(0, 3, 1, 2)

    elif dim == 3:

        points = points.permute(0, 2, 3, 4, 1).view(bts, -1, 3)
        vector = vector.permute(0, 2, 3, 4, 1).view(bts, -1, 3)

        x = points[:, :, 0]
        y = points[:, :, 1]
        z = points[:, :, 2]

        u = (x + 1.0) / float(downsampling_factor) - 1.0
        v = (y + 1.0) / float(downsampling_factor) - 1.0
        w = (z + 1.0) / float(downsampling_factor) - 1.0

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())
        w1 = torch.floor(w.detach())

        u1 = torch.clamp(u1, 0.0, dgs - 1.0)
        v1 = torch.clamp(v1, 0.0, dgs - 1.0)
        w1 = torch.clamp(w1, 0.0, dgs - 1.0)
        u2 = torch.clamp(u1 + 1, 0.0, dgs - 1.0)
        v2 = torch.clamp(v1 + 1, 0.0, dgs - 1.0)
        w2 = torch.clamp(w1 + 1, 0.0, dgs - 1.0)

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

        vector_on_grid = (
                batch_index_select(vector, 1, u1 * dgs ** 2 + v1 * dgs + w1) * gu * gv * gw +
                batch_index_select(vector, 1, u1 * dgs ** 2 + v1 * dgs + w2) * gu * gv * fw +
                batch_index_select(vector, 1, u1 * dgs ** 2 + v2 * dgs + w1) * gu * fv * gw +
                batch_index_select(vector, 1, u1 * dgs ** 2 + v2 * dgs + w2) * gu * fv * fw +
                batch_index_select(vector, 1, u2 * dgs ** 2 + v1 * dgs + w1) * fu * gv * gw +
                batch_index_select(vector, 1, u2 * dgs ** 2 + v1 * dgs + w2) * fu * gv * fw +
                batch_index_select(vector, 1, u2 * dgs ** 2 + v2 * dgs + w1) * fu * fv * gw +
                batch_index_select(vector, 1, u2 * dgs ** 2 + v2 * dgs + w2) * fu * fv * fw)
        vector_on_grid = vector_on_grid.view(bts, dgs, dgs, dgs, dim).permute(0, 4, 1, 2, 3)

    else:
        assert False, 'Impossible dimension'

    return vector_on_grid


def batched_scalar_interpolation(scalars, points):
    bts = points.size(0)
    dim = points.size(1)

    if dim == 2:
        # UPSAMPLE
        assert scalars.size(2) >= points.size(2)
        dsf = scalars.size(2) // points.size(2)
        if not dsf == 1:
            points = nn.functional.interpolate(points, scale_factor=dsf, mode='bilinear', align_corners=True)
        gs = points.size(2)

        u = points[:, 0]
        v = points[:, 1]

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())

        u1 = torch.clamp(u1, 0, gs - 1)
        v1 = torch.clamp(v1, 0, gs - 1)
        u2 = torch.clamp(u1 + 1, 0, gs - 1)
        v2 = torch.clamp(v1 + 1, 0, gs - 1)

        fu = (u - u1).view(bts, 1, gs, gs)
        fv = (v - v1).view(bts, 1, gs, gs)
        gu = (u1 + 1 - u).view(bts, 1, gs, gs)
        gv = (v1 + 1 - v).view(bts, 1, gs, gs)

        u1 = u1.long()
        v1 = v1.long()
        u2 = u2.long()
        v2 = v2.long()

        scalars_on_points = (batch_index_select(scalars.view(bts, -1), 1, u1 * gs + v1).view(bts, 1, gs, gs) * gu * gv +
                             batch_index_select(scalars.view(bts, -1), 1, u1 * gs + v2).view(bts, 1, gs, gs) * gu * fv +
                             batch_index_select(scalars.view(bts, -1), 1, u2 * gs + v1).view(bts, 1, gs, gs) * fu * gv +
                             batch_index_select(scalars.view(bts, -1), 1, u2 * gs + v2).view(bts, 1, gs, gs) * fu * fv)

    elif dim == 3:

        # UPSAMPLE
        assert scalars.size(2) >= points.size(2)
        dsf = scalars.size(2) // points.size(2)
        if not dsf == 1:
            points = nn.functional.interpolate(points, scale_factor=dsf, mode='trilinear', align_corners=True)
        gs = points.size(2)

        u = points[:, 0]
        v = points[:, 1]
        w = points[:, 2]

        u1 = torch.floor(u.detach())
        v1 = torch.floor(v.detach())
        w1 = torch.floor(w.detach())

        u1 = torch.clamp(u1, 0, gs - 1)
        v1 = torch.clamp(v1, 0, gs - 1)
        w1 = torch.clamp(w1, 0, gs - 1)
        u2 = torch.clamp(u1 + 1, 0, gs - 1)
        v2 = torch.clamp(v1 + 1, 0, gs - 1)
        w2 = torch.clamp(w1 + 1, 0, gs - 1)

        fu = (u - u1)
        fv = (v - v1)
        fw = (w - w1)
        gu = (u1 + 1 - u)
        gv = (v1 + 1 - v)
        gw = (w1 + 1 - w)

        u1 = u1.long()
        v1 = v1.long()
        w1 = w1.long()
        u2 = u2.long()
        v2 = v2.long()
        w2 = w2.long()

        scalars_on_points = (
                scalars[0, u1.view(-1), v1.view(-1), w1.view(-1)] * gu.view(-1) * gv.view(-1) * gw.view(-1) +
                scalars[0, u1.view(-1), v1.view(-1), w2.view(-1)] * gu.view(-1) * gv.view(-1) * fw.view(-1) +
                scalars[0, u1.view(-1), v2.view(-1), w1.view(-1)] * gu.view(-1) * fv.view(-1) * gw.view(-1) +
                scalars[0, u1.view(-1), v2.view(-1), w2.view(-1)] * gu.view(-1) * fv.view(-1) * fw.view(-1) +
                scalars[0, u2.view(-1), v1.view(-1), w1.view(-1)] * fu.view(-1) * gv.view(-1) * gw.view(-1) +
                scalars[0, u2.view(-1), v1.view(-1), w2.view(-1)] * fu.view(-1) * gv.view(-1) * fw.view(-1) +
                scalars[0, u2.view(-1), v2.view(-1), w1.view(-1)] * fu.view(-1) * fv.view(-1) * gw.view(-1) +
                scalars[0, u2.view(-1), v2.view(-1), w2.view(-1)] * fu.view(-1) * fv.view(-1) * fw.view(-1))
        scalars_on_points = scalars_on_points.view(bts, 1, gs, gs, gs)

    else:
        assert False, 'Impossible dimension'

    return scalars_on_points


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
