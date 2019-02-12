### Base ###
import fnmatch
import math
import os

from torch.nn.modules.module import Module

### Visualization ###
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import nibabel as nib


### Keops ###
# from pykeops.torch import Genred


def cprint(str):
    print(str)
    return str + '\n'


def read_vtk_file(filename, dimension=None, extract_connectivity=False):
    """
    Routine to read  vtk files
    Probably needs new case management
    """

    with open(filename, 'r') as f:
        content = f.readlines()
    fifth_line = content[4].strip().split(' ')

    assert fifth_line[0] == 'POINTS'
    assert fifth_line[2] == 'float'

    nb_points = int(fifth_line[1])
    points = []
    line_start_connectivity = None
    connectivity_type = nb_faces = nb_vertices_in_faces = None

    if dimension is None:
        dimension = DeformableObjectReader.__detect_dimension(content)

    assert isinstance(dimension, int)
    # logger.debug('Using dimension ' + str(dimension) + ' for file ' + filename)

    # Reading the points:
    for i in range(5, len(content)):
        line = content[i].strip().split(' ')
        # Saving the position of the start for the connectivity
        if line == ['']:
            continue
        elif line[0] in ['LINES', 'POLYGONS']:
            line_start_connectivity = i
            connectivity_type, nb_faces, nb_vertices_in_faces = line[0], int(line[1]), int(line[2])
            break
        else:
            points_for_line = np.array(line, dtype=float).reshape(int(len(line) / 3), 3)[:, :dimension]
            for p in points_for_line:
                points.append(p)
    points = np.array(points)
    assert len(points) == nb_points, 'Something went wrong during the vtk reading'

    # Reading the connectivity, if needed.
    if extract_connectivity:
        # Error checking
        if connectivity_type is None:
            RuntimeError('Could not determine connectivity type.')
        if nb_faces is None:
            RuntimeError('Could not determine number of faces type.')
        if nb_vertices_in_faces is None:
            RuntimeError('Could not determine number of vertices type.')

        if line_start_connectivity is None:
            raise KeyError('Could not read the connectivity for the given vtk file')

        connectivity = []

        for i in range(line_start_connectivity + 1, line_start_connectivity + 1 + nb_faces):
            line = content[i].strip().split(' ')
            number_vertices_in_line = int(line[0])

            if connectivity_type == 'POLYGONS':
                assert number_vertices_in_line == 3, 'Invalid connectivity: deformetrica only handles triangles for now.'
                connectivity.append([int(elt) for elt in line[1:]])
            elif connectivity_type == 'LINES':
                assert number_vertices_in_line >= 2, 'Should not happen.'
                for j in range(1, number_vertices_in_line):
                    connectivity.append([int(line[j]), int(line[j + 1])])

        connectivity = np.array(connectivity, dtype=int)

        # Some sanity checks:
        if connectivity_type == 'POLYGONS':
            assert len(connectivity) == nb_faces, 'Found an unexpected number of faces.'
            assert len(connectivity) * 4 == nb_vertices_in_faces

        return torch.from_numpy(points).float(), torch.from_numpy(connectivity)

    return torch.from_numpy(points).float()


def write_mesh(filename, points, connectivity=None):
    connec_names = {2: 'LINES', 3: 'POLYGONS'}

    with open(filename + '.vtk', 'w', encoding='utf-8') as f:
        s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {} float\n'.format(len(points))
        f.write(s)
        for p in points:
            str_p = [str(elt) for elt in p]
            if len(p) == 2:
                str_p.append(str(0.))
            s = ' '.join(str_p) + '\n'
            f.write(s)

        if connectivity is not None:
            a, connec_degree = connectivity.shape
            s = connec_names[connec_degree] + ' {} {}\n'.format(a, a * (connec_degree + 1))
            f.write(s)
            for face in connectivity:
                s = str(connec_degree) + ' ' + ' '.join([str(elt) for elt in face]) + '\n'
                f.write(s)


def write_meshes(points, connectivity, prefix, suffix, targets=None):
    for i, p in enumerate(points):
        write_mesh('%ssubject_%d%s' % (prefix, i, suffix), p, connectivity)
        if targets is not None:
            write_mesh('%ssubject_%d%s' % (prefix, i, '__target'), targets[i][0], targets[i][1])


def write_image(fn, intensities):
    tol = 1e-10
    nib.save(nib.Nifti1Image(np.clip(intensities[0], tol, 255.0 - tol).astype('uint8'), np.eye(4)), fn)


def write_images(intensities, prefix, suffix, targets=None):
    for i, intensities_ in enumerate(intensities):
        write_image('%ssubject_%d%s' % (prefix, i, suffix), intensities_)
        if targets is not None:
            write_image('%ssubject_%d%s' % (prefix, i, '__target'), targets[i])
            write_image('%ssubject_%d%s' % (prefix, i, '__tdiff'), np.abs(intensities_ - targets[i]))


def write_grids(grids, prefix, suffix):
    dim = grids.size(1)
    assert dim == 3
    for i, grid in enumerate(grids):
        # write_mesh('%ssubject_%d%s' % (prefix, i, suffix), grid.permute(1, 2, 3, 0).view(-1, 3).cpu().numpy())
        np.savetxt('%ssubject_%d%s.txt' % (prefix, i, suffix), grid.permute(1, 2, 3, 0).view(-1, 3).cpu().numpy())


def compute_centers_and_normals(points, connectivity):
    dimension = points.shape[1]
    a = points[connectivity[:, 0]]
    b = points[connectivity[:, 1]]
    if dimension == 2:
        centers = (a + b) / 2.
        normals = b - a
    elif dimension == 3:
        c = points[connectivity[:, 2]]
        centers = (a + b + c) / 3.
        normals = torch.cross(b - a, c - a) / 2.
    else:
        raise RuntimeError('Not expected')
    return centers, normals


def extract_subject_and_visit_ids(filename):
    out = []
    if filename[50:52] == '__':
        out.append(int(filename[49]))
        if filename[56:58] == '__':
            out.append(int(filename[55]))
        else:
            out.append(int(filename[55:57]))
    else:
        out.append(int(filename[49:51]))
        if filename[57:59] == '__':
            out.append(int(filename[56]))
        else:
            out.append(int(filename[56:58]))
    return out


def squared_distances(x, y):
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)


def convolve(x, y, p, kernel_width):
    sq = squared_distances(x, y)
    return torch.mm(torch.exp(-sq / (kernel_width ** 2)), p)


def splat_current_on_grid(points, connectivity, grid, kernel_width):
    dimension = points.shape[1]
    centers, normals = compute_centers_and_normals(points, connectivity)
    return convolve(grid.view(-1, dimension), centers, normals, kernel_width).view(grid.size())


def create_cross_sectional_brains_dataset__cn(path_to_data, number_of_datum_train, number_of_datum_test,
                                              random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # CN.
    number_of_datum = number_of_datum_train + number_of_datum_test
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

    return intensities_cn[:number_of_datum_train].unsqueeze(1), intensities_cn[number_of_datum_train:].unsqueeze(1)


def create_cross_sectional_brains_dataset__final(path_to_data, number_of_datum_train, number_of_datum_test,
                                                 random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    # CN.
    number_of_datum = number_of_datum_train + number_of_datum_test
    files = fnmatch.filter(os.listdir(path_to_data), 's*.nii')
    files = np.array(sorted(files))
    assert number_of_datum <= files.shape[0], \
        'Too many required CN brains. A maximum of %d are available' % files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

    intensities_cn = []
    for k, fl in enumerate(files__rdm):
        path_to_datum = os.path.join(path_to_data, fl)
        intensities_cn.append(torch.from_numpy(nib.load(path_to_datum).get_data()).float())
    intensities_cn = torch.stack(intensities_cn).unsqueeze(1)

    return intensities_cn[:number_of_datum_train], intensities_cn[number_of_datum_train:]


def create_cross_sectional_brains_dataset(path_to_data, number_of_datum_train, number_of_datum_test, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
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

    if number_of_datum_train == number_of_datum_test == 1:
        return intensities_cn[:1].unsqueeze(1), intensities_cn[1:].unsqueeze(1)
    else:
        return (torch.cat((intensities_cn[:number_of_datum_train_[0]],
                           intensities_ad[:number_of_datum_train_[1]],
                           intensities_mci[:number_of_datum_train_[2]]), dim=0).unsqueeze(1),
                torch.cat((intensities_cn[number_of_datum_train_[0]:],
                           intensities_ad[number_of_datum_train_[1]:],
                           intensities_mci[number_of_datum_train_[2]:]), dim=0).unsqueeze(1))


def smooth_intensities(intensities, sigma):
    kernel_size = int(5. * sigma + .5)
    # kernel_size = intensities.size(-1)
    grid = torch.stack(torch.meshgrid([torch.arange(kernel_size),
                                       torch.arange(kernel_size),
                                       torch.arange(kernel_size)]), dim=-1).float().type(str(intensities.type()))
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    weights = torch.exp(- torch.sum((grid - mean) ** 2., dim=-1) / (2 * variance))
    weights /= torch.sum(weights)
    filter = nn.Conv3d(1, 1, kernel_size, groups=1, bias=False, padding=int(mean))
    filter.weight.data = weights.view(1, 1, kernel_size, kernel_size, kernel_size)
    filter.weight.data.requires_grad_(False)
    return filter(intensities.unsqueeze(0))[0]


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


def batched_vector_interpolation(vector, points, downsampling_factor):
    bts = points.size(0)
    dim = points.size(1)
    dgs = points.size(2)
    nbp = dgs ** dim
    assert dim == 3

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

    return vector_on_grid


def batched_scalar_interpolation(scalars, points):
    bts = points.size(0)
    dim = points.size(1)
    assert dim == 3

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

    return scalars_on_points


def convolutive_interpolation(velocity, points, deformation_grid, kernel_width):
    dimension = points.size(1)
    velocity_on_points = convolve(points, deformation_grid.contiguous().view(-1, dimension),
                                  velocity.view(-1, dimension), kernel_width)
    return velocity_on_points


def compute_L2_attachment(points_1, points_2):
    return torch.sum((points_1.view(-1) - points_2.view(-1)) ** 2)


def compute_loss(deformed_sources, targets):
    loss = 0.0
    for (deformed_source, target) in zip(deformed_sources, targets):
        loss += compute_L2_attachment(deformed_source, target)
    return loss


# class Tanh_(Module):
#     def forward(self, input):
#         return torch.log(torch.abs(input)) * torch.tanh(input)


class Conv2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
            # Tanh_()
        )

    def forward(self, x):
        return self.net(x)


class Conv3d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
            # Tanh_()
        )

    def forward(self, x):
        return self.net(x)


class ConvTranspose2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
            # Tanh_()
        )

    def forward(self, x):
        return self.net(x)


class ConvTranspose3d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
            # Tanh_()
        )

    def forward(self, x):
        return self.net(x)


class Linear_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=bias),
            nn.Tanh()
            # Tanh_()
        )

    def forward(self, x):
        return self.net(x.view(-1, self.in_ch)).view(-1, self.out_ch)


# class Linear_Tanh_(nn.Module):
#     def __init__(self, in_ch, out_ch, bias=True):
#         nn.Module.__init__(self)
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.net = nn.Sequential(
#             nn.Linear(in_ch, out_ch, bias=bias),
#             # nn.Tanh()
#             Tanh_()
#         )
#
#     def forward(self, x):
#         return self.net(x.view(-1, self.in_ch)).view(-1, self.out_ch)


def plot_registrations(sources, targets,
                       sources_, targets_,
                       deformed_sources, deformed_grids,
                       deformation_grid, deformation_fields,
                       prefix, suffix):
    for k, (source, source_, target, target_, deformed_source, deformed_grid, deformation_field) in enumerate(
            zip(sources, sources_, targets, targets_, deformed_sources, deformed_grids, deformation_fields)):
        figsize = 7
        f, axes = plt.subplots(1, 2, figsize=(2 * figsize, figsize))

        ### FIRST FIGURE ###
        ax = axes[0]

        p = source.detach().cpu().numpy()
        c = source_.detach().cpu().numpy()
        ax.plot([p[c[:, 0]][:, 0], p[c[:, 1]][:, 0]],
                [p[c[:, 0]][:, 1], p[c[:, 1]][:, 1]], 'tab:blue', linewidth=2)

        g = deformed_grid.detach().cpu().numpy()
        ax.plot([g[:-1, :, 0].ravel(), g[1:, :, 0].ravel()],
                [g[:-1, :, 1].ravel(), g[1:, :, 1].ravel()], 'k', linewidth=0.5)
        ax.plot([g[:, :-1, 0].ravel(), g[:, 1:, 0].ravel()],
                [g[:, :-1, 1].ravel(), g[:, 1:, 1].ravel()], 'k', linewidth=0.5)

        g = deformation_grid.view(-1, dimension).detach().cpu().numpy()
        m = deformation_field.view(-1, dimension).detach().cpu().numpy()
        if np.sum(m ** 2) > 0:
            ax.quiver(g[:, 0], g[:, 1], m[:, 0], m[:, 1])

        ax.set_xlim((-2.6, 2.6))
        ax.set_ylim((-2.6, 2.6))

        ### SECOND FIGURE ###
        ax = axes[1]

        p = target.detach().cpu().numpy()
        c = target_.detach().cpu().numpy()
        ax.plot([p[c[:, 0]][:, 0], p[c[:, 1]][:, 0]],
                [p[c[:, 0]][:, 1], p[c[:, 1]][:, 1]], 'tab:red', linewidth=2)

        p = deformed_source.detach().cpu().numpy()
        c = source_.detach().cpu().numpy()
        ax.plot([p[c[:, 0]][:, 0], p[c[:, 1]][:, 0]],
                [p[c[:, 0]][:, 1], p[c[:, 1]][:, 1]], 'tab:blue', linewidth=2)

        ax.set_xlim((-2.6, 2.6))
        ax.set_ylim((-2.6, 2.6))
        ax.grid()

        #        plt.show()
        f.savefig('%s__subject_%d__%s.pdf' % (prefix, k, suffix), bbox_inches='tight')
        plt.close(f)


class Encoder2d(nn.Module):
    """
    in: in_grid_size * in_grid_size * 2
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.down1 = Conv2d_Tanh(2, 4)
        self.down2 = Conv2d_Tanh(4, 8)
        self.down3 = Conv2d_Tanh(8, 16)
        self.down4 = Conv2d_Tanh(16, 16)
        self.linear1 = nn.Linear(16 * n * n, latent_dimension)
        self.linear2 = nn.Linear(16 * n * n, latent_dimension)
        print('>> Encoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        m = self.linear1(x.view(x.size(0), -1)).view(x.size(0), -1)
        s = self.linear2(x.view(x.size(0), -1)).view(x.size(0), -1)
        return m, s


class Encoder3d(nn.Module):
    """
    in: in_grid_size * in_grid_size * in_grid_size * 3
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.down1 = Conv3d_Tanh(1, 4)
        self.down2 = Conv3d_Tanh(4, 8)
        self.down3 = Conv3d_Tanh(8, 16)
        self.down4 = Conv3d_Tanh(16, 32)
        # self.down5 = Conv3d_Tanh(32, 32)
        self.linear1 = nn.Linear(32 * n ** 3, latent_dimension)
        self.linear2 = nn.Linear(32 * n ** 3, latent_dimension)
        print('>> Encoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        # x = self.down5(x)
        m = self.linear1(x.view(x.size(0), -1)).view(x.size(0), -1)
        s = self.linear2(x.view(x.size(0), -1)).view(x.size(0), -1)
        return m, s


class Decoder2d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * 2
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.linear = Linear_Tanh(latent_dimension, 16 * self.inner_grid_size * self.inner_grid_size, bias=False)
        self.up1 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up2 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up3 = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> Decoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x).view(batch_size, 16, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class Decoder3d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.linear = Linear_Tanh(latent_dimension, 32 * self.inner_grid_size ** 3, bias=False)
        # self.up1 = ConvTranspose3d_Tanh(32, 32, bias=False)
        self.up1 = ConvTranspose3d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose3d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose3d_Tanh(8, 4, bias=False)
        self.up4 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> Decoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x).view(batch_size, 32, self.inner_grid_size, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x = self.up5(x)
        return x


class DeepDecoder3d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear2 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear3 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        # self.up1 = ConvTranspose3d_Tanh(32, 32, bias=False)
        self.up1 = ConvTranspose3d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose3d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose3d_Tanh(8, 4, bias=False)
        self.up4 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> DeepDecoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x).view(batch_size, 32, self.inner_grid_size, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x = self.up5(x)
        return x


class DeepDeepDecoder3d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear2 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear3 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear4 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        self.linear5 = Linear_Tanh(32 * self.inner_grid_size ** 3, 32 * self.inner_grid_size ** 3, bias=False)
        # self.up1 = ConvTranspose3d_Tanh(32, 32, bias=False)
        self.up1 = ConvTranspose3d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose3d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose3d_Tanh(8, 4, bias=False)
        self.up4 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> DeepDeepDecoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x).view(batch_size, 32, self.inner_grid_size, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x = self.up5(x)
        return x


class BayesianAtlas(nn.Module):

    def __init__(self, template_intensities, latent_dimension, alpha, number_of_time_points, downsampling_factor):
        nn.Module.__init__(self)

        self.latent_dimension = latent_dimension
        print('>> latent dimension = %d' % latent_dimension)
        self.alpha = alpha
        self.dimension = len(template_intensities.size()) - 1
        self.grid_size = template_intensities.size(1)
        self.downsampling_factor = downsampling_factor
        self.downsampled_grid_size = self.grid_size // self.downsampling_factor
        print('>> grid size = %d ; downsampling factor = %d ; downsampled grid size = %d' %
              (self.grid_size, self.downsampling_factor, self.downsampled_grid_size))
        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

        s = 3.
        dgs = self.downsampled_grid_size
        typ = str(template_intensities.type())
        if self.dimension == 2:
            self.L = torch.stack(torch.meshgrid([torch.arange(dgs), torch.arange(dgs)])).type(typ)
            self.L = (- 2 * alpha * (torch.sum(torch.cos(
                (2.0 * math.pi / float(dgs)) * self.L), dim=0) - self.dimension) + 1) ** s
            self.L = self.L.view(1, 1, 1, dgs, dgs, 1)
        else:
            self.L = torch.stack(torch.meshgrid([torch.arange(dgs), torch.arange(dgs),
                                            torch.arange(dgs)])).type(typ)
            self.L = (- 2 * alpha * (torch.sum(torch.cos(
                (2.0 * math.pi / float(dgs)) * self.L), dim=0) - self.dimension) + 1) ** s
            self.L = self.L.view(1, 1, 1, dgs, dgs, dgs, 1)

        self.template_intensities = template_intensities
        # self.template_intensities = nn.Parameter(template_intensities)
        print('>> Template intensities are %d^%d = %d parameters' %
              (self.grid_size, self.dimension, self.grid_size ** self.dimension))

        if self.dimension == 2:
            self.encoder = Encoder2d(self.grid_size, latent_dimension)
            self.decoder = DeepDecoder2d(latent_dimension, self.downsampled_grid_size)
        elif self.dimension == 3:
            self.encoder = Encoder3d(self.grid_size, latent_dimension)
            self.decoder = DeepDecoder3d(latent_dimension, self.downsampled_grid_size)
            # self.encoder = LargeEncoder3d(self.grid_size, latent_dimension)
            # self.decoder = LargeDecoder3d(latent_dimension, self.downsampled_grid_size)
        else:
            raise RuntimeError
        print('>> BayesianAtlas has %d parameters' % (sum([len(elt.view(-1)) for elt in self.parameters()]) +
                                                      self.grid_size ** self.dimension))

    def encode(self, observations):
        return self.encoder(observations)

    def forward(self, z):

        # INIT
        bts = z.size(0)
        gs = self.grid_size
        dsf = self.downsampling_factor
        dgs = self.downsampled_grid_size
        ntp = self.number_of_time_points

        # DECODE
        v = self.decoder(z)

        # # GAUSSIAN SMOOTHING
        # v = blabla

        # FLOW: SCALING AND SQUARING
        assert self.dimension == 3
        grid = torch.stack(torch.meshgrid(
            [torch.linspace(0.0, gs - 1.0, dgs),
             torch.linspace(0.0, gs - 1.0, dgs),
             torch.linspace(0.0, gs - 1.0, dgs)])).type(str(z.type())).view(1, 3, dgs, dgs, dgs).repeat(bts, 1, 1, 1, 1)

        v /= float(2 ** ntp)
        for t in range(ntp):
            # print(t)
            v += batched_vector_interpolation(v, grid + v, dsf)

        # INTERPOLATE
        intensities = batched_scalar_interpolation(self.template_intensities, grid + v)

        # RETURN
        return intensities

    def tamper_template_gradient(self, lr, print_info=False):
        tampered_template_gradient = (lr * smooth_intensities(self.template_intensities.grad, 0.5)).detach()
        self.template_intensities.grad = tampered_template_gradient
        if print_info:
            print('torch.min(tampered_template_gradient) =', torch.min(tampered_template_gradient),
                  'torch.max(tampered_template_gradient) =', torch.max(tampered_template_gradient))

    def clamp_template_intensities(self):
        torch.clamp(self.template_intensities.detach(), 0., 255., out=self.template_intensities.detach())
        self.template_intensities.requires_grad_(True)
        # print(torch.sum(self.template_intensities ** 2))

    def write_trajectories(self, observations, prefix):

        bts = observations.size(0)
        gs = self.grid_size
        dsf = self.downsampling_factor
        dgs = self.downsampled_grid_size
        ntp = self.number_of_time_points

        # ENCODE
        z, _ = model.encode(observations)

        # DECODE
        v = self.decoder(z)

        # FLOW: SCALING AND SQUARING
        assert self.dimension == 3
        grid = torch.stack(torch.meshgrid([torch.linspace(0.0, gs - 1.0, dgs),
                                        torch.linspace(0.0, gs - 1.0, dgs),
                                        torch.linspace(0.0, gs - 1.0, dgs)])).type(
            str(z.type())).view(1, 3, dgs, dgs, dgs).repeat(bts, 1, 1, 1, 1)

        # write_grids(x_.detach(), prefix + '__', '__grid__t_%d' % 0)
        write_images(batched_scalar_interpolation(self.template_intensities, grid).detach().cpu().numpy(),
                     prefix + '__', '__t_%d' % 0, targets=observations.detach().cpu().numpy())

        v /= float(2 ** ntp)
        for t in range(ntp):
            v += batched_vector_interpolation(v, grid + v, dsf)

            # write_grids(x_.detach(), prefix + '__', '__grid__t_%d' % (t + 1))
            write_images(batched_scalar_interpolation(self.template_intensities, grid + v).detach().cpu().numpy(),
                         prefix + '__', '__t_%d' % (t + 1), targets=observations.detach().cpu().numpy())


if __name__ == '__main__':

    ############################
    ##### GLOBAL VARIABLES #####
    ############################

    initialize_template = None
    initialize_encoder = None
    initialize_decoder = None

    initial_state = None
    initial_encoder_state = None
    initial_decoder_state = None

    # MODEL

    dataset = 'brains'

    number_of_datum_train = 5
    number_of_datum_test = 0

    number_of_time_points = 5

    data_size = 64
    downsampling_factor = 2

    # OPTIMIZATION

    number_of_epochs = 1000
    number_of_epochs_for_init = 1000
    number_of_epochs_for_warm_up = 0

    print_every_n_iters = 1
    save_every_n_iters = 50

    learning_rate = 5e-3
    learning_rate_ratio = 0.5
    learning_rate_decay = 0.95

    batch_size = 5

    device = 'cuda:01'
    # device = 'cpu'

    ############################
    ######## INITIALIZE ########
    ############################

    if dataset == 'brains':
        experiment_prefix = '1_first_attempt'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/data'))

        initialize_template = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/data/colin27.nii'))
        # initialize_encoder = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))
        # initialize_decoder = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/brains/output__33_bayesian_atlas_fourier_image/epoch_25000__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_encoder__epoch_9000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        dimension = 3
        latent_dimension = 3

        alpha = 0.5
        lambda_ = 1.
        kappa_ = 1.

        noise_variance = 1. ** 2

        intensities, intensities_test = create_cross_sectional_brains_dataset__final(
            path_to_meshes,
            number_of_datum_train, number_of_datum_test, random_seed=42)

    else:
        raise RuntimeError

    assert number_of_time_points > 1

    log = ''
    output_dir = os.path.join(path_to_meshes, '../output__' + experiment_prefix)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if (not torch.cuda.is_available()) and 'cuda' in device:
        device = 'cpu'
        print('>> CUDA is not available. Overridding with device = "cpu".')

    if device == 'cuda:01':
        device = 'cuda'
        cmd = '>> export CUDA_VISIBLE_DEVICES=1'
        print(cmd)
        os.system(cmd)
        # torch.cuda.device(1)

    ############################
    ###### TRAIN AND TEST ######
    ############################

    noise_dimension = 64 ** 3

    if initialize_template is None:
        initial_template__black = torch.zeros(intensities[0].size())
        initial_template__test = intensities_test[0]
        initial_template__mean = torch.mean(intensities, dim=0)
        initial_template_intensities = initial_template__mean
    else:
        initial_template_intensities = torch.from_numpy(nib.load(initialize_template).get_data()).float().unsqueeze(0)

    if 'cuda' in device:
        initial_template_intensities = initial_template_intensities.cuda()

    model = BayesianAtlas(initial_template_intensities, latent_dimension,
                          alpha, number_of_time_points, downsampling_factor)

    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=max(10, int(number_of_epochs / 100.)), gamma=learning_rate_decay)

    if 'cuda' in device:
        model.cuda()

    ########################
    ###### INITIALIZE ######
    ########################

    if initialize_encoder is not None:
        print('>> INITIALIZING THE ENCODER from file: %s' % initialize_encoder)
        latent_momenta__init = torch.from_numpy(np.loadtxt(initialize_encoder)).float()
        if 'cuda' in device:
            latent_momenta__init = latent_momenta__init.cuda()
        optimizer = Adam(model.encoder.parameters(), lr=learning_rate)

        for epoch in range(number_of_epochs_for_init + 1):

            np_attachment_loss = 0.0
            np_kullback_regularity_loss = 0.0
            np_total_loss = 0.0

            indexes = np.random.permutation(number_of_datum_train)
            for k in range(number_of_datum_train // batch_size):  # drops the last batch
                if 'cuda' in device:
                    batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]].cuda()
                else:
                    batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]]
                batch_latent_momenta__init = latent_momenta__init[indexes[k * batch_size:(k + 1) * batch_size]]

                # ENCODE AND SAMPLE
                means, log_variances = model.encode(batch_target_intensities)
                batch_latent_momenta = means + torch.zeros_like(means).normal_() * torch.exp(0.5 * log_variances)

                # LOSS
                attachment_loss = torch.sum(
                    (batch_latent_momenta - batch_latent_momenta__init) ** 2) / 0.01
                np_attachment_loss += attachment_loss.detach().cpu().numpy()

                kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
                np_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

                total_loss = attachment_loss + kullback_regularity_loss
                np_total_loss += total_loss.detach().cpu().numpy()

                # GRADIENT STEP
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            np_attachment_loss /= float(batch_size * (number_of_datum_train // batch_size))
            np_kullback_regularity_loss /= float(batch_size * (number_of_datum_train // batch_size))
            np_total_loss /= float(batch_size * (number_of_datum_train // batch_size))

            if epoch % print_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                log += cprint(
                    '\n[Epoch: %d] Learning rate = %.2E'
                    '\nTrain loss = %.3f (attachment = %.3f ; kullback regularity = %.3f)' %
                    (epoch, list(optimizer.param_groups)[0]['lr'],
                     np_total_loss, np_attachment_loss, np_kullback_regularity_loss))

            if epoch % save_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                with open(os.path.join(output_dir, 'init_encoder__log.txt'), 'w') as f:
                    f.write(log)

                torch.save(model.encoder.state_dict(),
                           os.path.join(output_dir, 'init_encoder__epoch_%d__model.pth' % epoch))

        initial_encoder_state = os.path.join(output_dir,
                                             'init_encoder__epoch_%d__model.pth' % number_of_epochs_for_init)

    if initialize_decoder is not None:
        print('>> INITIALIZING THE DECODER from file: %s' % initialize_decoder)
        latent_momenta__init = torch.from_numpy(np.loadtxt(initialize_decoder)).float()
        if 'cuda' in device:
            latent_momenta__init = latent_momenta__init.cuda()
        optimizer = Adam(model.decoder.parameters(), lr=learning_rate)

        for epoch in range(number_of_epochs_for_init + 1):

            np_attachment_loss = 0.0
            np_total_loss = 0.0

            indexes = np.random.permutation(number_of_datum_train)
            for k in range(number_of_datum_train // batch_size):  # drops the last batch
                if 'cuda' in device:
                    batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]].cuda()
                else:
                    batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]]
                batch_latent_momenta = latent_momenta__init[indexes[k * batch_size:(k + 1) * batch_size]]

                # DECODE
                deformed_template = model(batch_latent_momenta)

                # LOSS
                attachment_loss = torch.sum((deformed_template - batch_target_intensities) ** 2) / noise_variance
                np_attachment_loss += attachment_loss.detach().cpu().numpy()

                np_total_loss += total_loss.detach().cpu().numpy()

                # GRADIENT STEP
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            np_attachment_loss /= float(batch_size * (number_of_datum_train // batch_size))
            np_total_loss /= float(batch_size * (number_of_datum_train // batch_size))

            if epoch % print_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                log += cprint(
                    '\n[Epoch: %d] Learning rate = %.2E'
                    '\nTrain loss = %.3f (attachment = %.3f)' %
                    (epoch, list(optimizer.param_groups)[0]['lr'], np_total_loss, np_attachment_loss))

            if epoch % save_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                with open(os.path.join(output_dir, 'init_decoder__log.txt'), 'w') as f:
                    f.write(log)

                torch.save(model.decoder.state_dict(),
                           os.path.join(output_dir, 'init_decoder__epoch_%d__model.pth' % epoch))

            initial_decoder_state = os.path.join(output_dir,
                                                 'init_decoder__epoch_%d__model.pth' % number_of_epochs_for_init)

    # LOAD INITIALIZATIONS ----------------------------------------
    if initial_encoder_state is not None:
        print('>> initial_encoder_state = %s' % os.path.basename(initial_encoder_state))
        encoder_state_dict = torch.load(initial_encoder_state, map_location=lambda storage, loc: storage)
        # if 'cuda' in device:
        #     encoder_state_dict = encoder_state_dict.cuda()
        model.encoder.load_state_dict(encoder_state_dict)

    if initial_decoder_state is not None:
        print('>> initial_decoder_state = %s' % os.path.basename(initial_decoder_state))
        decoder_state_dict = torch.load(initial_decoder_state, map_location=lambda storage, loc: storage)
        # if 'cuda' in device:
        #     decoder_state_dict = decoder_state_dict.cuda()
        model.decoder.load_state_dict(decoder_state_dict)

    if initial_state is not None:
        print('>> initial_state = %s' % os.path.basename(initial_state))
        state_dict = torch.load(initial_state, map_location=lambda storage, loc: storage)
        # if 'cuda' in device:
        #     state_dict = state_dict.cuda()
        model.load_state_dict(state_dict)
    # -------------------------------------------------------------

    optimizer = Adam(model.parameters(), lr=learning_rate)
    for epoch in range(number_of_epochs + 1):
        # scheduler.step()

        #############
        ### TRAIN ###
        #############

        train_attachment_loss = 0.
        train_kullback_regularity_loss = 0.
        train_total_loss = 0.
        # z_mean = 0.
        # z_std = 0.

        indexes = np.random.permutation(number_of_datum_train)
        for k in range(number_of_datum_train // batch_size):  # drops the last batch
            if 'cuda' in device:
                batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]].cuda()
            else:
                batch_target_intensities = intensities[indexes[k * batch_size:(k + 1) * batch_size]]

            # ENCODE, SAMPLE AND DECODE
            means, log_variances = model.encode(batch_target_intensities)

            if epoch < number_of_epochs_for_warm_up + 1:
                batch_latent_momenta = means
                deformed_template = model(batch_latent_momenta)
            else:
                batch_latent_momenta = means + torch.zeros_like(means).normal_() * torch.exp(0.5 * log_variances)
                deformed_template = model(batch_latent_momenta)

            # z_mean += torch.sum(batch_latent_momenta.detach(), dim=0).detach().cpu().numpy()
            # z_std += torch.sum(batch_latent_momenta.detach() ** 2, dim=0).detach().cpu().numpy()

            # LOSS
            attachment_loss = torch.sum((deformed_template - batch_target_intensities) ** 2) / noise_variance
            train_attachment_loss += attachment_loss.detach().cpu().numpy()

            kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
            kullback_regularity_loss *= kappa_
            train_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

            total_loss = attachment_loss + kullback_regularity_loss
            train_total_loss += total_loss.detach().cpu().numpy()

            # GRADIENT STEP
            optimizer.zero_grad()
            total_loss.backward()
            # model.tamper_template_gradient(learning_rate_ratio, epoch < 100)
            optimizer.step()
            # model.clamp_template_intensities()
            # model.update_template(learning_rate_ratio * list(optimizer.param_groups)[0]['lr'])

            # noise_variance *= float(attachment_loss.detach().cpu().numpy() / float(noise_dimension * batch_size))

        train_attachment_loss /= float(batch_size * (number_of_datum_train // batch_size))
        train_kullback_regularity_loss /= float(batch_size * (number_of_datum_train // batch_size))
        train_total_loss /= float(batch_size * (number_of_datum_train // batch_size))

        ############
        ### TEST ###
        ############

        test_attachment_loss = 0.
        test_kullback_regularity_loss = 0.
        test_total_loss = 0.

        # batch_target_intensities = intensities_test.cuda()
        #
        # # ENCODE, SAMPLE AND DECODE
        # means, log_variances = model.encode(batch_target_intensities)
        # batch_latent_momenta = means
        # deformed_template = model(batch_latent_momenta, with_regularity_loss=True)
        #
        # # LOSS
        # attachment_loss = torch.sum((deformed_template - batch_target_intensities) ** 2) / noise_variance
        # test_attachment_loss += attachment_loss.detach().cpu().numpy()
        #
        # kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
        # test_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()
        #
        # total_loss = attachment_loss + kullback_regularity_loss
        # test_total_loss += total_loss.detach().cpu().numpy()
        #
        # test_attachment_loss /= float(number_of_datum_test)
        # test_kullback_regularity_loss /= float(number_of_datum_test)
        # test_total_loss /= float(number_of_datum_test)

        ################
        ### TEMPLATE ###
        ################

        template_intensities = model.template_intensities.view((1,) + model.template_intensities.size())
        template_latent_momenta, _ = model.encode(template_intensities)
        template_latent_momenta_norm = float(torch.norm(template_latent_momenta[0], p=2).detach().cpu().numpy())

        #############
        ### WRITE ###
        #############

        if epoch % print_every_n_iters == 0 or epoch == number_of_epochs:
            log += cprint(
                '\n[Epoch: %d] Learning rate = %.2E ; Noise std = %.2E ; Template latent q norm = %.3f'
                '\nTrain loss = %.3E (attachment = %.3E ; kullback regularity = %.3E)'
                '\nTest  loss = %.3E (attachment = %.3E ; kullback regularity = %.3E)' %
                (epoch, list(optimizer.param_groups)[0]['lr'], math.sqrt(noise_variance), template_latent_momenta_norm,
                 train_total_loss, train_attachment_loss, train_kullback_regularity_loss,
                 test_total_loss, test_attachment_loss, test_kullback_regularity_loss))

        if epoch % save_every_n_iters == 0 or epoch == number_of_epochs:
            with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
                f.write(log)

            torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_%d__model.pth' % epoch))

            n = 3
            if 'cuda' in device:
                model.write_trajectories(intensities[:n].cuda(), os.path.join(output_dir, 'epoch_%d__train' % epoch))
                # model.write_trajectories(intensities_test[:n].cuda(), os.path.join(output_dir, 'epoch_%d__test' % epoch))
                # model.write_trajectories(template_intensities, os.path.join(output_dir, 'epoch_%d__template' % epoch))
            else:
                model.write_trajectories(intensities[:n], os.path.join(output_dir, 'epoch_%d__train' % epoch))
                # model.write_trajectories(intensities_test[:n], os.path.join(output_dir, 'epoch_%d__test' % epoch))
                # model.write_trajectories(template_intensities, os.path.join(output_dir, 'epoch_%d__template' % epoch))
