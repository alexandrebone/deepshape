### Base ###
import fnmatch
import math
import os

### Visualization ###
import matplotlib.pyplot as plt
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


def write_mesh(filename, points, connectivity):
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


def write_deformations(vfields, grid, prefix, suffix):
    for i, vfield in enumerate(vfields):
        write_deformation('%ssubject_%d%s.vtk' % (prefix, i, suffix), vfield, grid)


def write_deformation(fn, vfield, grid):
    dimension = vfield.size(0)
    x = grid.reshape(-1, dimension)
    if dimension == 2:
        v = vfield.permute(1, 2, 0).view(-1, dimension).detach().cpu().numpy()
    elif dimension == 3:
        v = vfield.permute(1, 2, 3, 0).view(-1, dimension).detach().cpu().numpy()
    else:
        raise RuntimeError

    poly_data = vtkPolyData()
    points = vtkPoints()

    if dimension == 3:
        for i, x_ in enumerate(x):
            points.InsertPoint(i, x_)
    else:
        for i, x_ in enumerate(x):
            points.InsertPoint(i, np.concatenate([x_, [0.]]))

    poly_data.SetPoints(points)

    vectors = vtkDoubleArray()
    vectors.SetNumberOfComponents(3)
    if dimension == 3:
        for v_ in v:
            vectors.InsertNextTuple(v_)
    else:
        for v_ in v:
            vectors.InsertNextTuple(np.concatenate([v_, [0.]]))

    poly_data.GetPointData().SetVectors(vectors)

    writer = vtkPolyDataWriter()
    writer.SetInputData(poly_data)
    writer.SetFileName(fn)
    writer.Update()


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


def create_cross_sectional_starmen_dataset(path_to_starmen, number_of_starmen_train, number_of_starmen_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_starmen = number_of_starmen_train + number_of_starmen_test

    starmen_files = fnmatch.filter(os.listdir(path_to_starmen), 'SimulatedData__Reconstruction__starman__subject_*')
    starmen_files = np.array(sorted(starmen_files, key=extract_subject_and_visit_ids))

    assert number_of_starmen <= starmen_files.shape[0], 'Too many required starmen. A maximum of %d are available' % \
                                                        starmen_files.shape[0]
    starmen_files__rdm = starmen_files[np.random.choice(starmen_files.shape[0], size=number_of_starmen, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, starman_file in enumerate(starmen_files__rdm):
        path_to_starman = os.path.join(path_to_starmen, starman_file)

        p, c = read_vtk_file(path_to_starman, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_starmen_train],
            connectivities[:number_of_starmen_train],
            splats[:number_of_starmen_train],
            points[number_of_starmen_train:],
            connectivities[number_of_starmen_train:],
            splats[number_of_starmen_train:])


def create_cross_sectional_ellipsoids_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                              splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'ellipsoid*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required ellipsoids. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, starman_file in enumerate(files__rdm):
        path_to_starman = os.path.join(path_to_meshes, starman_file)

        p, c = read_vtk_file(path_to_starman, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_hippocampi_dataset__for_registration(
        path_to_meshes, number_of_meshes_train, number_of_meshes_test,
        splatting_grid, dimension, gkernel, gamma, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files_all = fnmatch.filter(os.listdir(path_to_meshes), 'sub-ADNI*')
    files_all = np.array(sorted(files_all))
    files_train = files_all[np.random.choice(files_all.shape[0], size=number_of_meshes, replace=None)]
    files_test = []
    for fl in files_all:
        if fl not in files_train:
            files_test.append(fl)

    points = []
    connectivities = []
    centers = []
    normals = []
    norms = []
    splats = []
    for k, fl in enumerate(files_test):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        ctr, nrl = compute_centers_and_normals(p, c)
        # s = convolve(splatting_grid.view(-1, dimension), ctr, nrl, kernel_width).view(splatting_grid.size())
        s = gkernel(gamma, splatting_grid.view(-1, dimension), ctr, nrl).view(splatting_grid.size())
        points.append(p)
        connectivities.append(c)
        centers.append(ctr)
        normals.append(nrl)
        norms.append(torch.sum(nrl * gkernel(gamma, ctr, ctr, nrl)))
        splats.append(s)

    splats = torch.stack(splats)

    return points, connectivities, centers, normals, norms, splats


def create_cross_sectional_circles_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'circle*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required circles. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_leaves_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                          splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'leaf*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required leaves. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_squares_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'square*')
    files = np.array(sorted(files, key=(lambda x: [int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])])))

    assert number_of_meshes <= files.shape[0], 'Too many required squares. A maximum of %d are available' % \
                                               files.shape[0]

    if number_of_meshes < len(files):
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]
    else:
        files__rdm = files

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


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


def compute_L2_attachment(points_1, points_2):
    return torch.sum((points_1.view(-1) - points_2.view(-1)) ** 2)


def compute_loss(deformed_sources, targets):
    loss = 0.0
    for (deformed_source, target) in zip(deformed_sources, targets):
        loss += compute_L2_attachment(deformed_source, target)
    return loss


class Conv2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Conv3d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ConvTranspose2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ConvTranspose3d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
            nn.Tanh()
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
        )

    def forward(self, x):
        return self.net(x.view(-1, self.in_ch)).view(-1, self.out_ch)


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
        self.down4 = Conv2d_Tanh(16, 32)
        self.linear1 = nn.Linear(32 * n * n, latent_dimension)
        self.linear2 = nn.Linear(32 * n * n, latent_dimension)
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
        self.down1 = Conv3d_Tanh(3, 4)
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


class Encoder3d_(nn.Module):
    """
    in: in_grid_size * in_grid_size * in_grid_size * 3
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.down1 = Conv3d_Tanh(3, 4)
        self.down2 = Conv3d_Tanh(4, 8)
        self.down3 = Conv3d_Tanh(8, 16)
        # self.down4 = Conv3d_Tanh(16, 16)
        self.linear1 = nn.Linear(16 * n * n * n, latent_dimension)
        self.linear2 = nn.Linear(16 * n * n * n, latent_dimension)
        print('>> Encoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # x = self.down4(x)
        m = self.linear1(x.view(x.size(0), -1)).view(x.size(0), -1)
        s = self.linear2(x.view(x.size(0), -1)).view(x.size(0), -1)
        return m, s


class DeepEncoder3d(nn.Module):
    """
    in: in_grid_size * in_grid_size * in_grid_size * 3
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.down1 = Conv3d_Tanh(3, 4)
        self.down2 = Conv3d_Tanh(4, 8)
        self.down3 = Conv3d_Tanh(8, 16)
        self.linear1__m = nn.Linear(16 * n ** 3, 16 * n ** 3)
        self.linear1__s = nn.Linear(16 * n ** 3, 16 * n ** 3)
        self.linear2__m = nn.Linear(16 * n ** 3, 16 * n ** 3)
        self.linear2__s = nn.Linear(16 * n ** 3, 16 * n ** 3)
        self.linear3__m = nn.Linear(16 * n ** 3, latent_dimension)
        self.linear3__s = nn.Linear(16 * n ** 3, latent_dimension)
        print('>> DeepEncoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        m, s = self.linear1__m(x.view(x.size(0), -1)), self.linear1__s(x.view(x.size(0), -1))
        m, s = self.linear2__m(m), self.linear2__s(s)
        m, s = self.linear3__m(m), self.linear3__s(s)
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


class DeepDecoder2d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 32 * self.inner_grid_size ** 2, bias=False)
        self.linear2 = Linear_Tanh(32 * self.inner_grid_size ** 2, 32 * self.inner_grid_size ** 2, bias=False)
        self.linear3 = Linear_Tanh(32 * self.inner_grid_size ** 2, 32 * self.inner_grid_size ** 2, bias=False)
        self.up1 = ConvTranspose2d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up4 = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> DeepDecoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x).view(batch_size, 32, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


class DeepDeepDecoder2d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 16 * self.inner_grid_size ** 2, bias=False)
        self.linear2 = Linear_Tanh(16 * self.inner_grid_size ** 2, 16 * self.inner_grid_size ** 2, bias=False)
        self.linear3 = Linear_Tanh(16 * self.inner_grid_size ** 2, 16 * self.inner_grid_size ** 2, bias=False)
        self.linear4 = Linear_Tanh(16 * self.inner_grid_size ** 2, 16 * self.inner_grid_size ** 2, bias=False)
        self.linear5 = Linear_Tanh(16 * self.inner_grid_size ** 2, 16 * self.inner_grid_size ** 2, bias=False)
        self.up1 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up2 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up3 = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> DeepDecoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x).view(batch_size, 16, self.inner_grid_size, self.inner_grid_size)
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
        self.inner_grid_size = int(out_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.linear = Linear_Tanh(latent_dimension, 16 * self.inner_grid_size ** 3, bias=False)
        self.up1 = ConvTranspose3d_Tanh(16, 8, bias=False)
        self.up2 = ConvTranspose3d_Tanh(8, 4, bias=False)
        self.up3 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> Decoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x).view(batch_size, 16, self.inner_grid_size, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
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


class DeepDecoder3d_(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -3)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 16 * self.inner_grid_size ** 3, bias=False)
        self.linear2 = Linear_Tanh(16 * self.inner_grid_size ** 3, 16 * self.inner_grid_size ** 3, bias=False)
        self.linear3 = Linear_Tanh(16 * self.inner_grid_size ** 3, 16 * self.inner_grid_size ** 3, bias=False)
        self.up1 = ConvTranspose3d_Tanh(16, 8, bias=False)
        self.up2 = ConvTranspose3d_Tanh(8, 4, bias=False)
        self.up3 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        print('>> DeepDecoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x).view(batch_size, 16, self.inner_grid_size, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class BayesianAtlas(nn.Module):

    def __init__(self,
                 template_points, template_connectivity,
                 bounding_box, latent_dimension, alpha,
                 splatting_grid, deformation_grid, number_of_time_points):
        nn.Module.__init__(self)

        self.deformation_grid = deformation_grid
        self.deformation_grid_size = deformation_grid.size(0)
        self.splatting_grid = splatting_grid
        self.splatting_grid_size = splatting_grid.size(0)

        self.template_connectivity = template_connectivity
        self.latent_dimension = latent_dimension
        self.alpha = alpha
        self.bounding_box = bounding_box
        self.dimension = template_points.size(1)

        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

        s = 3.0
        dgs = self.deformation_grid_size
        typ = str(template_points.type())
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

        # self.template_points = template_points
        self.template_points = nn.Parameter(template_points)
        print('>> Template points are %d x %d = %d parameters' % (
            template_points.size(0), template_points.size(1), template_points.size(0) * template_points.size(1)))
        if self.dimension == 2:
            self.encoder = Encoder2d(self.splatting_grid_size, self.latent_dimension)
            self.decoder = DeepDecoder2d(self.latent_dimension, self.deformation_grid_size)
        elif self.dimension == 3:
            self.encoder = Encoder3d(self.splatting_grid_size, self.latent_dimension)
            self.decoder = DeepDecoder3d(self.latent_dimension, self.deformation_grid_size)
            # self.encoder = DeepEncoder3d(self.splatting_grid_size, latent_dimension)
            # self.decoder = DeepDecoder3d(latent_dimension, deformation_grid_size)
        else:
            raise RuntimeError
        print('>> BayesianAtlas has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def encode(self, observations):
        return self.encoder(observations)

    def forward(self, q, with_regularity_loss=False):

        bts = q.size(0)
        ntp = self.number_of_time_points

        # DECODE
        v_t = []
        # dq0 = self.decoder(q * 0.0)
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))
            # dq1 = self.decoder(q * (t + 1) * self.dt)
            # v_t.append((dq1 - dq0) / self.dt)
            # dq0 = dq1

        # FLOW
        x = self.template_points.clone().view(1, -1, self.dimension).repeat(bts, 1, 1)
        for v in v_t:
            x += self.dt * batched_bilinear_interpolation(v, x, self.bounding_box, self.deformation_grid_size)

        # SOBOLEV REGULARITY
        if with_regularity_loss:
            w_t = torch.rfft(torch.stack(v_t), self.dimension, onesided=False, normalized=True)
            sobolev_regularity = torch.sum(self.L.expand(w_t.size()) * w_t ** 2) / float(ntp)
            return x, sobolev_regularity

        else:
            return x

    def tamper_template_gradient(self, kernel, gamma, lr, print_info=False):
        tampered_template_gradient = (lr * kernel(gamma, self.template_points.detach(), self.template_points.detach(),
                                                  self.template_points.grad.detach())).detach()
        self.template_points.grad = tampered_template_gradient
        if print_info:
            print('tampered template gradient max absolute value = %.3f' %
                  torch.max(torch.abs(tampered_template_gradient)))

    # def update_template(self, kernel, gamma, lr):
    #     update = - lr * kernel(
    #         gamma, self.template_points.detach(), self.template_points.detach(), self.template_points.grad.detach())
    #     self.template_points = self.template_points.detach() + update
    #     self.template_points.requires_grad_()
    #     print('template update min = %.3f ; max = %.3f' % (torch.min(update), torch.max(update)))

    def write_starmen(self, splats, points, connectivities, vizualisation_grid, prefix):

        # INITIALIZE
        batch_size = splats.size(0)

        # ENCODE
        q, _ = model.encode(splats)

        # DECODE
        v_t = []
        # dq0 = self.decoder(q * 0.0)
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))
            # dq1 = self.decoder(q * (t + 1) * self.dt)
            # v_t.append((dq1 - dq0) / self.dt)
            # dq0 = dq1

        # FLOW AND WRITE
        x = self.template_points.clone().view(1, -1, self.dimension).repeat(batch_size, 1, 1)
        g = vizualisation_grid.clone().view(1, -1, self.dimension).repeat(batch_size, 1, 1)

        # plot_registrations(
        #     self.template_points.view(1, -1, self.dimension).expand(batch_size, -1, self.dimension), points,
        #     self.template_connectivity.view(1, -1, self.dimension).expand(batch_size, -1, self.dimension), connectivities,
        #     x, g.view(batch_size, visualization_grid_size, visualization_grid_size, self.dimension),
        #     deformation_grid, v_t[0].permute(0, 2, 3, 1),
        #     prefix, 'j_%d' % (0))

        for j, v in enumerate(v_t):
            x += self.dt * batched_bilinear_interpolation(v, x, self.bounding_box, self.deformation_grid_size)
            g += self.dt * batched_bilinear_interpolation(v, g, self.bounding_box, self.deformation_grid_size)

            # plot_registrations(
            #     self.template_points.view(1, -1, self.dimension).expand(batch_size, -1, self.dimension), points,
            #     self.template_connectivity.view(1, -1, self.dimension).expand(batch_size, -1, self.dimension), connectivities,
            #     x, g.view(batch_size, visualization_grid_size, visualization_grid_size, self.dimension),
            #     deformation_grid, v.permute(0, 2, 3, 1),
            #     prefix, 'j_%d' % (j + 1))

        plot_registrations(
            self.template_points.view(1, -1, self.dimension).expand(n, -1, self.dimension), points,
            self.template_connectivity.view(1, -1, self.dimension).expand(n, -1, self.dimension), connectivities,
            x, g.view(batch_size, visualization_grid_size, visualization_grid_size, self.dimension),
            deformation_grid, torch.mean(torch.stack(v_t), dim=0).permute(0, 2, 3, 1),
            prefix, '')

    def write_meshes(self, splats, points, connectivities, prefix):

        # INITIALIZE
        batch_size = splats.size(0)

        # ENCODE
        q, _ = model.encode(splats)

        # DECODE
        v_t = []
        # dq0 = self.decoder(q * 0.0)
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))
            # dq1 = self.decoder(q * (t + 1) * self.dt)
            # v_t.append((dq1 - dq0) / self.dt)
            # dq0 = dq1

        # FLOW AND WRITE
        x = self.template_points.clone().view(1, -1, self.dimension).repeat(batch_size, 1, 1)

        write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                     prefix + '__', '__j_%d' % 0,
                     targets=[(elt_p.detach().cpu().numpy(), elt_c.detach().cpu().numpy())
                              for elt_p, elt_c in zip(points, connectivities)])

        for j, v in enumerate(v_t):
            x += self.dt * batched_bilinear_interpolation(v, x, self.bounding_box, self.deformation_grid_size)

            write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                         prefix + '__', '__j_%d' % (j + 1))
            write_deformations(v, self.deformation_grid.detach().cpu().numpy(),
                               prefix + '__', '__vfield__j_%d' % (j))


if __name__ == '__main__':

    ############################
    ##### GLOBAL VARIABLES #####
    ############################

    # experiment_prefix = '39_bayesian_atlas_fourier__alpha_0.5'
    # experiment_prefix = '4_bayesian_atlas_fourier__latent_space_2__lambda_1e-4'
    # experiment_prefix = '1_bayesian_atlas_fourier__latent_space_2__lambda_1e-4'
    # experiment_prefix = '47_bayesian_atlas_fourier__latent_10__current_5__lambda_1e-6__grid_16__dynamic__all_5__new_sobolev'

    # MODEL

    dataset = 'hippocampi'
    # dataset = 'circles'
    # dataset = 'ellipsoids'
    # dataset = 'starmen'
    # dataset = 'leaves'
    # dataset = 'squares'

    number_of_meshes_train = 32
    number_of_meshes_test = 0

    splatting_grid_size = 16
    deformation_grid_size = 16
    visualization_grid_size = 32

    number_of_time_points = 11

    initialize_template = None
    initialize_encoder = None
    initialize_decoder = None

    initial_state = None
    initial_encoder_state = None
    initial_decoder_state = None

    # OPTIMIZATION ------------------------------
    number_of_epochs = 100
    number_of_epochs_for_init = 1000
    number_of_epochs_for_warm_up = 0
    print_every_n_iters = 1
    save_every_n_iters = 100

    learning_rate = 5e-3
    learning_rate_decay = 0.95
    learning_rate_ratio = 5e-5

    batch_size = 32

    # device = 'cuda:01'
    device = 'cpu'
    # -------------------------------------------

    ############################
    ######## INITIALIZE ########
    ############################

    if dataset == 'starmen':
        experiment_prefix = '41_registration__alpha_0.5'
        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data'))

        # initialize_template = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__Template_hippocampus.vtk')
        # initialize_encoder = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')
        initialize_decoder = os.path.join(path_to_meshes, 'latent_positions.txt')

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/hippocampi/output__49_bayesian_atlas_fourier__latent_10__162_subjects/epoch_0__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_encoder__epoch_9000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        number_of_meshes_train = 1
        number_of_meshes_test = 1

        splatting_grid_size = 16
        deformation_grid_size = 32
        visualization_grid_size = 16

        dimension = 2
        latent_dimension = 1
        number_of_time_points = 5

        splatting_kernel_width = 1.0

        alpha = 0.5
        lambda_ = 1.
        noise_variance = 0.01 ** 2

        gamma_splatting = torch.from_numpy(np.array([1. / splatting_kernel_width ** 2.])).float()
        gkernel = Genred("Exp( - G * SqDist(X, Y) ) * P",
                         ['G = Pm(1)', 'X = Vx(2)', 'Y = Vy(2)', 'P = Vy(2)'],
                         reduction_op='Sum', axis=1)
        print('>> Gaussian kernel formula: %s' % gkernel.formula)

        bounding_box = torch.from_numpy(np.array([[-2.5, 2.5], [-2.5, 2.5]])).float()
        bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        (points, connectivities, splats,
         points_test, connectivities_test, splats_test) = create_cross_sectional_starmen_dataset(
            path_to_meshes, number_of_meshes_train, number_of_meshes_test,
            splatting_grid, splatting_kernel_width, dimension, random_seed=42)

        # OPTIMIZATION ------------------------------
        number_of_epochs = 100000
        number_of_epochs_for_init = 100000
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 1000
        save_every_n_iters = 5000

        learning_rate = 5e-4
        learning_rate_decay = 0.95
        learning_rate_ratio = 1e-4

        batch_size = 1

        # device = 'cuda:01'
        device = 'cpu'
        # -------------------------------------------

    elif dataset == 'ellipsoids':
        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/ellipsoids/data'))

        dimension = 3
        splatting_kernel_width = 0.15

        alpha = 3.
        lambda_ = 1e-2
        noise_variance = 0.01 ** 2

        latent_dimension = 2

        bounding_box = torch.from_numpy(np.array([[-1., 1.], [-1., 1.], [-1., 1.]])).float()
        bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.], [-2., 2.]])).float()

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        (points, connectivities, splats,
         points_test, connectivities_test, splats_test) = create_cross_sectional_ellipsoids_dataset(
            os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/ellipsoids/data')),
            number_of_meshes_train, number_of_meshes_test,
            splatting_grid, splatting_kernel_width, dimension, random_seed=42)

    elif dataset == 'circles':
        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/circles/data'))

        dimension = 2
        splatting_kernel_width = 0.15

        alpha = 1.
        lambda_ = 1e-4
        noise_variance = 0.01 ** 2

        latent_dimension = 2

        bounding_box = torch.from_numpy(np.array([[-1., 1.], [-1., 1.]])).float()
        bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        (points, connectivities, splats,
         points_test, connectivities_test, splats_test) = create_cross_sectional_circles_dataset(
            os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/circles/data')),
            number_of_meshes_train, number_of_meshes_test,
            splatting_grid, splatting_kernel_width, dimension, random_seed=42)

    elif dataset == 'leaves':
        experiment_prefix = '6_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__less_deep'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/leaves/data'))
        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/leaves/output__4_bayesian_atlas_fourier__latent_space_2__fixed_template__64_subjects__lambda_10__alpha_0.5/epoch_20000__model.pth'))
        initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                      '../../../examples/leaves/output__6_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__less_deep/epoch_20000__model.pth'))

        number_of_meshes_train = 441
        number_of_meshes_test = 0

        splatting_grid_size = 16
        deformation_grid_size = 32
        visualization_grid_size = 16

        number_of_time_points = 11

        dimension = 2
        splatting_kernel_width = 0.2

        alpha = 0.5
        lambda_ = 5.
        noise_variance = 0.01 ** 2

        latent_dimension = 2

        gamma_splatting = torch.from_numpy(np.array([1. / splatting_kernel_width ** 2.])).float()
        gkernel = Genred("Exp( - G * SqDist(X, Y) ) * P",
                         ['G = Pm(1)', 'X = Vx(2)', 'Y = Vy(2)', 'P = Vy(2)'],
                         reduction_op='Sum', axis=1)
        print('>> Gaussian kernel formula: %s' % gkernel.formula)

        # bounding_box = torch.from_numpy(np.array([[-1., 1.], [-1., 1.]])).float()
        bounding_box = torch.from_numpy(np.array([[-1.0, 0.6], [-0.8, 0.8]])).float()
        bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        (points, connectivities, splats,
         points_test, connectivities_test, splats_test) = create_cross_sectional_leaves_dataset(
            os.path.normpath(os.path.join(os.path.dirname(__file__), path_to_meshes)),
            number_of_meshes_train, number_of_meshes_test,
            splatting_grid, splatting_kernel_width, dimension, random_seed=42)

        # OPTIMIZATION --------------
        number_of_epochs = 50000
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 100
        save_every_n_iters = 1000

        learning_rate = 5e-4
        learning_rate_decay = 1.
        learning_rate_ratio = 5e-5

        batch_size = 32

        device = 'cuda:01'
        # device = 'cpu'
        # ----------------------------

    elif dataset == 'hippocampi':
        experiment_prefix = '53_bayesian_atlas_fourier__registration_test_162__continued'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/hippocampi/data'))

        # initialize_template = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__Template_hippocampus.vtk')
        # initialize_encoder = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')
        # initialize_decoder = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')

        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_encoder__epoch_9000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        initial_state = os.path.normpath(
            os.path.join(os.path.dirname(__file__),
                         '../../../examples/hippocampi/'
                         'output__52_bayesian_atlas_fourier__latent_10__162_subjects/epoch_25000__model.pth'))

        number_of_meshes_train = 162
        number_of_meshes_test = 0

        splatting_grid_size = 16
        deformation_grid_size = 32
        visualization_grid_size = 16

        number_of_time_points = 11

        dimension = 3
        latent_dimension = 10

        splatting_kernel_width = 5.
        varifold_kernel_width = 5.

        alpha = 0.5
        lambda_ = 1.
        noise_variance = 0.1 ** 2

        gamma_splatting = torch.from_numpy(np.array([1. / splatting_kernel_width ** 2.])).float()
        gamma_varifold = torch.from_numpy(np.array([1. / varifold_kernel_width ** 2.])).float()

        gkernel = Genred("Exp( - G * SqDist(X, Y) ) * P",
                         ['G = Pm(1)', 'X = Vx(3)', 'Y = Vy(3)', 'P = Vy(3)'],
                         reduction_op='Sum', axis=1)
        vkernel = Genred("Exp( - G * SqDist(X, Y) ) * Square( (Nx | Ny) ) * P",
                         ['G = Pm(1)', 'X = Vx(3)', 'Y = Vy(3)', 'Nx = Vx(3)', 'Ny = Vy(3)', 'P = Vy(1)'],
                         reduction_op='Sum', axis=1)
        print('>> Gaussian kernel formula: %s' % gkernel.formula)
        print('>> Varifold kernel formula: %s' % vkernel.formula)

        # bounding_box = torch.from_numpy(np.array([[5., 45.], [-45., 0.], [-35., 10.]])).float()
        bounding_box = torch.from_numpy(np.array([[0., 50.], [-50., 5.], [-40., 15.]])).float()
        bounding_box_visualization = bounding_box

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        points, connectivities, centers, normals, norms, splats = \
            create_cross_sectional_hippocampi_dataset__for_registration(
                path_to_meshes,
                number_of_meshes_train, number_of_meshes_test,
                splatting_grid, dimension, gkernel, gamma_splatting, random_seed=42)

        # OPTIMIZATION --------------
        number_of_epochs = 25000

        print_every_n_iters = 100
        save_every_n_iters = 10000

        learning_rate = 5e-4

        batch_size = 32

        device = 'cuda:01'
        # device = 'cpu'
        # ----------------------------

    elif dataset == 'squares':
        experiment_prefix = '6_bayesian_atlas_fourier__latent_space_2__all_subjects'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/data'))

        initialize_template = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                            '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__Template_square.vtk'))
        initialize_encoder = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))
        initialize_decoder = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__3_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init__except_template/epoch_11000__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__33_bayesian_atlas_fourier_image/init_encoder__epoch_25000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        number_of_meshes_train = 441
        number_of_meshes_test = 0

        splatting_grid_size = 32
        deformation_grid_size = 32
        visualization_grid_size = 16

        number_of_time_points = 11

        dimension = 2
        splatting_kernel_width = 0.2

        alpha = 0.5
        lambda_ = 1.
        noise_variance = 0.01 ** 2

        latent_dimension = 2

        gamma_splatting = torch.from_numpy(np.array([1. / splatting_kernel_width ** 2.])).float()
        gkernel = Genred("Exp( - G * SqDist(X, Y) ) * P",
                         ['G = Pm(1)', 'X = Vx(2)', 'Y = Vy(2)', 'P = Vy(2)'],
                         reduction_op='Sum', axis=1)
        print('>> Gaussian kernel formula: %s' % gkernel.formula)

        # bounding_box = torch.from_numpy(np.array([[-1., 1.], [-1., 1.]])).float()
        bounding_box = torch.from_numpy(np.array([[-0.8, 0.8], [-0.8, 0.8]])).float()
        bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

        splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
        deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
        visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

        (points, connectivities, splats,
         points_test, connectivities_test, splats_test) = create_cross_sectional_squares_dataset(
            os.path.normpath(os.path.join(os.path.dirname(__file__), path_to_meshes)),
            number_of_meshes_train, number_of_meshes_test,
            splatting_grid, splatting_kernel_width, dimension, random_seed=42)

        # OPTIMIZATION --------------
        number_of_epochs = 25000
        number_of_epochs_for_init = 25000
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 100
        save_every_n_iters = 1000

        learning_rate = 5e-4
        learning_rate_decay = 1.
        learning_rate_ratio = 5e-5

        batch_size = 32

        device = 'cuda:01'
        # device = 'cpu'
        # ----------------------------

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

    ##################
    ###### MAIN ######
    ##################

    np.random.seed()

    if dataset in ['circles', 'ellipsoids', 'starmen', 'leaves', 'squares']:
        noise_dimension = points.size(1)

        if initialize_template is not None:
            initial_p, initial_c = read_vtk_file(initialize_template, dimension=dimension, extract_connectivity=True)
        else:
            initial_p = torch.mean(points, dim=0)
            # initial_p = points_test[0]
            initial_c = connectivities[0]

        if 'cuda' in device:
            initial_p = initial_p.cuda()
            initial_c = initial_c.cuda()
        model = BayesianAtlas(initial_p, initial_c,
                              bounding_box, latent_dimension, alpha,
                              splatting_grid, deformation_grid, number_of_time_points)
    elif dataset == 'hippocampi':
        noise_dimension = 10e3

        if initialize_template is not None:
            initial_p, initial_c = read_vtk_file(initialize_template, dimension=dimension, extract_connectivity=True)
        else:
            # path_to_initial_hippocampus = os.path.join(
            #     path_to_meshes, 'DeterministicAtlas__Reconstruction__mesh__subject_colin27.vtk')
            # path_to_initial_hippocampus = os.path.join(path_to_meshes, 'right_hippocampus_ellipsoid.vtk')
            path_to_initial_hippocampus = os.path.join(path_to_meshes, 'epoch_1000__train__subject_2__j_0.vtk')
            initial_p, initial_c = read_vtk_file(path_to_initial_hippocampus,
                                                 dimension=dimension, extract_connectivity=True)
        if 'cuda' in device:
            initial_p = initial_p.cuda()
            initial_c = initial_c.cuda()

        model = BayesianAtlas(initial_p, initial_c,
                              bounding_box, latent_dimension, alpha,
                              splatting_grid, deformation_grid, number_of_time_points)
    else:
        raise RuntimeError

    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = Adam([model.template_points], lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=max(10, int(number_of_epochs / 50.)), gamma=learning_rate_decay)

    if 'cuda' in device:
        # model.template_points = model.template_points.type(torch.cuda.FloatTensor)
        # model.template_connectivity = model.template_connectivity.cuda()
        model.cuda()

        bounding_box = bounding_box.cuda()
        model.bounding_box = model.bounding_box.cuda()
        splatting_grid = splatting_grid.cuda()
        deformation_grid = deformation_grid.cuda()
        visualization_grid = visualization_grid.cuda()

        splats = splats.cuda()

        gamma_splatting = gamma_splatting.cuda()

        if dataset == 'hippocampi':
            gamma_varifold = gamma_varifold.cuda()

            centers = [elt.cuda() for elt in centers]
            normals = [elt.cuda() for elt in normals]
            norms = [elt.cuda() for elt in norms]

            points = [elt.cuda() for elt in points]
            connectivities = [elt.cuda() for elt in connectivities]

        else:
            points = points.cuda()
            connectivities = connectivities.cuda()


    ########################
    ###### INITIALIZE ######
    ########################

    # LOAD INITIALIZATIONS ----------------------------------------
    if initial_state is not None:
        print('>> initial_state = %s' % os.path.basename(initial_state))
        state_dict = torch.load(initial_state, map_location=lambda storage, loc: storage)
        # if 'cuda' in device:
        #     state_dict = state_dict.cuda()
        model.load_state_dict(state_dict)
    # -------------------------------------------------------------

    batch_target_splats = splats
    if dimension == 2:
        batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
    elif dimension == 3:
        batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
    else:
        raise RuntimeError

    latent_momenta, _ = model.encode(batch_target_splats)
    latent_momenta = nn.Parameter(latent_momenta)

    optimizer = Adam([latent_momenta], lr=learning_rate)
    for epoch in range(number_of_epochs + 1):
        # scheduler.step()

        #############
        ### TRAIN ###
        #############

        train_attachment_loss = 0.
        train_sobolev_regularity_loss = 0.
        train_kullback_regularity_loss = 0.
        train_total_loss = 0.

        indexes = np.random.permutation(len(splats))
        for k in range(number_of_meshes_train // batch_size):  # drops the last batch
            batch_target_splats = splats[indexes[k * batch_size:(k + 1) * batch_size]]
            batch_latent_momenta = latent_momenta[indexes[k * batch_size:(k + 1) * batch_size]]

            if dimension == 2:
                batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
            elif dimension == 3:
                batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
            else:
                raise RuntimeError

            # DECODE
            deformed_template_points, sobolev_regularity_loss = model(batch_latent_momenta, with_regularity_loss=True)

            # LOSS
            if dataset in ['circles', 'ellipsoids', 'starmen', 'leaves', 'squares']:
                batch_target_points = points[indexes[k * batch_size:(k + 1) * batch_size]]
                attachment_loss = torch.sum((deformed_template_points - batch_target_points) ** 2) / noise_variance

            elif dataset == 'hippocampi':
                batch_target_centers = [centers[index] for index in indexes[k * batch_size:(k + 1) * batch_size]]
                batch_target_normals = [normals[index] for index in indexes[k * batch_size:(k + 1) * batch_size]]
                batch_target_norms = [norms[index] for index in indexes[k * batch_size:(k + 1) * batch_size]]

                # Current
                attachment_loss = 0.0
                for p1, c2, n2, norm2 in zip(
                        deformed_template_points, batch_target_centers, batch_target_normals, batch_target_norms):
                    c1, n1 = compute_centers_and_normals(p1, model.template_connectivity)
                    attachment_loss += (
                            norm2 +
                            torch.sum(n1 * gkernel(gamma_splatting, c1, c1, n1)) - 2 *
                            torch.sum(n1 * gkernel(gamma_splatting, c1, c2, n2)))
                attachment_loss /= noise_variance

            else:
                raise RuntimeError

            train_attachment_loss += attachment_loss.detach().cpu().numpy()

            sobolev_regularity_loss *= lambda_
            train_sobolev_regularity_loss += sobolev_regularity_loss.detach().cpu().numpy()

            # kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
            # train_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

            total_loss = attachment_loss + sobolev_regularity_loss
            train_total_loss += total_loss.detach().cpu().numpy()

            # GRADIENT STEP
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        train_attachment_loss /= float(batch_size * (len(splats) // batch_size))
        train_sobolev_regularity_loss /= float(batch_size * (len(splats) // batch_size))
        train_kullback_regularity_loss /= float(batch_size * (len(splats) // batch_size))
        train_total_loss /= float(batch_size * (len(splats) // batch_size))

        #############
        ### WRITE ###
        #############

        if epoch % print_every_n_iters == 0 or epoch == number_of_epochs:
            log += cprint(
                '\n[Epoch: %d] Learning rate = %.2E ; Noise std = %.2E'
                '\nTrain loss = %.3f (attachment = %.3f ; sobolev regularity = %.3f ; kullback regularity = %.3f)' %
                (epoch, list(optimizer.param_groups)[0]['lr'], math.sqrt(noise_variance),
                 train_total_loss, train_attachment_loss, train_sobolev_regularity_loss, train_kullback_regularity_loss))

        if epoch % save_every_n_iters == 0 or epoch == number_of_epochs:
            with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
                f.write(log)

            torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_%d__model.pth' % epoch))
            np.savetxt(os.path.join(output_dir, 'epoch_%d__latent_momenta.txt' % epoch),
                       latent_momenta.detach().cpu().numpy())

            n = 3
            if dataset == 'starmen' and False:
                model.write_starmen(splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                                    visualization_grid, os.path.join(output_dir, 'epoch_%d__train' % epoch))

            if dataset in ['ellipsoids', 'hippocampi']:
                model.write_meshes(
                    splats[:n].permute(0, 4, 1, 2, 3), points[:n], connectivities[:n],
                    os.path.join(output_dir, 'epoch_%d__train' % epoch))

            if dataset in ['starmen', 'circles', 'leaves', 'squares']:
                model.write_meshes(
                    splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                    os.path.join(output_dir, 'epoch_%d__train' % epoch))

