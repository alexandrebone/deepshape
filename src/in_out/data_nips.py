### Base ###
import fnmatch
import math
import os

### Visualization ###
import matplotlib.pyplot as plt

### Core ###
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

### Keops ###
from pykeops.torch import Genred

import nibabel as nib
import PIL.Image as pimg
from torchvision.utils import save_image


### VTK ###
from vtk import vtkPolyData, vtkPoints, vtkDoubleArray, vtkPolyDataWriter


def write_image(fn, intensities):
    tol = 1e-10
    pimg.fromarray(np.clip(intensities[0], tol, 255.0 - tol).astype('uint8')).save(fn + '.png')
    # nib.save(nib.Nifti1Image(np.clip(intensities[0], tol, 255.0 - tol).astype('uint8'), np.eye(4)), fn)


def write_images(intensities, prefix, suffix, targets=None):
    for i, intensities_ in enumerate(intensities):
        write_image('%ssubject_%d%s' % (prefix, i, suffix), intensities_)
        if targets is not None:
            write_image('%ssubject_%d%s' % (prefix, i, '__target'), targets[i])
            write_image('%ssubject_%d%s' % (prefix, i, '__tdiff'), np.abs(intensities_ - targets[i]))


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


def write_gradient(fn, vfield, grid):
    dimension = vfield.size(1)

    x = grid.detach().cpu().numpy()
    v = vfield.detach().cpu().numpy()

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


def squared_distances(x, y):
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)


def convolve(x, y, p, kernel_width):
    sq = squared_distances(x, y)
    return torch.mm(torch.exp(-sq / (kernel_width ** 2)), p)


def splat_current_on_grid(points, connectivity, grid, kernel_width):
    dimension = points.shape[1]
    centers, normals = compute_centers_and_normals(points, connectivity)
    return convolve(grid.view(-1, dimension), centers, normals, kernel_width).view(grid.size())


def compute_noise_dimension(template, multi_object_attachment, dimension, objects_name=None):
    """
    Compute the dimension of the spaces where the norm are computed, for each object.
    """
    assert len(template.object_list) == len(multi_object_attachment.attachment_types)
    assert len(template.object_list) == len(multi_object_attachment.kernels)

    objects_noise_dimension = []
    for k in range(len(template.object_list)):

        if multi_object_attachment.attachment_types[k] in ['current', 'varifold', 'pointcloud']:
            noise_dimension = 1
            for d in range(dimension):
                length = template.bounding_box[d, 1] - template.bounding_box[d, 0]
                assert length >= 0
                noise_dimension *= math.floor(length / multi_object_attachment.kernels[k].kernel_width + 1.0)
            noise_dimension *= dimension

        elif multi_object_attachment.attachment_types[k] in ['landmark']:
            noise_dimension = dimension * template.object_list[k].points.shape[0]

        elif multi_object_attachment.attachment_types[k] in ['L2']:
            noise_dimension = template.object_list[k].intensities.size

        else:
            raise RuntimeError('Unknown noise dimension for the attachment type: '
                               + multi_object_attachment.attachment_types[k])

        objects_noise_dimension.append(noise_dimension)

    if objects_name is not None:
        print('>> Objects noise dimension:')
        for (object_name, object_noise_dimension) in zip(objects_name, objects_noise_dimension):
            print('\t\t[ %s ]\t%d' % (object_name, int(object_noise_dimension)))

    return objects_noise_dimension
