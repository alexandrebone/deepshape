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


def write_meshes(points, connectivity, prefix, suffix):
    for i, p in enumerate(points):
        write_mesh('%ssubject_%d%s' % (prefix, i, suffix), p, connectivity)


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
        self.down4 = Conv2d_Tanh(16, 16)
        self.linear = nn.Linear(16 * n * n, latent_dimension)
        print('>> Encoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.linear(x.view(x.size(0), -1)).view(x.size(0), -1)
        return x


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
        self.down4 = Conv3d_Tanh(16, 16)
        self.linear = nn.Linear(16 * n * n * n, latent_dimension)
        print('>> Encoder3d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.linear(x.view(x.size(0), -1)).view(x.size(0), -1)
        return x


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


class BayesianAtlas(nn.Module):

    def __init__(self,
                 template_points, template_connectivity,
                 latent_dimension_half, splatting_grid_size, deformation_grid_size,
                 deformation_kernel_width, number_of_time_points):
        nn.Module.__init__(self)

        self.template_connectivity = template_connectivity
        self.latent_dimension_half = latent_dimension_half
        self.deformation_kernel_width = deformation_kernel_width
        self.deformation_grid_size = deformation_grid_size
        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

        self.template_points = nn.Parameter(template_points)
        print('>> Template points are %d x %d = %d parameters' % (
            template_points.size(0), template_points.size(1), template_points.size(0) * template_points.size(1)))
        if dimension == 2:
            self.encoder = Encoder2d(splatting_grid_size, latent_dimension_half * 2)
            self.decoder = Decoder2d(latent_dimension_half, deformation_grid_size)
        elif dimension == 3:
            self.encoder = Encoder3d(splatting_grid_size, latent_dimension_half * 2)
            self.decoder = Decoder3d(latent_dimension_half, deformation_grid_size)
        else:
            raise RuntimeError
        print('>> BayesianAtlas has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def encode(self, observations):
        z = self.encoder(observations)
        return z[:, :latent_dimension_half], z[:, latent_dimension_half:]

    def forward(self, q, with_regularity_loss=False):

        # DECODE
        v_t = []
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))

        # FLOW
        x = self.template_points.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)
        for v in v_t:
            x += batched_bilinear_interpolation(v, x, bounding_box, deformation_grid_size)

        # SOBOLEV REGULARITY
        if with_regularity_loss:
            dgs = self.deformation_grid_size
            alpha = 3.
            s = 3.
            w_t = torch.rfft(torch.stack(v_t), dimension, onesided=False)

            if dimension == 2:
                L = torch.stack(torch.meshgrid([torch.arange(dgs), torch.arange(dgs)])).type(str(w_t.type()))
                L = (- 2 * alpha * (torch.sum(torch.cos(
                    (2.0 * math.pi / float(dgs)) * L), dim=0) - dimension) + 1) ** s
                L = L.view(1, 1, 1, self.deformation_grid_size, self.deformation_grid_size, 1).expand(w_t.size())
            else:
                L = torch.stack(torch.meshgrid([torch.arange(dgs), torch.arange(dgs),
                                                torch.arange(dgs)])).type(str(w_t.type()))
                L = (- 2 * alpha * (torch.sum(torch.cos(
                    (2.0 * math.pi / float(dgs)) * L), dim=0) - dimension) + 1) ** s
                L = L.view(1, 1, 1, dgs, dgs, dgs, 1).expand(w_t.size())

            sobolev_regularity = torch.sum(L * w_t ** 2)
            return x, sobolev_regularity

        else:
            return x

    def write_starmen(self, splats, points, connectivities, vizualisation_grid, prefix):

        # INITIALIZE
        batch_size = splats.size(0)

        # ENCODE
        q, _ = model.encode(splats)

        # DECODE
        v_t = []
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))

        # FLOW AND WRITE
        x = self.template_points.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)
        g = vizualisation_grid.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)

        # plot_registrations(
        #     self.template_points.view(1, -1, dimension).expand(batch_size, -1, dimension), points,
        #     self.template_connectivity.view(1, -1, dimension).expand(batch_size, -1, dimension), connectivities,
        #     x, g.view(batch_size, visualization_grid_size, visualization_grid_size, dimension),
        #     deformation_grid, v_t[0].permute(0, 2, 3, 1),
        #     prefix, 'j_%d' % (0))

        for j, v in enumerate(v_t):
            x += batched_bilinear_interpolation(v.permute(0, 2, 3, 1), x, bounding_box, deformation_grid_size)
            g += batched_bilinear_interpolation(v.permute(0, 2, 3, 1), g, bounding_box, deformation_grid_size)

            # plot_registrations(
            #     self.template_points.view(1, -1, dimension).expand(batch_size, -1, dimension), points,
            #     self.template_connectivity.view(1, -1, dimension).expand(batch_size, -1, dimension), connectivities,
            #     x, g.view(batch_size, visualization_grid_size, visualization_grid_size, dimension),
            #     deformation_grid, v.permute(0, 2, 3, 1),
            #     prefix, 'j_%d' % (j + 1))

        plot_registrations(
            self.template_points.view(1, -1, dimension).expand(n, -1, dimension), points,
            self.template_connectivity.view(1, -1, dimension).expand(n, -1, dimension), connectivities,
            x, g.view(batch_size, visualization_grid_size, visualization_grid_size, dimension),
            deformation_grid, torch.mean(torch.stack(v_t), dim=0).permute(0, 2, 3, 1),
            prefix, '')

    def write_ellipsoids(self, splats, points, connectivities, prefix):

        # INITIALIZE
        batch_size = splats.size(0)

        # ENCODE
        q, _ = model.encode(splats)

        # DECODE
        v_t = []
        for t in range(self.number_of_time_points - 1):
            v_t.append(self.decoder(q * (t + 1) * self.dt))

        # FLOW AND WRITE
        x = self.template_points.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)

        write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                     prefix + '__', '__j_%d' % 0)

        for j, v in enumerate(v_t):
            x += batched_bilinear_interpolation(v, x, bounding_box, deformation_grid_size)

            write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                         prefix + '__', '__j_%d' % (j + 1))


############################
##### GLOBAL VARIABLES #####
############################

# experiment_prefix = '39_bayesian_atlas_fourier__small_latent_space'
experiment_prefix = '1_bayesian_atlas_fourier__latent_space_3'

# MODEL

# starmen or ellipoids
dataset = 'ellipsoids'
# dataset = 'starmen'

number_of_meshes_train = 16
number_of_meshes_test = 16

splatting_grid_size = 16
deformation_grid_size = 16
visualization_grid_size = 32

latent_dimension_half = 3
number_of_time_points = 6

# OPTIMIZATION

number_of_epochs = 1000
print_every_n_iters = 1
save_every_n_iters = 100

batch_size = 16

learning_rate = 1e-3

device = 'cpu'

lambda_ = 1e-2

############################
######## INITIALIZE ########
############################


if dataset == 'starmen':
    path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data'))

    dimension = 2
    splatting_kernel_width = 1.0
    deformation_kernel_width = 1.0

    bounding_box = torch.from_numpy(np.array([[-2.5, 2.5], [-2.5, 2.5]])).float()
    bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

    splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
    deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
    visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

    (points, connectivities, splats,
     points_test, connectivities_test, splats_test) = create_cross_sectional_starmen_dataset(
        path_to_meshes, number_of_meshes_train, number_of_meshes_test,
        splatting_grid, splatting_kernel_width, dimension, random_seed=42)

elif dataset == 'ellipsoids':
    path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/ellipsoids/data'))

    dimension = 3
    splatting_kernel_width = 0.5
    deformation_kernel_width = 0.5

    bounding_box = torch.from_numpy(np.array([[-1., 1.], [-1., 1.], [-1., 1.]])).float()
    # bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

    splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
    deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)

    (points, connectivities, splats,
     points_test, connectivities_test, splats_test) = create_cross_sectional_ellipsoids_dataset(
        os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/ellipsoids/data')),
        number_of_meshes_train, number_of_meshes_test,
        splatting_grid, splatting_kernel_width, dimension, random_seed=42)
else:
    raise RuntimeError

latent_dimension = latent_dimension_half * 2

assert number_of_time_points > 1

log = ''
output_dir = os.path.join(path_to_meshes, '../output__' + experiment_prefix)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if (not torch.cuda.is_available()) and device == 'cuda':
    device = 'cpu'
    print('>> CUDA is not available. Overridding with device = "cpu".')

############################
###### TRAIN AND TEST ######
############################


noise_variance = 0.01 ** 2
noise_dimension = points.size(1)

model = BayesianAtlas(torch.mean(points, dim=0), connectivities[0],
                      latent_dimension_half, splatting_grid_size, deformation_grid_size,
                      deformation_kernel_width, number_of_time_points)

optimizer = Adam(model.parameters(), lr=learning_rate)

if device == 'cuda':
    model.template_connectivity = model.template_connectivity.cuda()
    model.cuda()

    bounding_box = bounding_box.cuda()
    splatting_grid = splatting_grid.cuda()
    deformation_grid = deformation_grid.cuda()
    visualization_grid = visualization_grid.cuda()

    points = points.cuda()
    connectivities = connectivities.cuda()
    splats = splats.cuda()

    points_test = points_test.cuda()
    connectivities_test = connectivities_test.cuda()
    splats_test = splats_test.cuda()

for epoch in range(number_of_epochs + 1):

    #############
    ### TRAIN ###
    #############

    train_attachment_loss = 0.
    train_sobolev_regularity_loss = 0.
    train_kullback_regularity_loss = 0.
    train_total_loss = 0.

    indexes = np.random.permutation(number_of_meshes_train)
    for k in range(number_of_meshes_train // batch_size):  # drops the last batch
        batch_target_points = points[indexes[k * batch_size:(k + 1) * batch_size]]
        batch_target_connec = connectivities[indexes[k * batch_size:(k + 1) * batch_size]]
        batch_target_splats = splats[indexes[k * batch_size:(k + 1) * batch_size]]

        if dimension == 2:
            batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
        elif dimension == 3:
            batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
        else:
            raise RuntimeError

        # ENCODE, SAMPLE AND DECODE
        means, log_variances = model.encode(batch_target_splats)
        batch_latent_momenta = means + torch.zeros_like(means).normal_() * torch.exp(0.5 * log_variances)
        deformed_template_points, sobolev_regularity_loss = model(batch_latent_momenta, with_regularity_loss=True)

        # LOSS
        attachment_loss = torch.sum((deformed_template_points - batch_target_points) ** 2) / noise_variance
        train_attachment_loss += attachment_loss.detach().cpu().numpy()

        sobolev_regularity_loss *= lambda_
        train_sobolev_regularity_loss += sobolev_regularity_loss.detach().cpu().numpy()

        kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
        train_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

        total_loss = attachment_loss + sobolev_regularity_loss + kullback_regularity_loss
        train_total_loss += total_loss.detach().cpu().numpy()

        # GRADIENT STEP
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # noise_variance *= float(attachment_loss.detach().cpu().numpy() / float(noise_dimension * batch_size))

    train_attachment_loss /= float(batch_size * (number_of_meshes_train // batch_size))
    train_sobolev_regularity_loss /= float(batch_size * (number_of_meshes_train // batch_size))
    train_kullback_regularity_loss /= float(batch_size * (number_of_meshes_train // batch_size))
    train_total_loss /= float(batch_size * (number_of_meshes_train // batch_size))

    ############
    ### TEST ###
    ############

    test_attachment_loss = 0.
    test_sobolev_regularity_loss = 0.
    test_kullback_regularity_loss = 0.
    test_total_loss = 0.

    batch_target_points = points_test
    batch_target_connec = connectivities_test
    batch_target_splats = splats_test

    if dimension == 2:
        batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
    elif dimension == 3:
        batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
    else:
        raise RuntimeError

    # ENCODE, SAMPLE AND DECODE
    batch_latent_momenta, log_variances = model.encode(batch_target_splats)
    deformed_template_points, sobolev_regularity_loss = model(batch_latent_momenta, with_regularity_loss=True)

    # LOSS
    attachment_loss = torch.sum((deformed_template_points - batch_target_points) ** 2) / noise_variance
    test_attachment_loss += attachment_loss.detach().cpu().numpy()

    sobolev_regularity_loss *= lambda_
    test_sobolev_regularity_loss += sobolev_regularity_loss.detach().cpu().numpy()

    kullback_regularity_loss = - torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
    test_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

    total_loss = attachment_loss + sobolev_regularity_loss + kullback_regularity_loss
    test_total_loss += total_loss.detach().cpu().numpy()

    test_attachment_loss /= float(number_of_meshes_test)
    test_sobolev_regularity_loss /= float(number_of_meshes_test)
    test_kullback_regularity_loss /= float(number_of_meshes_test)
    test_total_loss /= float(number_of_meshes_test)

    ################
    ### TEMPLATE ###
    ################

    template_splat = splat_current_on_grid(model.template_points, model.template_connectivity,
                                           splatting_grid, splatting_kernel_width)
    if dimension == 2:
        template_splat = template_splat.permute(2, 0, 1)
    elif dimension == 3:
        template_splat = template_splat.permute(3, 0, 1, 2)
    template_latent_momenta, _ = model.encode(template_splat.view((1,) + template_splat.size()))
    template_latent_momenta_norm = float(torch.norm(template_latent_momenta[0], p=2).detach().cpu().numpy())

    #############
    ### WRITE ###
    #############

    if epoch % print_every_n_iters == 0:
        log += cprint(
            '\n[Epoch: %d] Noise std = %.2E ; Template latent q norm = %.3f'
            '\nTrain loss = %.3f (attachment = %.3f ; sobolev regularity = %.3f ; kullback regularity = %.3f)'
            '\nTest  loss = %.3f (attachment = %.3f ; sobolev regularity = %.3f ; kullback regularity = %.3f)' %
            (epoch, math.sqrt(noise_variance), template_latent_momenta_norm,
             train_total_loss, train_attachment_loss, train_sobolev_regularity_loss, train_kullback_regularity_loss,
             test_total_loss, test_attachment_loss, test_sobolev_regularity_loss, test_kullback_regularity_loss))

    if epoch % save_every_n_iters == 0 and not epoch == 0:
        with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
            f.write(log)

        if dataset == 'starmen':
            n = 3
            model.write_starmen(splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                                visualization_grid, os.path.join(output_dir, 'epoch_%d__train' % epoch))
            model.write_starmen(splats_test[:n].permute(0, 3, 1, 2), points_test[:n], connectivities_test[:n],
                                visualization_grid, os.path.join(output_dir, 'epoch_%d__test' % epoch))
            model.write_starmen(template_splat.view((1,) + template_splat.size()),
                                model.template_points.view((1,) + model.template_points.size()),
                                model.template_connectivity.view((1,) + model.template_connectivity.size()),
                                visualization_grid, os.path.join(output_dir, 'epoch_%d__template' % epoch))

        elif dataset == 'ellipsoids':
            n = 3
            model.write_ellipsoids(splats[:n].permute(0, 4, 1, 2, 3), points[:n], connectivities[:n],
                                   os.path.join(output_dir, 'epoch_%d__train' % epoch))
            model.write_ellipsoids(splats_test[:n].permute(0, 4, 1, 2, 3), points_test[:n], connectivities_test[:n],
                                   os.path.join(output_dir, 'epoch_%d__test' % epoch))
            model.write_ellipsoids(template_splat.view((1,) + template_splat.size()),
                                   model.template_points.view((1,) + model.template_points.size()),
                                   model.template_connectivity.view((1,) + model.template_connectivity.size()),
                                   os.path.join(output_dir, 'epoch_%d__template' % epoch))
