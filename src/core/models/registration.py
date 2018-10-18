### Base ###
import os
import numpy as np 
import torch 
import torch.nn as nn
from torch.optim import Adam
import fnmatch
from torch.utils.data import TensorDataset, DataLoader
import itertools

### Visualization ###
#import seaborn as sns
#sns.set(color_codes=True)
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
# rc('font', **{'family':'serif','serif':['Palatino']})
import pandas as pd
import re


def read_vtk_file(filename, dimension, extract_connectivity=False):
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


def splat_current_on_grid(points, connectivity, grid, kernel_type='torch', kernel_width=1.):
    dimension = points.shape[1]
    centers, normals = compute_centers_and_normals(points, connectivity)
    return convolve(grid.view(-1, dimension), centers, normals, kernel_width).view(grid.size())


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class fully_connected(nn.Module):
    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x.view(-1, self.in_ch)).view(x.size())


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        nn.Module.__init__(self)
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)
        self.convolve = nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x_, x):
        x = self.upsample(x)
        x = torch.cat([x_, x], dim=1)
        x = self.convolve(x)
        return x


class unet_voxelmorph(nn.Module):
    """
    in: 64*64*4
    """

    def __init__(self, grid_size):
        n = int(grid_size * 2 ** -4)
        nn.Module.__init__(self)
        self.down1 = down(4, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 32)
        self.down4 = down(32, 32)
        self.fully_connected = fully_connected(32 * n * n, 32 * n * n)
        self.up4 = up(32, 32)
        self.up3 = up(32, 32)
        self.up2 = up(32, 16)
        self.up1 = up(16, 2)
        #        self.out_layer = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        print('Net has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x0 = x
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        y4 = self.fully_connected(x4)
        y3 = self.up4(x3, y4)
        y2 = self.up3(x2, y3)
        y1 = self.up2(x1, y2)
        y0 = self.up1(x0[:, :2], y1)
        #        print('parameter: {}'.format(sum([elt.view(-1)[0] for elt in self.parameters()])))
        return y0


# return self.out_layer(y0)

def deform(points, grid, deformation_field, kernel_type='torch', kernel_width=1.):
    dimension = points.size(1)
    return points + convolve(points, grid.contiguous().view(-1, dimension),
                             deformation_field.permute(1, 2, 0).contiguous().view(-1, dimension), kernel_width)


def compute_L2_attachment(points_1, points_2):
    return torch.sum((points_1.view(-1) - points_2.view(-1)) ** 2)


def compute_loss(sources, targets, deformations,
                 splatting_grid, kernel_type, kernel_width):
    loss = 0.0
    for (source, target, deformation) in zip(sources, targets, deformations):
        deformed_source = deform(source, splatting_grid, deformation,
                                 kernel_type=kernel_type, kernel_width=kernel_width)
        loss += compute_L2_attachment(deformed_source, target)
    return loss


def deform_and_write(sources, targets, deformations,
                     sources_, targets_,
                     splatting_grid, kernel_type, kernel_width,
                     prefix):
    for k, (source, source_, target, target_, deformation) in enumerate(
            zip(sources, sources_, targets, targets_, deformations)):
        deformed_source = deform(source, splatting_grid, deformation,
                                 kernel_type=kernel_type, kernel_width=kernel_width)
        write_mesh(prefix + '__%d__source' % k,
                   source.detach().cpu().numpy(), source_.detach().numpy())
        write_mesh(prefix + '__%d__target' % k,
                   target.detach().cpu().numpy(), target_.detach().numpy())
        write_mesh(prefix + '__%d__target_recontructed' % k,
                   deformed_source.detach().cpu().numpy(), source_.detach().numpy())


def deform_and_plot(sources, targets, deformations,
                    sources_, targets_,
                    splatting_grid, kernel_type, kernel_width,
                    prefix, suffix):
    for k, (source, source_, target, target_, deformation) in enumerate(
            zip(sources, sources_, targets, targets_, deformations)):
        deformed_source = deform(source, splatting_grid, deformation, kernel_type=kernel_type,
                                 kernel_width=kernel_width)
        deformed_grid = deform(splatting_grid.view(-1, dimension), splatting_grid, deformation, kernel_type=kernel_type,
                               kernel_width=kernel_width).view(grid_size, grid_size, dimension).detach().cpu().numpy()

        figsize = 7
        f, axes = plt.subplots(1, 2, figsize=(2 * figsize, figsize))

        ### FIRST FIGURE ###
        ax = axes[0]

        p = source.detach().cpu().numpy()
        c = source_.detach().cpu().numpy()
        ax.plot([p[c[:, 0]][:, 0], p[c[:, 1]][:, 0]],
                [p[c[:, 0]][:, 1], p[c[:, 1]][:, 1]], 'tab:blue', linewidth=2)

        g = deformed_grid
        ax.plot([g[:-1, :, 0].ravel(), g[1:, :, 0].ravel()],
                [g[:-1, :, 1].ravel(), g[1:, :, 1].ravel()], 'k', linewidth=0.5)
        ax.plot([g[:, :-1, 0].ravel(), g[:, 1:, 0].ravel()],
                [g[:, :-1, 1].ravel(), g[:, 1:, 1].ravel()], 'k', linewidth=0.5)

        ax.set_xlim((-2.4, 2.4))
        ax.set_ylim((-2.4, 2.4))

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

        ax.set_xlim((-2.4, 2.4))
        ax.set_ylim((-2.4, 2.4))
        ax.grid()

        f.savefig('%s__%d__%s.pdf' % (prefix, k, suffix), bbox_inches='tight')
        plt.close(f)


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


def create_paired_starmen_train_test_datasets(path_to_starmen, number_of_train_pairs, number_of_test_pairs,
                                              splatting_grid, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_pairs = number_of_train_pairs + number_of_test_pairs
    grid_size = splatting_grid.size(0)

    starmen_files = fnmatch.filter(os.listdir(path_to_starmen), 'SimulatedData__Reconstruction__starman__subject_*')
    starmen_files = sorted(starmen_files, key=extract_subject_and_visit_ids)
    paired_starmen_files__all = np.array(list(itertools.combinations(starmen_files, 2)))

    assert number_of_pairs <= paired_starmen_files__all.shape[0], \
        'Too many required pairs of starmen. A maximum of %d are available' % paired_starmen_files__all.shape[0]

    paired_starmen_files__rdm = paired_starmen_files__all[np.random.choice(
        paired_starmen_files__all.shape[0], size=number_of_pairs, replace=None)]

    sources = []
    targets = []
    sources_ = []
    targets_ = []
    splats = []
    for k, pair_starmen_files in enumerate(paired_starmen_files__rdm):
        path_to_starman_source = os.path.join(path_to_starmen, pair_starmen_files[0])
        path_to_starman_target = os.path.join(path_to_starmen, pair_starmen_files[1])

        starman_source__p, starman_source__c = read_vtk_file(path_to_starman_source, dimension=dimension,
                                                             extract_connectivity=True)
        starman_target__p, starman_target__c = read_vtk_file(path_to_starman_target, dimension=dimension,
                                                             extract_connectivity=True)

        starman_source__s = splat_current_on_grid(starman_source__p, starman_source__c, splatting_grid,
                                                  kernel_type=kernel_type, kernel_width=kernel_width)
        starman_target__s = splat_current_on_grid(starman_target__p, starman_target__c, splatting_grid,
                                                  kernel_type=kernel_type, kernel_width=kernel_width)

        sources.append(starman_source__p)
        sources_.append(starman_source__c)
        targets.append(starman_target__p)
        targets_.append(starman_target__c)
        splats.append(torch.cat((starman_source__s, starman_target__s), dimension).permute(2, 0, 1))

    sources = torch.stack(sources)
    targets = torch.stack(targets)
    sources_ = torch.stack(sources_)
    targets_ = torch.stack(targets_)
    splats = torch.stack(splats)

    return (splats[:number_of_train_pairs], sources[:number_of_train_pairs], sources_[:number_of_train_pairs],
            targets[:number_of_train_pairs], targets_[:number_of_train_pairs],
            splats[number_of_train_pairs:], sources[number_of_train_pairs:], sources_[number_of_train_pairs:],
            targets[number_of_train_pairs:], targets_[number_of_train_pairs:])


############################
##### GLOBAL VARIABLES #####
############################

path_to_starmen = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data'))

dimension = 2
kernel_type = 'torch'
kernel_width = 1.
deformation_multiplier = 1e-6

grid_size = 32
bounding_box = torch.from_numpy(np.array([[-2.5, 2.5], [-2.5, 2.5]]))

device = 'cpu'

experiment_prefix = '1'

number_of_epochs = 5
print_every_n_iters = 1
save_every_n_iters = 5
batch_size = 16

splatting_grid = compute_grid(bounding_box, margin=0., grid_size=grid_size)
number_of_train_pairs = 100
number_of_test_pairs = 10

(splats_train, sources_train, sources_train_, targets_train, targets_train_,
 splats_test,  sources_test,  sources_test_,  targets_test,  targets_test_) = create_paired_starmen_train_test_datasets(
    path_to_starmen, number_of_train_pairs, number_of_test_pairs,
    splatting_grid, dimension, random_seed=42)


############################
######## INITIALIZE ########
############################

output_dir = 'ouput__' + experiment_prefix
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


if (not torch.cuda.is_available()) and device == 'gpu':
    device = 'cpu'
    print('>> CUDA is not available. Overridding with device = "cpu".')


############################
###### TRAIN AND TEST ######
############################

model = unet_voxelmorph(grid_size)
optimizer = Adam(model.parameters(), lr=1e-3)

if device == 'gpu':
    model.cuda()
    splatting_grid.cuda()

for epoch in range(number_of_epochs + 1):

    train_loss_current_epoch = 0.

    ### TRAIN ###
    indexes = np.random.permutation(len(splats_train))
    for k in range(number_of_train_pairs // batch_size):  # drops the last batch
        splats = splats_train[indexes[k * batch_size:(k + 1) * batch_size]]
        sources = sources_train[indexes[k * batch_size:(k + 1) * batch_size]]
        targets = targets_train[indexes[k * batch_size:(k + 1) * batch_size]]

        if device == 'gpu':
            splats.cuda()
            sources.cuda()
            targets.cuda()

        deformations = model(splats)
        loss = compute_loss(
            sources, targets, deformations * deformation_multiplier,
            splatting_grid=splatting_grid, kernel_type=kernel_type, kernel_width=kernel_width)
        train_loss_current_epoch += loss.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_current_epoch /= float(batch_size * (number_of_train_pairs // batch_size))

    ### TEST ###

    deformations = model(splats_test)
    loss = compute_loss(
        sources_test, targets_test, deformations * deformation_multiplier,
        splatting_grid=splatting_grid, kernel_type=kernel_type, kernel_width=kernel_width)
    test_loss_current_epoch = loss.detach().cpu().numpy() / float(number_of_test_pairs)

    if epoch % print_every_n_iters == 0:
        print('[Epoch: %d]  Train loss = %.3f \t Test loss = %.3f' % (
        epoch, train_loss_current_epoch, test_loss_current_epoch))

    if epoch % save_every_n_iters == 0:
        n = 3
        deform_and_plot(sources_train[:n], targets_train[:n], model(splats_train[:n]) * deformation_multiplier,
                        sources_train_[:n], targets_train_[:n],
                        splatting_grid, kernel_type, kernel_width,
                        os.path.join(output_dir, 'train'),
                        'epoch_%d' % epoch)

        deform_and_plot(sources_test[:n], targets_test[:n], model(splats_test[:n]) * deformation_multiplier,
                        sources_test_[:n], targets_test_[:n],
                        splatting_grid, kernel_type, kernel_width,
                        os.path.join(output_dir, 'test'),
                        'epoch_%d' % epoch)