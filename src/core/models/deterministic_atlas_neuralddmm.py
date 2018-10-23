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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import rc


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


def create_cross_sectional_starmen_dataset(path_to_starmen, number_of_starmen,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

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

    return points, connectivities, splats


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


def bilinear_interpolation(velocity, points, bounding_box, grid_size, device='cpu'):
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


class ConvTranspose2d_Tanh(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias),
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


class Decoder(nn.Module):
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
        print('>> Decoder has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.linear(x).view(x.size(0), 16, self.inner_grid_size, self.inner_grid_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class Hamiltonian(nn.Module):
    def __init__(self, in_dimension, inner_dimension=None):
        nn.Module.__init__(self)
        if inner_dimension is None:
            inner_dimension = in_dimension
        self.linear1 = Linear_Tanh(in_dimension, inner_dimension, bias=False)
        self.linear2 = Linear_Tanh(inner_dimension, inner_dimension, bias=False)
        self.linear3 = Linear_Tanh(inner_dimension, 1, bias=False)
        print('>> Hamiltonian has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


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
        m = deformation_field.permute(1, 2, 0).view(-1, dimension).detach().cpu().numpy()
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
        f.savefig('%s__%d__%s.pdf' % (prefix, k, suffix), bbox_inches='tight')
        plt.close(f)


############################
##### GLOBAL VARIABLES #####
############################

experiment_prefix = '15__deterministic_atlas__conserved_norm'

# MODEL

path_to_starmen = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data'))
number_of_starmen = 160

dimension = 2

splatting_kernel_width = 1.0
deformation_kernel_width = 1.0

bounding_box = torch.from_numpy(np.array([[-2.5, 2.5], [-2.5, 2.5]])).float()
bounding_box_visualization = torch.from_numpy(np.array([[-2., 2.], [-2., 2.]])).float()

splatting_grid_size = 32
deformation_grid_size = 8
visualization_grid_size = 32

latent_dimension_half = 10
number_of_time_points = 10

# OPTIMIZATION

number_of_epochs = 1000
print_every_n_iters = 1
save_every_n_iters = 100

batch_size = 16

learning_rate = 1e-3

device = 'cpu'

############################
######## INITIALIZE ########
############################

splatting_grid = compute_grid(bounding_box, margin=0., grid_size=splatting_grid_size)
deformation_grid = compute_grid(bounding_box, margin=0., grid_size=deformation_grid_size)
visualization_grid = compute_grid(bounding_box_visualization, margin=0., grid_size=visualization_grid_size)

points, connectivities, splats = create_cross_sectional_starmen_dataset(
    path_to_starmen, number_of_starmen, splatting_grid, splatting_kernel_width, dimension, random_seed=42)

latent_dimension = latent_dimension_half * 2

if number_of_time_points == 1:
    dt = 1
else:
    dt = 1. / float(number_of_time_points - 1)

log = ''
output_dir = os.path.join(path_to_starmen, '../output__' + experiment_prefix)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if (not torch.cuda.is_available()) and device == 'cuda':
    device = 'cpu'
    print('>> CUDA is not available. Overridding with device = "cpu".')

############################
###### TRAIN AND TEST ######
############################

template_points = torch.mean(points, dim=0).requires_grad_()
template_connec = connectivities[0]
latent_momenta = torch.randn(number_of_starmen, latent_dimension_half, requires_grad=True)

hamiltonian = Hamiltonian(latent_dimension, inner_dimension=latent_dimension_half)
decoder = Decoder(latent_dimension_half, deformation_grid_size)

optimizer = Adam(list(hamiltonian.parameters()) +
                 list(decoder.parameters()) +
                 [template_points, latent_momenta], lr=learning_rate)

if device == 'cuda':
    template_points = template_points.cuda()
    template_connec = template_connec.cuda()
    latent_momenta = latent_momenta.cuda()

    hamiltonian = hamiltonian.cuda()
    decoder = decoder.cuda()

    bounding_box = bounding_box.cuda()
    splatting_grid = splatting_grid.cuda()
    deformation_grid = deformation_grid.cuda()
    visualization_grid = visualization_grid.cuda()

    points = points.cuda()
    connectivities = connectivities.cuda()
    # splats = splats.cuda()

for epoch in range(number_of_epochs + 1):

    #############
    ### TRAIN ###
    #############

    train_loss_current_epoch = 0.

    indexes = np.random.permutation(number_of_starmen)
    for k in range(number_of_starmen // batch_size):  # drops the last batch
        batch_target_points = points[indexes[k * batch_size:(k + 1) * batch_size]]
        batch_target_connec = connectivities[indexes[k * batch_size:(k + 1) * batch_size]]
        batch_latent_momenta = latent_momenta[indexes[k * batch_size:(k + 1) * batch_size]]

        # SHOOT
        p_t = [torch.zeros(batch_size, latent_dimension_half, device=device).requires_grad_()]
        q_t = [batch_latent_momenta]
        for t in range(number_of_time_points):
            pq = torch.cat((p_t[t], q_t[t]), dim=1)
            h = hamiltonian(pq)
            dh_dp, dh_dq = torch.autograd.grad([h], [p_t[t], q_t[t]], [torch.ones(h.size(), device=device)],
                                               create_graph=True)
            p_t.append(p_t[t] + dt * dh_dq)
            q_t.append(q_t[t] - dt * dh_dp)

        # DECODE
        q0_norm = torch.norm(q_t[0], p=2, dim=1)
        v_t = []
        for (p, q) in zip(p_t, q_t):
            # pq = torch.cat((p, q / q_norm.view(-1, 1).expand(q.size())), dim=1)
            qt_norm = torch.norm(q, p=2, dim=1)
            v_t.append(decoder(q * (q0_norm / qt_norm).view(-1, 1).expand(q.size())))
            # v_t.append(v * q_norm.view(-1, 1, 1, 1).expand(v.size()))

        # FLOW
        deformed_template_points = template_points.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)
        for vs in v_t:
            for p, v in zip(deformed_template_points, vs):
                p += convolutive_interpolation(v.permute(1, 2, 0), p, deformation_grid, deformation_kernel_width)
                # p += bilinear_interpolation(v.permute(1, 2, 0), p, bounding_box, deformation_grid_size, device=device)

        # LOSS
        loss = compute_loss(deformed_template_points, batch_target_points)
        train_loss_current_epoch += loss.detach().cpu().numpy()

        # GRADIENT STEP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_current_epoch /= float(batch_size * (number_of_starmen // batch_size))

    #############
    ### WRITE ###
    #############

    if epoch % print_every_n_iters == 0:
        log += cprint('[Epoch: %d]  Train loss = %.3f \t Test loss = %.3f' % (epoch, train_loss_current_epoch, 0.0))

    if epoch % save_every_n_iters == 0:

        with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
            f.write(log)

        n = 3

        batch_target_points = points[:n]
        batch_target_connec = connectivities[:n]
        batch_latent_momenta = latent_momenta[:n]

        p_t = [torch.zeros(n, latent_dimension_half, device=device).requires_grad_()]
        q_t = [batch_latent_momenta]
        for t in range(number_of_time_points):
            pq = torch.cat((p_t[t], q_t[t]), dim=1)
            h = hamiltonian(pq)
            dh_dp, dh_dq = torch.autograd.grad([h], [p_t[t], q_t[t]], [torch.ones(h.size(), device=device)],
                                               create_graph=True)
            p_t.append(p_t[t] + dt * dh_dq)
            q_t.append(q_t[t] - dt * dh_dp)
        q0_norm = torch.norm(q_t[0], p=2, dim=1)
        v_t = []
        for (p, q) in zip(p_t, q_t):
            qt_norm = torch.norm(q, p=2, dim=1)
            v_t.append(decoder(q * (q0_norm / qt_norm).view(-1, 1).expand(q.size())))
        deformed_template_points = template_points.clone().view(1, -1, dimension).repeat(batch_size, 1, 1)
        deformed_grids = visualization_grid.clone().view(1, -1, dimension).repeat(n, 1, 1)
        for j, vs in enumerate(v_t):
            for p, g, v in zip(deformed_template_points, deformed_grids, vs):
                p += convolutive_interpolation(v.permute(1, 2, 0), p, deformation_grid, deformation_kernel_width)
                g += convolutive_interpolation(v.permute(1, 2, 0), g, deformation_grid, deformation_kernel_width)
                # p += bilinear_interpolation(v.permute(1, 2, 0), p, bounding_box, deformation_grid_size, device=device)
                # g += bilinear_interpolation(v.permute(1, 2, 0), g, bounding_box, deformation_grid_size, device=device)
            plot_registrations(template_points.view(1, -1, dimension).expand(n, -1, dimension), batch_target_points,
                               template_connec.view(1, -1, dimension).expand(n, -1, dimension), batch_target_connec,
                               deformed_template_points,
                               deformed_grids.view(n, visualization_grid_size, visualization_grid_size, dimension),
                               deformation_grid, vs,
                               os.path.join(output_dir, 'train'), 'epoch_%d__j_%d' % (epoch, j))
        deformed_grids = deformed_grids.view(n, visualization_grid_size, visualization_grid_size, dimension)
        plot_registrations(template_points.view(1, -1, dimension).expand(n, -1, dimension), batch_target_points,
                           template_connec.view(1, -1, dimension).expand(n, -1, dimension), batch_target_connec,
                           deformed_template_points, deformed_grids, deformation_grid, v_t[0],
                           os.path.join(output_dir, 'train'), 'epoch_%d' % (epoch))
