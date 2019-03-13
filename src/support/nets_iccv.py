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

### IMPORTS ###
from support.base_iccv import *
from in_out.data_iccv import *


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
        # self.up4 = nn.ConvTranspose3d(4, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.up4 = ConvTranspose3d_Tanh(4, 3, bias=False)
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


class BayesianAtlas(nn.Module):

    def __init__(self,
                 template_points, template_connectivity,
                 bounding_box, latent_dimension, deformation_kernel_width,
                 splatting_grid, deformation_grid, number_of_time_points):
        nn.Module.__init__(self)

        self.deformation_grid = deformation_grid
        self.deformation_grid_size = deformation_grid.size(0)
        self.splatting_grid = splatting_grid
        self.splatting_grid_size = splatting_grid.size(0)

        self.template_connectivity = template_connectivity
        self.latent_dimension = latent_dimension
        self.deformation_kernel_width = deformation_kernel_width
        self.bounding_box = bounding_box
        self.dimension = template_points.size(1)

        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

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
        else:
            raise RuntimeError
        print('>> BayesianAtlas has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def encode(self, x):
        """
        x -> z
        """
        return self.encoder(x)

    def decode(self, z):
        """
        z -> y
        """

        # INIT
        bts = z.size(0)
        ntp = self.number_of_time_points
        dkw = self.deformation_kernel_width
        dim = self.dimension

        # DECODE
        v = self.decoder(z)

        # NORMALIZE
        norm = torch.sqrt(torch.sum(z ** 2, dim=1) / torch.sum(torch.mean(v ** 2, tuple(range(2, dim+2))), dim=1))
        if dim == 2:
            norm = norm.view(bts, 1, 1, 1).expand(v.size())
        elif dim == 3:
            norm = norm.view(bts, 1, 1, 1, 1).expand(v.size())
        v = v * norm

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v, dkw)

        # FLOW
        x = self.template_points.clone().view(1, -1, self.dimension).repeat(bts, 1, 1)
        for t in range(ntp):
            x += batched_bilinear_interpolation(v, x, self.bounding_box, self.deformation_grid_size)

        return x

    def forward(self, z):
        # print('>> Please avoid this forward method.')
        return self.decode(z)

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


    def write_meshes(self, splats, points, connectivities, prefix):

        # INIT
        z, _ = self.encode(splats)
        bts = z.size(0)
        ntp = self.number_of_time_points
        dkw = self.deformation_kernel_width
        dim = self.dimension

        # DECODE
        v = self.decoder(z)

        # NORMALIZE
        norm = torch.sqrt(torch.sum(z ** 2, dim=1) / torch.sum(torch.mean(v ** 2, tuple(range(2, dim + 2))), dim=1))
        if dim == 2:
            norm = norm.view(bts, 1, 1, 1).expand(v.size())
        elif dim == 3:
            norm = norm.view(bts, 1, 1, 1, 1).expand(v.size())
        v = v * norm

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v, dkw)

        # FLOW AND WRITE
        x = self.template_points.clone().view(1, -1, self.dimension).repeat(bts, 1, 1)
        write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                     prefix + '__', '__j_%d' % 0,
                     targets=[(elt_p.detach().cpu().numpy(), elt_c.detach().cpu().numpy())
                              for elt_p, elt_c in zip(points, connectivities)])

        for t in range(ntp):
            x += batched_bilinear_interpolation(v, x, self.bounding_box, self.deformation_grid_size)

            write_meshes(x.detach().cpu().numpy(), self.template_connectivity.detach().cpu().numpy(),
                         prefix + '__', '__t_%d' % (t + 1))
            write_deformations(v, self.deformation_grid.detach().cpu().numpy(),
                               prefix + '__', '__vfield__t_%d' % (t))