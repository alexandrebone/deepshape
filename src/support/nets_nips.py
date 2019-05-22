from support.base_nips import *
from in_out.data_nips import *


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

    def __init__(self, in_grid_size, latent_dimension__s, latent_dimension__a, init_var__s=1.0, init_var__a=1.0):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -4)
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a
        self.init_var__s = init_var__s
        self.init_var__a = init_var__a

        self.down1 = Conv2d_Tanh(1, 4)
        self.down2 = Conv2d_Tanh(4, 8)
        self.down3 = Conv2d_Tanh(8, 16)
        self.down4 = Conv2d_Tanh(16, 32)
        self.linear1__s = nn.Linear(32 * n * n, latent_dimension__s)
        self.linear2__s = nn.Linear(32 * n * n, latent_dimension__s)
        self.linear1__a = nn.Linear(32 * n * n, latent_dimension__a)
        self.linear2__a = nn.Linear(32 * n * n, latent_dimension__a)
        print('>> Encoder2d has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        mean__s = self.linear1__s(x.view(x.size(0), -1)).view(x.size(0), -1)
        logv__s = self.linear2__s(x.view(x.size(0), -1)).view(x.size(0), -1) + np.log(self.init_var__s)
        mean__a = self.linear1__a(x.view(x.size(0), -1)).view(x.size(0), -1)
        logv__a = self.linear2__a(x.view(x.size(0), -1)).view(x.size(0), -1) + np.log(self.init_var__a)
        return mean__s, logv__s, mean__a, logv__a


class Encoder3d(nn.Module):
    """
    in: in_grid_size * in_grid_size * in_grid_size * 3
    out: latent_dimension
    """

    def __init__(self, in_grid_size, latent_dimension, init_var=1.):
        nn.Module.__init__(self)
        n = int(in_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.init_var = init_var
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
        s = self.linear2(x.view(x.size(0), -1)).view(x.size(0), -1) + np.log(self.init_var)
        return m, s


class DeepDecoder2d(nn.Module):
    """
    in: latent_dimension
    out: out_grid_size * out_grid_size * 3
    """

    def __init__(self, latent_dimension, out_channels, out_grid_size):
        nn.Module.__init__(self)
        self.inner_grid_size = int(out_grid_size * 2 ** -4)
        self.latent_dimension = latent_dimension
        self.linear1 = Linear_Tanh(latent_dimension, 32 * self.inner_grid_size ** 2, bias=False)
        self.linear2 = Linear_Tanh(32 * self.inner_grid_size ** 2, 32 * self.inner_grid_size ** 2, bias=False)
        self.linear3 = Linear_Tanh(32 * self.inner_grid_size ** 2, 32 * self.inner_grid_size ** 2, bias=False)
        self.up1 = ConvTranspose2d_Tanh(32, 16, bias=False)
        self.up2 = ConvTranspose2d_Tanh(16, 8, bias=False)
        self.up3 = ConvTranspose2d_Tanh(8, 4, bias=False)
        self.up4 = ConvTranspose2d_Tanh(4, out_channels, bias=False)
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

        # self.normalizer = 1.
        # z = torch.from_numpy(np.array(1.)).float()
        # m = self.forward(z)
        # self.normalizer = torch.sqrt(torch.sum(v *, dim=tuple(range(1, dim+2))))

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


class MetamorphicAtlas(nn.Module):

    def __init__(self, template_intensities, number_of_time_points, downsampling_factor,
                 latent_dimension__s, latent_dimension__a,
                 kernel_width__s, kernel_width__a,
                 initial_lambda_square__s=1., initial_lambda_square__a=1.):
        nn.Module.__init__(self)

        self.decode_count = 0

        self.dimension = len(template_intensities.size()) - 1
        self.latent_dimension__s = latent_dimension__s
        self.latent_dimension__a = latent_dimension__a

        self.downsampling_factor = downsampling_factor
        self.grid_size = template_intensities.size(1)
        self.downsampled_grid_size = self.grid_size // self.downsampling_factor
        assert self.grid_size == template_intensities.size(2)

        self.number_of_time_points = number_of_time_points
        self.dt = 1. / float(number_of_time_points - 1)

        self.kernel_width__s = kernel_width__s
        self.kernel_width__a = kernel_width__a

        # self.template_intensities = template_intensities
        self.template_intensities = nn.Parameter(template_intensities)
        print('>> Template intensities are %d ** %d = %d parameters' % (
            template_intensities.size(1), self.dimension, template_intensities.view(-1).size(0)))

        if self.dimension == 2:
            self.encoder = Encoder2d(self.grid_size, latent_dimension__s, latent_dimension__a,
                                     init_var__s=(initial_lambda_square__s / np.sqrt(latent_dimension__s)),
                                     init_var__a=(initial_lambda_square__a / np.sqrt(latent_dimension__a)))
            self.decoder__s = DeepDecoder2d(latent_dimension__s, 2, self.downsampled_grid_size)
            self.decoder__a = DeepDecoder2d(latent_dimension__a, 1, self.grid_size)

        elif self.dimension == 3:
            assert False
            # self.encoder = Encoder3d(self.splatting_grid_size, self.latent_dimension,
            #                          init_var=initial_lambda_square / np.sqrt(latent_dimension))
            # self.decoder = DeepDecoder3d(self.latent_dimension, self.deformation_grid_size)

        else:
            raise RuntimeError

        print('>> BayesianAtlas has {} parameters'.format(sum([len(elt.view(-1)) for elt in self.parameters()])))

    def encode(self, x):
        """
        x -> z
        """
        return self.encoder(x)

    def decode(self, s, a):
        """
        z -> y
        """

        # INIT
        bts = s.size(0)
        assert bts == a.size(0)
        ntp = self.number_of_time_points
        kws = self.kernel_width__s
        kwa = self.kernel_width__a
        dim = self.dimension
        gs = self.grid_size
        dgs = self.downsampled_grid_size
        dsf = self.downsampling_factor

        # DECODE
        # s = z[:, :self.latent_dimension__s]
        # a = z[:, -self.latent_dimension__a:]

        v_star = self.decoder__s(s)
        n_star = self.decoder__a(a)

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v_star, kws, scaled=False)
        n = batched_scalar_smoothing(n_star, kwa, scaled=False)

        # NORMALIZE
        s_norm_squared = torch.sum(s ** 2, dim=1)
        a_norm_squared = torch.sum(a ** 2, dim=1)
        v_norm_squared = torch.sum(v * v_star, dim=tuple(range(1, dim + 2)))
        n_norm_squared = torch.sum(n * n_star, dim=tuple(range(1, dim + 2)))
        normalizer__s = torch.where(s_norm_squared > 1e-10,
                                    torch.sqrt(s_norm_squared / v_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(s.type())))
        normalizer__a = torch.where(a_norm_squared > 1e-10,
                                    torch.sqrt(a_norm_squared / n_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(a.type())))

        if dim == 2:
            normalizer__s = normalizer__s.view(bts, 1, 1, 1).expand(v.size())
            normalizer__a = normalizer__a.view(bts, 1, 1, 1).expand(n.size())
        elif dim == 3:
            assert False
            # normalizer = normalizer.view(bts, 1, 1, 1, 1).expand(v.size())
        v = v * normalizer__s
        n = n * normalizer__a

        if self.decode_count < 10:
            print('>> normalizer shape  = %.3E ; max(abs(v)) = %.3E' %
                  (normalizer__s.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(v.detach().cpu().numpy()))))
            print('>> normalizer appea  = %.3E ; max(abs(n)) = %.3E' %
                  (normalizer__a.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(n.detach().cpu().numpy()))))
            self.decode_count += 1

        # FLOW
        grid = torch.stack(torch.meshgrid(
            [torch.linspace(0.0, gs - 1.0, dgs),
             torch.linspace(0.0, gs - 1.0, dgs)])).type(str(s.type())).view(1, 2, dgs, dgs).repeat(bts, 1, 1, 1)

        x = grid.clone() + v / float(2 ** ntp)
        for t in range(ntp):
            x += batched_vector_interpolation(x - grid, x, dsf)

        # INTERPOLATE
        intensities = batched_scalar_interpolation(self.template_intensities + n, x)

        return intensities

    def forward(self, s, a):
        # print('>> Please avoid this forward method.')
        return self.decode(s, a)

    def tamper_template_gradient(self, lr, print_info=False):
        pass

        # template_gradient = self.template_points.grad.detach()
        #
        # norms = torch.sqrt(torch.sum(template_gradient ** 2, dim=1))
        # std_robust = 1.4826 * torch.median(torch.abs(norms - torch.median(norms)))
        #
        # cutoff_factor = 2.5
        # threshold = torch.median(norms) + cutoff_factor * std_robust
        # norms = norms.contiguous().view(-1, 1).expand(template_gradient.size())
        # cutoff_template_gradient = torch.where(
        #     norms > threshold,
        #     cutoff_factor * std_robust * template_gradient / norms,
        #     template_gradient)
        #
        # tampered_template_gradient = lr * convolve(
        #     self.template_points.detach(), self.template_points.detach(), cutoff_template_gradient, torch.sqrt(1/gamma))
        # # tampered_template_gradient = (lr * kernel(gamma, cutoff_template_gradient, self.template_points.detach(),
        # #                                           self.template_points.grad.detach())).detach()
        #
        # # write_gradient('grad.vtk', template_gradient, self.template_points.detach())
        # # write_gradient('grad_normalized.vtk', template_gradient / norms, self.template_points.detach())
        # # write_gradient('grad_cutoff.vtk', cutoff_template_gradient, self.template_points.detach())
        # # write_gradient('grad_smoothed.vtk', tampered_template_gradient, self.template_points.detach())
        #
        # if zero:
        #     self.template_points.grad = tampered_template_gradient * 0.0
        # else:
        #     self.template_points.grad = tampered_template_gradient
        #
        # if print_info:
        #     print('tampered template gradient max absolute value = %.3f' %
        #           torch.max(torch.abs(tampered_template_gradient)))

    # def update_template(self, kernel, gamma, lr):
    #     update = - lr * kernel(
    #         gamma, self.template_points.detach(), self.template_points.detach(), self.template_points.grad.detach())
    #     self.template_points = self.template_points.detach() + update
    #     self.template_points.requires_grad_()
    #     print('template update min = %.3f ; max = %.3f' % (torch.min(update), torch.max(update)))

    def write(self, observations, prefix, mean, std):
        s, _, a, _ = self.encode(observations)

        # INIT
        bts = s.size(0)
        assert bts == a.size(0)
        ntp = self.number_of_time_points
        kws = self.kernel_width__s
        kwa = self.kernel_width__a
        dim = self.dimension
        gs = self.grid_size
        dgs = self.downsampled_grid_size
        dsf = self.downsampling_factor

        # DECODE
        # s = z[:, :self.latent_dimension__s]
        # a = z[:, -self.latent_dimension__a:]

        v_star = self.decoder__s(s)
        n_star = self.decoder__a(a)

        # GAUSSIAN SMOOTHING
        v = batched_vector_smoothing(v_star, kws, scaled=False)
        n = batched_scalar_smoothing(n_star, kwa, scaled=False)

        # NORMALIZE
        s_norm_squared = torch.sum(s ** 2, dim=1)
        a_norm_squared = torch.sum(a ** 2, dim=1)
        v_norm_squared = torch.sum(v * v_star, dim=tuple(range(1, dim + 2)))
        n_norm_squared = torch.sum(n * n_star, dim=tuple(range(1, dim + 2)))
        normalizer__s = torch.where(s_norm_squared > 1e-10,
                                    torch.sqrt(s_norm_squared / v_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(s.type())))
        normalizer__a = torch.where(a_norm_squared > 1e-10,
                                    torch.sqrt(a_norm_squared / n_norm_squared),
                                    torch.from_numpy(np.array(0.0)).float().type(str(a.type())))

        if dim == 2:
            normalizer__s = normalizer__s.view(bts, 1, 1, 1).expand(v.size())
            normalizer__a = normalizer__a.view(bts, 1, 1, 1).expand(n.size())
        elif dim == 3:
            assert False
            # normalizer = normalizer.view(bts, 1, 1, 1, 1).expand(v.size())
        v = v * normalizer__s
        n = n * normalizer__a

        if self.decode_count < 10:
            print('>> normalizer shape  = %.3E ; max(abs(v)) = %.3E' %
                  (normalizer__s.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(v.detach().cpu().numpy()))))
            print('>> normalizer appea  = %.3E ; max(abs(n)) = %.3E' %
                  (normalizer__a.detach().cpu().numpy().reshape(-1)[0], np.max(np.abs(n.detach().cpu().numpy()))))
            self.decode_count += 1

        # FLOW
        grid = torch.stack(torch.meshgrid(
            [torch.linspace(0.0, gs - 1.0, dgs),
             torch.linspace(0.0, gs - 1.0, dgs)])).type(str(s.type())).view(1, 2, dgs, dgs).repeat(bts, 1, 1, 1)

        x = grid.clone() + v / float(2 ** ntp)
        for t in range(ntp):
            x += batched_vector_interpolation(x - grid, x, dsf)

        # INTERPOLATE
        intensities = batched_scalar_interpolation(self.template_intensities + n, x)

        # WRITE
        images = []
        for i in range(bts):
            template = mean + std * self.template_intensities.clone()
            appearance = mean + std * (self.template_intensities + n[i])
            shape = mean + std * batched_scalar_interpolation(self.template_intensities, x[i].unsqueeze(0))[0]
            metamorphosis = mean + std * intensities[i]
            target = mean + std * observations[i]

            images_i = []
            images_i.append(template)
            images_i.append(appearance)
            images_i.append(shape)
            images_i.append(metamorphosis)
            images_i.append(target)
            images += images_i

            write_image(prefix + '__subject_%d__0__template' % i, template.detach().cpu().numpy())
            write_image(prefix + '__subject_%d__1__appearance' % i, appearance.detach().cpu().numpy())
            write_image(prefix + '__subject_%d__2__shape' % i, shape.detach().cpu().numpy())
            write_image(prefix + '__subject_%d__3__metamorphosis' % i, metamorphosis.detach().cpu().numpy())
            write_image(prefix + '__subject_%d__4__target' % i, target.detach().cpu().numpy())

            save_image(torch.cat(images_i).unsqueeze(1), prefix + '__subject_%d__5__reconstructions.pdf' % i,
                       nrow=5, normalize=True, range=(0., 255.))

        save_image(torch.cat(images).unsqueeze(1), prefix + '__reconstructions.pdf',
                   nrow=5, normalize=True, range=(0., 255.))
