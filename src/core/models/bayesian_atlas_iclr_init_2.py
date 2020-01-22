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
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR

### IMPORTS ###
from in_out.datasets_iclr import *
from support.nets_iclr_init_2 import MetamorphicAtlas
from support.base_iclr import *

if __name__ == '__main__':

    ############################
    ##### GLOBAL VARIABLES #####
    ############################

    # MODEL
    # dataset = 'mnist'
    # dataset = 'brains'
    dataset = 'eyes'

    print('>> Run with the [ %s ] dataset.' % dataset)

    ############################
    ######## INITIALIZE ########
    ############################

    if dataset == 'mnist':
        experiment_prefix = '7_atlas__digit_8__ter'
        output_dir = os.path.join(os.path.dirname(__file__), '../../../examples/mnist', experiment_prefix)

        digit = 8

        number_of_images_train = 320
        number_of_images_test = 0

        downsampling_factor = 1
        number_of_time_points = 5

        dimension = 2
        latent_dimension__s = 3
        latent_dimension__a = 2

        kernel_width__s = 5.
        kernel_width__a = 2.5

        lambda_square__s = 0.8 ** 2
        lambda_square__a = 0.4 ** 2
        noise_variance = 0.1 ** 2

        intensities, intensities_test, intensities_template, intensities_mean, intensities_std = load_mnist(
            number_of_images_train, number_of_images_test, digit, random_seed=42)

        # OPTIMIZATION ------------------------------
        number_of_epochs = 10000
        print_every_n_iters = 100
        save_every_n_iters = 1000

        learning_rate = 1e-3
        learning_rate_ratio = 1

        batch_size = min(64, number_of_images_train)

        device = 'cuda'
        # device = 'cpu'
        # -------------------------------------------

    elif dataset == 'brains':
        experiment_prefix = '5_atlas__latent_10_5__SVF'

        output_dir = os.path.join(os.path.dirname(__file__), '../../../examples/brains', experiment_prefix)

        number_of_images_train = 320
        number_of_images_test = 0

        downsampling_factor = 2
        number_of_time_points = 5

        dimension = 2
        latent_dimension__s = 10
        latent_dimension__a = 5

        kernel_width__s = 5
        kernel_width__a = 2.5

        lambda_square__s = 10 ** 2
        lambda_square__a = 10 ** 2
        noise_variance = 0.1 ** 2

        (intensities, intensities_test, intensities_template,
         intensities_mean, intensities_std) = create_cross_sectional_brains_dataset__128(
            number_of_images_train, number_of_images_test, random_seed=42)

        # OPTIMIZATION ------------------------------
        number_of_epochs = 10000
        print_every_n_iters = 100
        save_every_n_iters = 1000

        learning_rate = 1e-3
        learning_rate_ratio = 1

        batch_size = min(64, number_of_images_train)

        device = 'cuda'
        # device = 'cpu'
        # -------------------------------------------

    elif dataset == 'eyes':
        experiment_prefix = '7__init__only_a'

        output_dir = os.path.join(os.path.dirname(__file__), '../../../examples/eyes', experiment_prefix)

        path_to__det_init__v_star = os.path.join(
            output_dir, sorted(fnmatch.filter(os.listdir(output_dir), 'det_init__v_star__*'),
                               key=(lambda x: int(x.split('.')[0].split('__')[2].split('_')[1])))[-1])
        path_to__det_init__n_star = os.path.join(
            output_dir, sorted(fnmatch.filter(os.listdir(output_dir), 'det_init__n_star__*'),
                               key=(lambda x: int(x.split('.')[0].split('__')[2].split('_')[1])))[-1])
        path_to__det_init__template = os.path.join(
            output_dir, sorted(fnmatch.filter(os.listdir(output_dir), 'det_init__template__*'),
                               key=(lambda x: int(x.split('.')[0].split('__')[2].split('_')[1])))[-1])
        print('>> path_to__det_init__v_star = %s' % path_to__det_init__v_star)
        print('>> path_to__det_init__n_star = %s' % path_to__det_init__n_star)
        print('>> path_to__det_init__template = %s' % path_to__det_init__template)
        det_init__v_star = torch.from_numpy(np.load(path_to__det_init__v_star)[()])
        det_init__n_star = torch.from_numpy(np.load(path_to__det_init__n_star)[()])
        det_init__template = torch.from_numpy(np.load(path_to__det_init__template)[()])

        number_of_images_train = int(det_init__v_star.size(0))
        number_of_images_test = 0
        print('>> number_of_images_train = %d' % number_of_images_train)

        downsampling_factor = 2
        number_of_time_points = 5

        dimension = 2
        latent_dimension__s = 2
        latent_dimension__a = 2

        kernel_width__s = 20
        kernel_width__a = 2.5

        lambda_square__s = 1 ** 2
        lambda_square__a = 0.5 ** 2
        noise_variance = 0.05 ** 2

        intensities, intensities_test, intensities_template, intensities_mean, intensities_std = load_eyes(
            number_of_images_train, number_of_images_test, random_seed=42)

        # OPTIMIZATION ------------------------------
        number_of_epochs = 20000
        print_every_n_iters = 100
        save_every_n_iters = 500

        learning_rate = 1e-3
        learning_rate_ratio = 1

        batch_size = min(64, number_of_images_train)

        device = 'cuda'
        # device = 'cpu'
        # -------------------------------------------

    else:
        raise RuntimeError

    assert number_of_time_points > 1

    log = ''
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if (not torch.cuda.is_available()) and 'cuda' in device:
        device = 'cpu'
        print('>> CUDA is not available. Overridding with device = "cpu".')

    ##################
    ###### MAIN ######
    ##################

    model = MetamorphicAtlas(
        det_init__template, number_of_time_points, downsampling_factor,
        latent_dimension__s, latent_dimension__a,
        kernel_width__s, kernel_width__a,
        initial_lambda_square__s=lambda_square__s, initial_lambda_square__a=lambda_square__a)

    noise_dimension = model.grid_size ** model.dimension

    if 'cuda' in device:
        model.cuda()
        model.template_intensities = model.template_intensities.cuda()
        det_init__v_star = det_init__v_star.cuda()
        det_init__n_star = det_init__n_star.cuda()
        intensities = intensities.cuda()
        intensities_test = intensities_test.cuda()

    elif device == 'cpu':
        omp_num_threads = 36
        print('>> OMP_NUM_THREADS will be set to ' + str(omp_num_threads))
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
        torch.set_num_threads(omp_num_threads)

    # s = torch.zeros((number_of_images_train, latent_dimension__s)).float()
    # a = torch.zeros((number_of_images_train, latent_dimension__a)).float()
    s = torch.randn((number_of_images_train, latent_dimension__s)).float() * lambda_square__s
    a = torch.randn((number_of_images_train, latent_dimension__a)).float() * lambda_square__a

    if 'cuda' in device:
        s = s.cuda()
        a = a.cuda()

    model.s = torch.nn.Parameter(s)
    model.a = torch.nn.Parameter(a)

    det_init__v = batched_vector_smoothing(det_init__v_star, kernel_width__s, scaled=False).detach()
    det_init__n = batched_scalar_smoothing(det_init__n_star, kernel_width__a, scaled=False).detach()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = LBFGS(model.parameters(), lr=learning_rate)

    for epoch in range(number_of_epochs + 1):
        # scheduler.step()

        #############
        ### TRAIN ###
        #############

        indexes = np.random.permutation(number_of_images_train)
        for k in range(number_of_images_train // batch_size):  # drops the last batch

            batch_target__v = det_init__v[indexes[k * batch_size:(k + 1) * batch_size]]
            batch_target__n = det_init__n[indexes[k * batch_size:(k + 1) * batch_size]]
            batch_latent__s = model.s[indexes[k * batch_size:(k + 1) * batch_size]]
            batch_latent__a = model.a[indexes[k * batch_size:(k + 1) * batch_size]]

            # ENCODE, SAMPLE AND DECODE
            v, n = model.decode_and_normalize(batch_latent__s, batch_latent__a)

            # LOSS
            attachment_loss__v = torch.mean((v - batch_target__v) ** 2)
            attachment_loss__n = torch.mean((n - batch_target__n) ** 2)
            train_attachment_loss__v = attachment_loss__v.detach().cpu().numpy()
            train_attachment_loss__n = attachment_loss__n.detach().cpu().numpy()

            # total_loss = attachment_loss__v + attachment_loss__n
            total_loss = attachment_loss__n
            train_total_loss = total_loss.detach().cpu().numpy()

            # GRADIENT STEP
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ##############
            ### UPDATE ###
            ##############
            #
            # ss_s_var /= float(batch_size)
            # ss_a_var /= float(batch_size)
            #
            # if epoch > int(number_of_epochs * 0.5):
            #     # if epoch > 0:
            #     noise_variance *= float(train_attachment_loss / float(noise_dimension))
            #     lambda_square__s = float(ss_s_var)
            #     lambda_square__a = float(ss_a_var)
            #
            # # if (500 <= epoch <= 1000 and not epoch % 100) or (1000 <= epoch <= 5000 and not epoch % 500):
            # if (int(0.25 * number_of_epochs) <= epoch <= int(0.5 * number_of_epochs) and not epoch % int(0.01 * number_of_epochs)):
            # if (epoch <= int(0.5 * number_of_epochs) and not epoch % int(0.01 * number_of_epochs)):
            #     batch_latent__s = torch.zeros((1, latent_dimension__s)).float().cuda()
            #     batch_latent__a = torch.zeros((1, latent_dimension__a)).float().cuda()
            #     transformed_template = model(batch_latent__s, batch_latent__a)
            #     model.template_intensities.data = transformed_template[0]

        ############
        ### TEST ###
        ############

        test_attachment_loss = 0.
        test_kullback_regularity_loss__s = 0.
        test_kullback_regularity_loss__a = 0.
        test_total_loss = 0.

        if number_of_images_test > 1 and epoch % print_every_n_iters == 0:
            batch_target_intensities = intensities_test

            # ENCODE, SAMPLE AND DECODE
            means__s, log_variances__s, means__a, log_variances__a = model.encode(batch_target_intensities)
            stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

            ss_s_mean = torch.mean(means__s).cpu().detach().numpy()
            ss_s_var = torch.mean(means__s ** 2 + stds__s ** 2).cpu().detach().numpy()
            ss_a_mean = torch.mean(means__a).cpu().detach().numpy()
            ss_a_var = torch.mean(means__a ** 2 + stds__a ** 2).cpu().detach().numpy()

            batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
            batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
            transformed_template = model(batch_latent__s, batch_latent__a)

            # LOSS
            attachment_loss = torch.sum((transformed_template - batch_target_intensities) ** 2) / noise_variance
            test_attachment_loss = attachment_loss.detach().cpu().numpy()

            kullback_regularity_loss__s = torch.sum(
                (means__s.pow(2) + log_variances__s.exp()) / lambda_square__s - log_variances__s + np.log(
                    lambda_square__s))
            test_kullback_regularity_loss__s = kullback_regularity_loss__s.detach().cpu().numpy()

            kullback_regularity_loss__a = torch.sum(
                (means__a.pow(2) + log_variances__a.exp()) / lambda_square__a - log_variances__a + np.log(
                    lambda_square__a))
            test_kullback_regularity_loss__a = kullback_regularity_loss__a.detach().cpu().numpy()

            total_loss = attachment_loss + kullback_regularity_loss__s + kullback_regularity_loss__a
            test_total_loss = total_loss.detach().cpu().numpy()

            test_attachment_loss /= float(batch_size)
            test_kullback_regularity_loss__s /= float(batch_size)
            test_kullback_regularity_loss__a /= float(batch_size)
            test_total_loss /= float(batch_size)

        ################
        ### TEMPLATE ###
        ################

        # template_intensities = model.template_intensities.view((1,) + model.template_intensities.size())
        # template_latent_s, _, template_latent_a, _ = model.encode(template_intensities)
        # template_latent_s_norm = float(torch.norm(template_latent_s[0], p=2).detach().cpu().numpy())
        # template_latent_a_norm = float(torch.norm(template_latent_a[0], p=2).detach().cpu().numpy())

        #############
        ### WRITE ###
        #############

        if epoch % print_every_n_iters == 0 or epoch == number_of_epochs:
            log += cprint(
                '\n[Epoch: %d] Learning rate = %.2E'
                '\nTrain loss = %.3f (attachment__v = %.3f ; attachment__n = %.3f' %
                (epoch, list(optimizer.param_groups)[0]['lr'],
                 train_total_loss, train_attachment_loss__v, train_attachment_loss__n))

        if epoch % save_every_n_iters == 0 or epoch == number_of_epochs:
            with open(os.path.join(output_dir, 'dec_init__log.txt'), 'w') as f:
                f.write(log)

            torch.save(model.state_dict(), os.path.join(output_dir, 'dec_init__model__epoch_%d.pth' % epoch))

            n = 5
            # model.write(intensities[:n], os.path.join(output_dir, 'train__epoch_%d' % epoch),
            #             intensities_mean, intensities_std)
            model.write_decode_and_normalize(intensities[:n], det_init__n[:n], model.s[:n], model.a[:n],
                                             os.path.join(output_dir, 'dec_init__train__epoch_%d' % epoch),
                                             intensities_mean, intensities_std)
