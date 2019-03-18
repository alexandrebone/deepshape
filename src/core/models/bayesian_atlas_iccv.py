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
from in_out.datasets_iccv import *
from support.nets_iccv import BayesianAtlas
from support.base_iccv import *



if __name__ == '__main__':

    ############################
    ##### GLOBAL VARIABLES #####
    ############################

    # experiment_prefix = '39_bayesian_atlas_fourier__alpha_0.5'
    # experiment_prefix = '4_bayesian_atlas_fourier__latent_space_2__lambda_1e-4'
    # experiment_prefix = '1_bayesian_atlas_fourier__latent_space_2__lambda_1e-4'
    # experiment_prefix = '47_bayesian_atlas_fourier__latent_10__current_5__lambda_1e-6__grid_16__dynamic__all_5__new_sobolev'

    # MODEL

    # dataset = 'hippocampi'
    # dataset = 'circles'
    # dataset = 'ellipsoids'
    dataset = 'starmen'
    # dataset = 'leaves'
    # dataset = 'squares'

    print(dataset)

    number_of_meshes_train = 16
    number_of_meshes_test = 0

    splatting_grid_size = 16
    deformation_grid_size = 16
    visualization_grid_size = 32

    number_of_time_points = 5

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

    learning_rate = 5e-2
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
        experiment_prefix = '7_new_scaling_and_new_init'
        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data'))

        initialize_template = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/starmen/data/SimulatedData__EstimatedParameters__Template_starman__tp_22__age_70.00.vtk'))
        # initialize_encoder = os.path.join(path_to_meshes, 'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')
        # initialize_decoder = os.path.join(path_to_meshes, 'latent_positions.txt')

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/hippocampi/output__49_bayesian_atlas_fourier__latent_10__162_subjects/epoch_0__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_encoder__epoch_9000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        number_of_meshes_train = 1
        number_of_meshes_test = 0

        splatting_grid_size = 16
        deformation_grid_size = 16
        visualization_grid_size = 16

        dimension = 2
        latent_dimension = 1
        number_of_time_points = 5

        deformation_kernel_width = 3.
        splatting_kernel_width = 1.0

        lambda_square = 0.1 ** 2
        noise_variance = 0.01 ** 2
        # noise_variance = 0.1 ** 2

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
        number_of_epochs = 5000
        number_of_epochs_for_init = 1000
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 200
        save_every_n_iters = 1000

        learning_rate = 1e-3
        learning_rate_decay = 0.95
        learning_rate_ratio = 1.

        batch_size = 1

        device = 'cuda:01'
        # device = 'cpu'
        # -------------------------------------------

    elif dataset == 'ellipsoids':
        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/ellipsoids/data'))

        dimension = 3
        splatting_kernel_width = 0.15

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
        number_of_epochs = 1000
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 100
        save_every_n_iters = 250

        learning_rate = 5e-4
        learning_rate_decay = 1.
        learning_rate_ratio = 5e-5

        batch_size = 32

        device = 'cuda:01'
        # device = 'cpu'
        # ----------------------------

    elif dataset == 'hippocampi':
        experiment_prefix = '1_first_attempt'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/hippocampi/data'))

        initialize_template = os.path.join(path_to_meshes,
                                           'PrincipalGeodesicAnalysis__EstimatedParameters__Template_hippocampus.vtk')
        # initialize_encoder = os.path.join(path_to_meshes,
        #                                   'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')
        # initialize_decoder = os.path.join(path_to_meshes,
        #                                   'PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt')

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/hippocampi/output__49_bayesian_atlas_fourier__latent_10__162_subjects/epoch_0__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_encoder__epoch_9000__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        number_of_meshes_train = 1
        number_of_meshes_test = 0

        splatting_grid_size = 16
        deformation_grid_size = 16
        visualization_grid_size = 16

        number_of_time_points = 5

        dimension = 3
        latent_dimension = 10

        deformation_kernel_width = 3.
        splatting_kernel_width = 5.
        varifold_kernel_width = 5.

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

        (points, connectivities, centers, normals, norms, splats,
         points_test, connectivities_test, centers_test, normals_test, norms_test, splats_test) = \
            create_cross_sectional_hippocampi_dataset(
                path_to_meshes,
                number_of_meshes_train, number_of_meshes_test,
                splatting_grid, dimension, gkernel, gamma_splatting, random_seed=42)

        # OPTIMIZATION --------------
        number_of_epochs = 2000
        number_of_epochs_for_init = 25000
        number_of_epochs_for_warm_up = 0

        print_every_n_iters = 1
        save_every_n_iters = 500

        learning_rate = 1e-2
        learning_rate_decay = 1.
        learning_rate_ratio = 5e-5

        batch_size = 1

        device = 'cuda:01'
        # device = 'cpu'
        # ----------------------------

    elif dataset == 'squares':
        experiment_prefix = '3_first_attempt'

        path_to_meshes = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/data'))

        initialize_template = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                            '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__Template_square.vtk'))
        initialize_encoder = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))
        initialize_decoder = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../examples/squares/data/PrincipalGeodesicAnalysis__EstimatedParameters__LatentPositions.txt'))

        # initial_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__3_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init__except_template/epoch_11000__model.pth'))
        # initial_encoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__1_first_attempt/init_encoder__epoch_500__model.pth'))
        # initial_decoder_state = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../examples/squares/output__2_bayesian_atlas_fourier__latent_space_2__64_subjects__lambda_10__alpha_0.5__init/init_decoder__epoch_4000__model.pth'))

        # number_of_meshes_train = 224
        number_of_meshes_train = 32
        # number_of_meshes_train = 441

        # number_of_meshes_test = 192
        number_of_meshes_test = 0

        splatting_grid_size = 16
        deformation_grid_size = 16
        visualization_grid_size = 16

        number_of_time_points = 6

        dimension = 2
        deformation_kernel_width = 5.
        splatting_kernel_width = 0.2

        noise_variance = 0.01 ** 2
        lambda_square = .1 ** 2

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
        number_of_epochs_for_init = 500
        number_of_epochs_for_warm_up = 0
        print_every_n_iters = 1000
        save_every_n_iters = 1000

        learning_rate = 1e-3
        learning_rate_decay = 1.
        learning_rate_ratio = 1e-1

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
                              bounding_box, latent_dimension, deformation_kernel_width,
                              splatting_grid, deformation_grid, number_of_time_points, lambda_square)
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
                              bounding_box, latent_dimension, deformation_kernel_width,
                              splatting_grid, deformation_grid, number_of_time_points, lambda_square)
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
        splats_test = splats_test.cuda()

        gamma_splatting = gamma_splatting.cuda()

        if dataset == 'hippocampi':
            gamma_varifold = gamma_varifold.cuda()

            centers = [elt.cuda() for elt in centers]
            normals = [elt.cuda() for elt in normals]
            norms = [elt.cuda() for elt in norms]

            centers_test = [elt.cuda() for elt in centers_test]
            normals_test = [elt.cuda() for elt in normals_test]
            norms_test = [elt.cuda() for elt in norms_test]

            points = [elt.cuda() for elt in points]
            connectivities = [elt.cuda() for elt in connectivities]

            points_test = [elt.cuda() for elt in points_test]
            connectivities_test = [elt.cuda() for elt in connectivities_test]

        else:
            points = points.cuda()
            connectivities = connectivities.cuda()

            points_test = points_test.cuda()
            connectivities_test = connectivities_test.cuda()

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

            indexes = np.random.permutation(number_of_meshes_train)
            for k in range(number_of_meshes_train // batch_size):  # drops the last batch
                batch_target_splats = splats[indexes[k * batch_size:(k + 1) * batch_size]]
                batch_latent_momenta__init = latent_momenta__init[indexes[k * batch_size:(k + 1) * batch_size]]

                if dimension == 2:
                    batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
                elif dimension == 3:
                    batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
                else:
                    raise RuntimeError

                # ENCODE AND SAMPLE
                means, log_variances = model.encode(batch_target_splats)
                batch_latent_momenta = means + torch.zeros_like(means).normal_() * torch.exp(0.5 * log_variances)

                # LOSS
                attachment_loss = torch.sum((batch_latent_momenta - batch_latent_momenta__init) ** 2) / noise_variance
                np_attachment_loss += attachment_loss.detach().cpu().numpy()

                kullback_regularity_loss = torch.sum(
                    (means.pow(2) + log_variances.exp()) / lambda_square - log_variances + np.log(lambda_square))
                np_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

                total_loss = attachment_loss + kullback_regularity_loss
                np_total_loss += total_loss.detach().cpu().numpy()

                # GRADIENT STEP
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            np_attachment_loss /= float(batch_size * (number_of_meshes_train // batch_size))
            np_kullback_regularity_loss /= float(batch_size * (number_of_meshes_train // batch_size))
            np_total_loss /= float(batch_size * (number_of_meshes_train // batch_size))

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
        latent_momenta__init = torch.from_numpy(np.loadtxt(initialize_decoder)).float().view(-1, latent_dimension)
        if 'cuda' in device:
            latent_momenta__init = latent_momenta__init.cuda()
        optimizer = Adam(model.decoder.parameters(), lr=learning_rate)

        for epoch in range(number_of_epochs_for_init + 1):

            np_attachment_loss = 0.0
            np_total_loss = 0.0

            indexes = np.random.permutation(number_of_meshes_train)
            for k in range(number_of_meshes_train // batch_size):  # drops the last batch
                batch_latent_momenta = latent_momenta__init[indexes[k * batch_size:(k + 1) * batch_size]]

                # DECODE
                deformed_template_points = model(batch_latent_momenta)

                # LOSS
                if dataset == 'hippocampi':
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
                    batch_target_points = points[indexes[k * batch_size:(k + 1) * batch_size]]
                    attachment_loss = torch.sum((deformed_template_points - batch_target_points) ** 2) / noise_variance

                np_attachment_loss += attachment_loss.detach().cpu().numpy()

                total_loss = attachment_loss
                np_total_loss += total_loss.detach().cpu().numpy()

                # GRADIENT STEP
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            np_attachment_loss /= float(batch_size * (number_of_meshes_train // batch_size))
            np_total_loss /= float(batch_size * (number_of_meshes_train // batch_size))

            if epoch % print_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                log += cprint(
                    '\n[Epoch: %d] Learning rate = %.2E'
                    '\nTrain loss = %.3f (attachment = %.3f)' %
                    (epoch, list(optimizer.param_groups)[0]['lr'],
                     np_total_loss, np_attachment_loss))

            if epoch % save_every_n_iters == 0 or epoch == number_of_epochs_for_init:
                with open(os.path.join(output_dir, 'init_decoder__log.txt'), 'w') as f:
                    f.write(log)

                torch.save(model.decoder.state_dict(),
                           os.path.join(output_dir, 'init_decoder__epoch_%d__model.pth' % epoch))

                n = 3
                if dataset == 'starmen' and False:
                    model.write_starmen(splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                                        visualization_grid,
                                        os.path.join(output_dir, 'init_decoder__epoch_%d__train' % epoch))

                if dataset in ['ellipsoids', 'hippocampi']:
                    model.write_meshes(
                        splats[:n].permute(0, 4, 1, 2, 3), points[:n], connectivities[:n],
                        os.path.join(output_dir, 'init_decoder__epoch_%d__train' % epoch))

                if dataset in ['starmen', 'circles', 'leaves', 'squares']:
                    model.write_meshes(
                        splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                        os.path.join(output_dir, 'init_decoder__epoch_%d__train' % epoch))

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
    # optimizer = LBFGS(model.parameters(), lr=learning_rate)

    for epoch in range(number_of_epochs + 1):
        # scheduler.step()

        #############
        ### TRAIN ###
        #############

        train_attachment_loss = 0.
        train_kullback_regularity_loss = 0.
        train_total_loss = 0.
        ss_z_mean = 0.
        ss_z_var = 0.

        indexes = np.random.permutation(number_of_meshes_train)
        for k in range(number_of_meshes_train // batch_size):  # drops the last batch
            batch_target_splats = splats[indexes[k * batch_size:(k + 1) * batch_size]]

            if dimension == 2:
                batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
            elif dimension == 3:
                batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
            else:
                raise RuntimeError

            # ENCODE, SAMPLE AND DECODE
            means, log_variances = model.encode(batch_target_splats)
            stds = torch.exp(0.5 * log_variances)

            ss_z_mean += torch.mean(means).cpu().detach().numpy()
            ss_z_var += torch.mean(means ** 2 + stds ** 2).cpu().detach().numpy()

            if epoch < number_of_epochs_for_warm_up + 1:
                batch_latent_momenta = means
                deformed_template_points = model(batch_latent_momenta)
            else:
                batch_latent_momenta = means + torch.zeros_like(means).normal_() * stds
                deformed_template_points = model(batch_latent_momenta)

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

                # # Varifold
                # attachment_loss = 0.0
                # for p1, c2, n2, norm2 in zip(
                #         deformed_template_points, batch_target_centers, batch_target_normals, batch_target_norms):
                #     c1, n1 = compute_centers_and_normals(p1, model.template_connectivity)
                #     a1 = torch.norm(n1, 2, 1).view(-1, 1)
                #     a2 = torch.norm(n2, 2, 1).view(-1, 1)
                #     u1 = n1 / a1
                #     u2 = n2 / a2
                #     attachment_loss += (
                #             torch.sum(a1 * vkernel(gamma_varifold, c1, c1, u1, u1, a1)) +
                #             torch.sum(a2 * vkernel(gamma_varifold, c2, c2, u2, u2, a2)) - 2 *
                #             torch.sum(a1 * vkernel(gamma_varifold, c1, c2, u1, u2, a2)))
                # attachment_loss /= noise_variance

            else:
                raise RuntimeError

            train_attachment_loss += attachment_loss.detach().cpu().numpy()

            kullback_regularity_loss = torch.sum(
                (means.pow(2) + log_variances.exp()) / lambda_square - log_variances + np.log(lambda_square))
            train_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

            total_loss = attachment_loss + kullback_regularity_loss
            # total_loss = attachment_loss
            train_total_loss += total_loss.detach().cpu().numpy()

            # GRADIENT STEP
            optimizer.zero_grad()
            total_loss.backward()
            model.tamper_template_gradient(gkernel, gamma_splatting, learning_rate_ratio, epoch < 20)
            optimizer.step()
            # model.update_template(gkernel, gamma_splatting, learning_rate_ratio * list(optimizer.param_groups)[0]['lr'])

        train_attachment_loss /= float(batch_size * (number_of_meshes_train // batch_size))
        train_kullback_regularity_loss /= float(batch_size * (number_of_meshes_train // batch_size))
        train_total_loss /= float(batch_size * (number_of_meshes_train // batch_size))
        ss_z_mean /= float(batch_size * (number_of_meshes_train // batch_size))
        ss_z_var /= float(batch_size * (number_of_meshes_train // batch_size))

        ############
        ### TEST ###
        ############

        test_attachment_loss = 0.
        test_kullback_regularity_loss = 0.
        test_total_loss = 0.

        if number_of_meshes_test > 1:

            batch_target_splats = splats_test
            if dimension == 2:
                batch_target_splats = batch_target_splats.permute(0, 3, 1, 2)
            elif dimension == 3:
                batch_target_splats = batch_target_splats.permute(0, 4, 1, 2, 3)
            else:
                raise RuntimeError

            # ENCODE, SAMPLE AND DECODE
            means, log_variances = model.encode(batch_target_splats)
            batch_latent_momenta = means
            deformed_template_points = model(batch_latent_momenta)

            # LOSS
            if dataset in ['circles', 'ellipsoids', 'starmen', 'leaves', 'squares']:
                batch_target_points = points_test
                attachment_loss = torch.sum((deformed_template_points - batch_target_points) ** 2) / noise_variance

            elif dataset == 'hippocampi':
                batch_target_centers = centers_test
                batch_target_normals = normals_test
                batch_target_norms = norms_test

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

                # # Varifold
                # attachment_loss = 0.0
                # for p1, c2, n2, norm2 in zip(
                #         deformed_template_points, batch_target_centers, batch_target_normals, batch_target_norms):
                #     c1, n1 = compute_centers_and_normals(p1, model.template_connectivity)
                #     a1 = torch.norm(n1, 2, 1).view(-1, 1)
                #     a2 = torch.norm(n2, 2, 1).view(-1, 1)
                #     u1 = n1 / a1
                #     u2 = n2 / a2
                #     attachment_loss += (torch.sum(a1 * vkernel(gamma_varifold, c1, c1, u1, u1, a1)) +
                #                         torch.sum(a2 * vkernel(gamma_varifold, c2, c2, u2, u2, a2)) - 2 *
                #                         torch.sum(a1 * vkernel(gamma_varifold, c1, c2, u1, u2, a2)))
                # attachment_loss /= noise_variance

            else:
                raise RuntimeError

            test_attachment_loss += attachment_loss.detach().cpu().numpy()

            kullback_regularity_loss = torch.sum(
                (means.pow(2) + log_variances.exp()) / lambda_square - log_variances + np.log(lambda_square))
            test_kullback_regularity_loss += kullback_regularity_loss.detach().cpu().numpy()

            total_loss = attachment_loss + kullback_regularity_loss
            test_total_loss += total_loss.detach().cpu().numpy()

            test_attachment_loss /= float(number_of_meshes_test)
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

        ##############
        ### UPDATE ###
        ##############

        # if epoch > 100000:
        noise_variance *= train_attachment_loss / (model.template_points.size(0) * dimension)
        lambda_square = ss_z_var

        #############
        ### WRITE ###
        #############

        if epoch % print_every_n_iters == 0 or epoch == number_of_epochs:
            log += cprint(
                '\n[Epoch: %d] Learning rate = %.2E ; Noise std = %.2E ; Template latent q norm = %.3f'
                '\nss_z_mean = %.2E ; ss_z_var = %.2E ; lambda = %.2E'
                '\nTrain loss = %.3f (attachment = %.3f ; kullback regularity = %.3f)'
                '\nTest  loss = %.3f (attachment = %.3f ; kullback regularity = %.3f)' %
                (epoch, list(optimizer.param_groups)[0]['lr'], math.sqrt(noise_variance), template_latent_momenta_norm,
                 ss_z_mean, ss_z_var, np.sqrt(lambda_square),
                 train_total_loss, train_attachment_loss, train_kullback_regularity_loss,
                 test_total_loss, test_attachment_loss, test_kullback_regularity_loss))

        if epoch % save_every_n_iters == 0 or epoch == number_of_epochs:
            with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
                f.write(log)

            torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_%d__model.pth' % epoch))

            n = 3
            if dataset in ['ellipsoids', 'hippocampi']:
                model.write_meshes(
                    splats[:n].permute(0, 4, 1, 2, 3), points[:n], connectivities[:n],
                    os.path.join(output_dir, 'epoch_%d__train' % epoch))
                if number_of_meshes_test > 1:
                    model.write_meshes(
                        splats_test[:n].permute(0, 4, 1, 2, 3), points_test[:n], connectivities_test[:n],
                        os.path.join(output_dir, 'epoch_%d__test' % epoch))
                model.write_meshes(
                    template_splat.view((1,) + template_splat.size()),
                    model.template_points.view((1,) + model.template_points.size()),
                    model.template_connectivity.view((1,) + model.template_connectivity.size()),
                    os.path.join(output_dir, 'epoch_%d__template' % epoch))

            if dataset in ['starmen', 'circles', 'leaves', 'squares']:
                model.write_meshes(
                    splats[:n].permute(0, 3, 1, 2), points[:n], connectivities[:n],
                    os.path.join(output_dir, 'epoch_%d__train' % epoch))
                if number_of_meshes_test > 1:
                    model.write_meshes(
                        splats_test[:n].permute(0, 3, 1, 2), points_test[:n], connectivities_test[:n],
                        os.path.join(output_dir, 'epoch_%d__test' % epoch))
                model.write_meshes(
                    template_splat.view((1,) + template_splat.size()),
                    model.template_points.view((1,) + model.template_points.size()),
                    model.template_connectivity.view((1,) + model.template_connectivity.size()),
                    os.path.join(output_dir, 'epoch_%d__template' % epoch))
