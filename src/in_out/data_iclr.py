### Base ###
import math

### Core ###
import numpy as np
import torch
import PIL.Image as pimg


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
