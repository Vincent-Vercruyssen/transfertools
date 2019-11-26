# -*- coding: UTF-8 -*-
"""

Compute different kernels.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import numpy as np

from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------

def is_pos_def(X):
    """ Check for positive definiteness.
    """
    
    return np.all(np.linalg.eigvals(X) > 0)


def kernel(X1, X2=None, kernel_type='rbf', sigma=1.0, degree=1):
    """ Compute the kernel for a given X1 and X2.

    Parameters
    ----------
    X1 : np.array of shape (n_samples, n_features)
        Input instances X1.
    X2 : np.array of shape (n_samples, n_features)
        Input instances of X2.
    kernel_type : str (default='rbf')
        Different kernel types in:
        [rbf, linear, sigmoid, polynomial, primal, sam]
    sigma : float (default=1.0)
        Sigma (bandwidth) of the rbf kernel.
        Sigma parameter of the sam kernel.
    degree : int (default=1)
        Degree of the polynomial kernel.

    Returns
    -------
    K : np.array of shape (n_samples, n_samples)
        Kernel computed on X1 and X2.
    """

    # make copy
    if X2 is None:
        X2 = X1.copy()

    # data shape
    n1, nf1 = X1.shape
    n2, nf2 = X2.shape

    # dimension check
    if not(nf1 == nf2):
        raise ValueError('Dimensions of kernel input matrices should be equal')

    # compute the kernels
    if kernel_type.lower() == 'primal':
        K = X1

    elif kernel_type.lower() == 'linear':
        K = np.dot(X1, X2.T)

    elif kernel_type.lower() == 'rbf':
        K = np.exp(-cdist(X1, X2) / (2.0 * (sigma ** 2)))

    elif kernel_type.lower() == 'sigmoid':
        K = 1.0 / (np.exp(np.dot(X1, X2.T)) + 1.0)

    elif kernel_type.lower() == 'polynomial':
        K = np.power(np.dot(X1, X2.T) + 1.0, degree)

    elif kernel_type.lower() == 'sam':
        D = np.dot(X1, X2.T)
        D_flat = D.ravel()
        acos_func = np.vectorize(math.acos)
        D_flat_acos = acos_func(D_flat)
        D = D_flat_acos.reshape(D.shape)

        # kernel
        K = np.exp(np.power(D, 2) / (2.0 * (sigma ** 2)))

    else:
        raise ValueError(kernel_type,
            'not in [rbf, linear, sigmoid, polynomial, primal, sam]')
    
    # check output dimensions
    assert K.shape == (n1, n2), 'kernel matrix has wrong dimensions'

    return K
