# -*- coding: UTF-8 -*-
"""

Transfer component analysis.

Reference:
    Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2010).
    Domain adaptation via transfer component analysis.
    IEEE Transactions on Neural Networks, 22(2), 199-210.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator
from scipy.sparse.linalg import eigs

from .base import BaseDetector
from ..utils.preprocessing import TransferScaler
from ..utils.utils import *


# ----------------------------------------------------------------------------
# TCA class
# ----------------------------------------------------------------------------

class TCA(BaseEstimator, BaseDetector):
    """ Transfer component analysis.

    Parameters
    ----------
    mu : float (default=0.1)
        Trade-off parameter.

    n_components : int (default=1)
        Number of transfer components to retain.

    kernel_type : str (default='rbf')
        Different kernel types in:
        [rbf, linear, sigmoid, polynomial, primal, sam]

    sigma : float (default=1.0)
        Sigma (bandwidth) of the rbf kernel.
        Sigma parameter of the sam kernel.

    degree : int (default=1)
        Degree of the polynomial kernel.

    scaling : str (default='none')
        Scale the source and target domain before transfer.

    Attributes
    ----------
    type_ : str
        The type of transfer learning (e.g., domain adaptation).

    X_trans_ : np.array of shape (<= n_samples,)
        The (transformed) source instances that are transferred.

    Ixs_trans_ : np.array of shape (n_samples, n_features)
        The indices of the instances selected for transfer.
    """

    def __init__(self,
                 mu=0.1,
                 n_components=2,
                 kernel_type='rbf',
                 sigma=1.0,
                 degree=2,
                 scaling='none',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            scaling=scaling,
            tol=tol,
            verbose=verbose)

        # initialize parameters
        self.mu = float(mu)
        self.n_components = int(n_components)
        self.kernel_type = str(kernel_type)
        self.sigma = float(sigma)
        self.degree = int(degree)
        
        # type
        self.type_ = 'domain_adaptation'

    def fit(self, Xs=None, Xt=None, ys=None, yt=None):
        """ Fit the model on data X.

        Parameters
        ----------
        Xs : np.array of shape (n_samples, n_features), optional (default=None)
            The source instances.
        Xt : np.array of shape (n_samples, n_features), optional (default=None)
            The target instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.
        yt : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the target instances.

        Returns
        -------
        self : object
        """

        # check all inputs
        Xs, Xt, ys, yt = self._check_all_inputs(Xs, Xt, ys, yt)

        ns, nfs = Xs.shape
        nt, nft = Xt.shape

        # number of components
        if not(self.n_components <= ns + nt - 1):
            raise ValueError('`n_components` too large')

        # feature normalization/standardization!
        self.target_scaler_ = TransferScaler(self.scaling)
        self.source_scaler_ = TransferScaler(self.scaling)
        Xt = self.target_scaler_.fit_transform(Xt)
        Xs = self.source_scaler_.fit_transform(Xs)

        # kernel matrix
        self.X_ = np.concatenate((Xs, Xt), axis=0)
        K = kernel(self.X_, self.X_, kernel_type=self.kernel_type,
            sigma=self.sigma, degree=self.degree)

        # kernel matrix should be postive definite
        # adapted from: https://github.com/wmkouw/libTLDA/blob/master/libtlda/tca.py
        if not is_pos_def(K):
            print('Warning: covariate matrices not PSD.')

            regct = -6
            while not is_pos_def(K):
                print('Adding regularization:', 10 ** regct)

                # Add regularization
                K += np.eye(ns + nt) * (10.0 ** regct)

                # Increment regularization counter
                regct += 1

        # coefficient matrix L
        L = np.vstack((
            np.hstack((
                np.ones((ns, ns)) / ns ** 2,
                -1.0 * np.ones((ns, nt)) / (ns * nt))),
            np.hstack((
                -1.0 * np.ones((nt, ns)) / (ns * nt),
                np.ones((nt, nt)) / (nt ** 2)))
            ))

        # centering matrix H
        H = np.eye(ns + nt) - np.ones((ns + nt, ns + nt)) / float(ns + nt)

        # matrix Lagrangian objective function: (I + mu*K*L*K)^{-1}*K*H*K
        J = np.dot(np.linalg.inv(np.eye(ns + nt) +
                   self.mu * np.dot(np.dot(K, L), K)),
                   np.dot(np.dot(K, H), K))

        # eigenvector decomposition as solution to trace minimization
        _, C = eigs(J, k=self.n_components)

        # transformation/embedding matrix
        self.C_ = np.real(C)

        # transform the source data
        self.Xs_trans_ = np.dot(K[:ns, :], self.C_)
        self.Xt_trans_ = np.dot(K[ns:, :], self.C_)
        self.Ixs_trans_ = np.arange(0, ns, 1)
        
        return self

    def transfer(self, Xs, ys=None, return_indices=False):
        """ Apply transfer to the source instances.
        
        Parameters
        ----------
        Xs : np.array of shape (n_samples, n_features)
            The source instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.
        return_indices : bool, optional (default=False)
            Also return the indices of the source instances
            selected for transfer.

        Returns
        -------
        Xs_trans : np.array of shape (n_samples, n_features)
            The (transformed) source instances after transfer.
        Ixs_trans : np.array of shape (<= n_samples,)
            The indices of the source instances selected for transfer.
        """
        
        # check all inputs
        Xs, ys = self._check_all_inputs(Xs=Xs, ys=ys)

        ns, _ = Xs.shape

        # scaling
        Xs = self.source_scaler_.transform(Xs)

        # compute the kernel matrix
        K = kernel(Xs, self.X_, kernel_type=self.kernel_type,
            sigma=self.sigma, degree=self.degree)

        # map data onto transfer components
        Xs_trans = np.dot(K, self.C_)

        if return_indices:
            return Xs_trans, np.arange(0, ns, 1)
        return Xs_trans
