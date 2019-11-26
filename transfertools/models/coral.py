# -*- coding: UTF-8 -*-
"""

Correlation alignment.

Reference:
    Sun, B., Feng, J., & Saenko, K. (2016, March).
    Return of frustratingly easy domain adaptation.
    In Thirtieth AAAI Conference on Artificial Intelligence.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator

from .base import BaseDetector
from ..utils.preprocessing import TransferScaler


# ----------------------------------------------------------------------------
# CORAL class
# ----------------------------------------------------------------------------

class CORAL(BaseEstimator, BaseDetector):
    """ Correlation alignment algorithm.

    Parameters
    ----------
    scaling : str (default='standard')
        Scale the source and target domain before transfer.
        Standard scaling is indicated in the paper.

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
                 scaling='standard',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            scaling=scaling,
            tol=tol,
            verbose=verbose)
        
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

        # align means: feature normalization/standardization!
        self.target_scaler_ = TransferScaler(self.scaling)
        self.source_scaler_ = TransferScaler(self.scaling)
        Xt = self.target_scaler_.fit_transform(Xt)
        Xs = self.source_scaler_.fit_transform(Xs)

        # align covariances: denoising - noising transformation
        Cs = np.cov(Xs.T) + np.eye(nfs)
        Ct = np.cov(Xt.T) + np.eye(nft)
        csp = sp.linalg.fractional_matrix_power(Cs, -1/2)
        ctp = sp.linalg.fractional_matrix_power(Ct, 1/2)
        self.A_ = np.dot(csp, ctp)

        # transferred source instances
        self.Xt_trans_ = Xt
        self.Xs_trans_ = np.dot(Xs, self.A_).real
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

        # transform
        Xs_trans = np.dot(Xs, self.A_).real

        if return_indices:
            return Xs_trans, np.arange(0, ns, 1)
        return Xs_trans
