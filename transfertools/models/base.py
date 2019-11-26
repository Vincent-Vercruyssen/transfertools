# -*- coding: UTF-8 -*-
"""

Base class for all transfer models.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from abc import abstractmethod, ABCMeta
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# BaseDetector
# ----------------------------------------------------------------------------

class BaseDetector(metaclass=ABCMeta):
    """ Abstract class for the transfer learning algorithms.

    Parameters
    ----------
    scaling : str (default='none')
        Scale the source and target domain before transfer.

    tol : float in (0, +inf), optional (default=1e-10)
        The tolerance.

    verbose : bool, optional (default=False)
        Verbose or not.

    Attributes
    ----------
    type_ : str
        The type of transfer learning (e.g., domain adaptation).

    Xs_trans_ : np.array of shape (<= n_samples,)
        The (transformed) source instances after transfer.

    Xt_trans_ : np.array of shape (<= n_samples,)
        The (transformed) target instances after transfer.

    Ixs_trans_ : np.array of shape (n_samples, n_features)
        The indices of the source instances selected for transfer.
    """

    @abstractmethod
    def __init__(self,
                 scaling='none',
                 tol=1e-10,
                 verbose=False):
        super().__init__()

        # initialize
        self.scaling = str(scaling)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    # must be implemented by derived classes
    @abstractmethod
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

        pass

    @abstractmethod
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
        
        pass

    # must NOT be implemented by derived classes
    def fit_transfer(self, Xs=None, Xt=None, ys=None, yt=None, return_indices=False, *args, **kwargs):
        """ Fit the model and determine the instances to transfer.

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
        return_indices : bool, optional (default=False)
            Also return the indices of the source instances
            selected for transfer.

        Returns
        -------
        Xs_trans : np.array of shape (n_samples, n_features)
            The (transformed) source instances after transfer.
        Xt_trans : np.array of shape (n_samples, n_features)
            The (transformed) target instances after transfer.
        Ixs_trans : np.array of shape (<= n_samples,)
            The indices of the source instances selected for transfer.
        """
        
        self.fit(Xs, Xt, ys, yt, *args, **kwargs)
        if return_indices:
            return self.Xs_trans_, self.Xt_trans_, self.Ixs_trans_
        return self.Xs_trans_, self.Xt_trans_

    # methods used by all derived classes
    def _check_all_inputs(self, Xs=None, Xt=None, ys=None, yt=None):
        """ Check all the inputs using Scikit-learn's functionality.

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
        Xs : np.array of shape (n_samples, n_features), optional (default=None)
            The source instances.
        Xt : np.array of shape (n_samples, n_features), optional (default=None)
            The target instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.
        yt : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the target instances.
        """

        # input matrices check
        if Xs is not None:
            if ys is None:
                ys = np.zeros(Xs.shape[0])
            Xs, ys = check_X_y(Xs, ys)
            source = True
        else:
            source = False
        
        if Xt is not None:
            if yt is None:
                yt = np.zeros(Xt.shape[0])
            Xt, yt = check_X_y(Xt, yt)
            target = True
        else:
            target = False

        # dimension check
        if source and target:
            if not(Xs.shape[1] == Xt.shape[1]):
                raise ValueError('Source and target domain should have the same dimensions')
        
        if source and target:
            return Xs, Xt, ys, yt
        elif source and not(target):
            return Xs, ys
        elif target and not(source):
            return Xt, yt
        else:
            return None

