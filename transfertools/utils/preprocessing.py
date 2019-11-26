# -*- coding: UTF-8 -*-
"""

Preprocessing.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ----------------------------------------------------------------------------
# scaler
# ----------------------------------------------------------------------------

class TransferScaler:
    """ Feature normalization.

    Parameters
    ----------
    scaling : str (default='none')
        Type of scaling to apply.

    Attributes
    ----------
    scaler_ : object
        The fitted scaler.
    """

    def __init__(self, scaling='none'):

        # initialize parameters
        self.scaling = str(scaling).lower()

    def fit_transform(self, X):
        """ Fit scaler to X and transform data.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        Xs : np.array of shape (n_samples, n_features)
            The scaled input instances.
        """

        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        """ Fit scaler to X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        self : object
        """
        
        # scaling types
        if self.scaling == 'none':
            pass
        
        elif self.scaling == 'standard':
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)

        elif self.scaling == 'minmax':
            self.scaler_ = MinMaxScaler(feature_range=(0, 1))
            self.scaler_.fit(X)

        else:
            raise ValueError(self.scaling,
                'not in [none, standard, minmax]')

    def transform(self, X):
        """ Transform data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        Xs : np.array of shape (n_samples, n_features)
            The scaled input instances.
        """
        
        # apply scaler
        if self.scaling == 'none':
            return X
        
        elif self.scaling == 'standard':
            return self.scaler_.transform(X)

        elif self.scaling == 'minmax':
            return self.scaler_.transform(X)

        else:
            raise ValueError(self.scaling,
                'not in [none, standard, minmax]')
