# -*- coding: UTF-8 -*-
"""

Cluster-based instance transfer.

Reference:
    Vercruyssen, V., Meert, W., & Davis, J. (2017).
    Transfer learning for time series anomaly detection.
    In CEUR Workshop Proceedings (Vol. 1924, pp. 27-37).

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from .base import BaseDetector
from ..utils.preprocessing import TransferScaler


# ----------------------------------------------------------------------------
# CBIT class
# ----------------------------------------------------------------------------

class CBIT(BaseEstimator, BaseDetector):
    """ Cluster-based instance transfer.

    Parameters
    ----------
    n_clusters : int (default=10)
        Number of clusters.

    beta : float in [0.0, +inf[ (default=2.0)
        Size ratio between two clusters that serves as the
        threshold to distinguish small from large clusters.

    contamination : float in [0.0, 1.0] (default=0.1)
        Expected proportion of anomalies/outliers in the data.

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
                 n_clusters=10,
                 beta=2.0,
                 contamination=0.1,
                 scaling='standard',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            scaling=scaling,
            tol=tol,
            verbose=verbose)
        
        # initialize parameters
        self.n_clusters = int(n_clusters)
        self.beta = float(beta)
        self.c = float(contamination)

        # type
        self.type_ = 'instance_selection'

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

        ns, _ = Xs.shape
        nt, _ = Xt.shape

        # align means: feature normalization/standardization!
        self.target_scaler_ = TransferScaler(self.scaling)
        self.source_scaler_ = TransferScaler(self.scaling)
        Xt = self.target_scaler_.fit_transform(Xt)
        Xs = self.source_scaler_.fit_transform(Xs)

        # cluster the target data
        cluster_labels, self.centroids_ = self._cluster_data(Xt)

        # calculate the cluster radii and cluster sizes
        # radius = distance between center and farthest point
        self.cluster_radii_ = np.zeros(self.n_clusters, dtype=float)
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)
        for i, x in enumerate(Xt):
            l = cluster_labels[i]
            cluster_sizes[l] += 1
            d = np.linalg.norm(x - self.centroids_[l])
            if d > self.cluster_radii_[l]:
                self.cluster_radii_[l] = d
        sizes_sorted = np.sort(cluster_sizes)[::-1]
        index_sorted = np.argsort(cluster_sizes)[::-1]

        # determine the small and large clusters
        cumsum, i = 0, 0
        stop = False
        large_only = False
        while not(stop):
            if i < self.n_clusters - 1:
                cs = sizes_sorted[i]
                ns = sizes_sorted[i+1]
                cumsum = cumsum + cs

                # condition 1
                if float(cumsum) > (nt * (1.0 - self.c)):
                    stop = True

                # condition 2
                if cs / ns >= self.beta:
                    stop = True
            
            else:
                stop = True
                large_only = True
            i += 1
        if large_only:
            self.large_clusters_ = index_sorted
        else:
            self.large_clusters_ = index_sorted[:i]
        
        # transferred source instances
        self.Ixs_trans_ = self._cluster_based_instance_transfer(Xs, ys)
        if len(self.Ixs_trans_) > 0:
            self.Xs_trans_ = Xs[self.Ixs_trans_, :]
        else:
            self.Xs_trans_ = np.array([])
        self.Xt_trans_ = Xt
        
        return self

    def transfer(self, Xs, ys=None, return_indices=False):
        """ Apply transfer to the source instances.
            Actually requires the source labels.
        
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

        # decide which instances to transfer
        Ixs_trans = self._cluster_based_instance_transfer(Xs, ys)
        if len(Ixs_trans) > 0:
            Xs_trans = Xs[Ixs_trans, :]
        else:
            Xs_trans = np.array([])

        if return_indices:
            return Xs_trans, Ixs_trans
        return Xs_trans

    def _cluster_based_instance_transfer(self, Xs, ys):
        """ Cluster the data using k-means.

        Parameters
        ----------
        Xs : np.array of shape (n_samples, n_features)
            The source instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.

        Returns
        -------
        Ix_trans : np.array of shape (<= n_samples,)
            The indices of the source instances selected for transfer.
        """

        # to which cluster belong the instances in X
        cluster_labels = self.clusterer_.predict(Xs)

        # transfer
        Ix_trans = []
        for i, l in enumerate(cluster_labels):
            cr = self.cluster_radii_[l]
            cc = self.centroids_[l, :]
            d = np.linalg.norm(Xs[i, :] - cc)
            
            # label = normal --> should be in large cluster
            if ys[i] == -1:
                if l in self.large_clusters_ and d <= cr:
                    Ix_trans.append(i)

            # label = anomaly
            elif ys[i] == 1:
                if not(l in self.large_clusters_ and d <= cr):
                    Ix_trans.append(i)

            # no label
            else:
                pass

        return np.array(Ix_trans)

    def _cluster_data(self, X):
        """ Cluster the data using k-means.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The source instances.

        Returns
        -------
        cluster_labels : np.array of shape (n_samples,)
            Cluster labels of the instances in X.
        centroids : np.array of shape (self.n_clusters, n_features)
            The cluster centroids.
        """

        self.clusterer_ = KMeans(n_clusters=self.n_clusters, init='k-means++')
        self.clusterer_.fit(X)

        return self.clusterer_.labels_, self.clusterer_.cluster_centers_

