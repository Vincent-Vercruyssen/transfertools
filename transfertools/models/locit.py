# -*- coding: UTF-8 -*-
"""

Localized instance transfer.

Reference:
    V. Vercruyssen, W. Meert, J. Davis.
    Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.
    In AAAI Conference on Artificial Intelligence, New York, 2020.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .base import BaseDetector
from ..utils.preprocessing import TransferScaler


# ----------------------------------------------------------------------------
# LocIT class
# ----------------------------------------------------------------------------

class LocIT(BaseEstimator, BaseDetector):
    """ Localized instance transfer algorithm.

    Parameters
    ----------
    psi : int (default=10)
        Neighborhood size.

    transfer_threshold : float in [0.0, 1.0], optional (default=0.5)
        Threshold for the classifier that predicts whether a source
        instance will be transferred or not. The higher, the stricter.
        A threshold of 0.0 (1.0) means all (no) instances are transferred.

    train_selection : str (default='random')
        How to select the negative training instances:
        'farthest'  --> select the farthest instance
        'random'    --> random instance selected
        'edge'      --> select the (psi+1)'th instance

    scaling : str (default='standard')
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
                 psi=10,
                 transfer_threshold=0.5,
                 train_selection='random',
                 scaling='standard',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            scaling=scaling,
            tol=tol,
            verbose=verbose)
        
        # initialize parameters
        self.psi = int(psi)
        self.transfer_threshold = float(transfer_threshold)
        self.train_selection = str(train_selection).lower()

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

        self.Xt_trans_ = Xt

        # fit classifier on the target domain
        self._fit_transfer_classifier(Xt)

        # transferred instances
        self.Ixs_trans_ = self._predict_transfer_classifier(Xs)
        self.Xs_trans_ = Xs[self.Ixs_trans_, :]
        
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

        # transferred instances
        Ixs_trans = self._predict_transfer_classifier(Xs)
        Xs_trans = Xs[Ixs_trans, :]

        if return_indices:
            return Xs_trans, np.arange(0, ns, 1)
        return Xs_trans

    def _fit_transfer_classifier(self, Xt):
        """ Fit the transfer classifier.

        Parameters
        ----------
        Xt : np.array of shape (n_samples, n_features)
            The target instances.
        """

        n, _ = Xt.shape

        # nearest neighbor search structures
        self.target_tree_ = BallTree(Xt, leaf_size=32, metric='euclidean')
        _, Ixs = self.target_tree_.query(Xt, k=n)

        # construct training instances
        X_train = np.zeros((2 * n, 2), dtype=float)
        y_train = np.zeros(2 * n, dtype=float)
        random_ixs = np.arange(0, n, 1)
        np.random.shuffle(random_ixs)

        for i in range(n):
            # POS training instances
            # local mean and covaraiance matrix of the current point
            NN_x = Xt[Ixs[i, 1:self.psi+1], :]
            mu_x = np.mean(NN_x, axis=0)
            C_x = np.cov(NN_x.T)

            # local mean and covariance matrix of the nearest neighbor
            nn_ix = Ixs[i, 1]
            NN_nn = Xt[Ixs[nn_ix, 1:self.psi+1], :]
            mu_nn = np.mean(NN_nn, axis=0)
            C_nn = np.cov(NN_nn.T)

            # NEG training instances
            # local mean and covariance matrix
            if self.train_selection == 'farthest':
                r_ix = Ixs[i, -1]
            elif self.train_selection == 'edge':
                r_ix = Ixs[i, self.psi+2]
            elif self.train_selection == 'random':
                r_ix = random_ixs[i]
            else:
                raise ValueError(self.train_selection,
                    'not in [farthest, edge, random]')
            NN_r = Xt[Ixs[r_ix, 1:self.psi], :]
            mu_r = np.mean(NN_r, axis=0)
            C_r = np.cov(NN_r.T)
            
            # training instances
            f_pos = np.array([float(np.linalg.norm(mu_x - mu_nn)), float(
                np.linalg.norm(C_x - C_nn)) / float(np.linalg.norm(C_x) + self.tol)])
            f_neg = np.array([float(np.linalg.norm(mu_x - mu_r)), float(
                np.linalg.norm(C_x - C_r)) / float(np.linalg.norm(C_x) + self.tol)])

            # labels
            X_train[2*i, :] = f_pos
            y_train[2*i] = 1.0
            X_train[2*i+1, :] = f_neg
            y_train[2*i+1] = 0.0
        
        # replace NaN and inf by
        X_train = np.nan_to_num(X_train)

        # scale training instances
        self.scaler_ = StandardScaler()
        X_train = self.scaler_.fit_transform(X_train)

        # train the classifier
        self.clf = self._optimal_transfer_classifier(X_train, y_train)

    def _predict_transfer_classifier(self, X):
        """ Predict transfer with the classifier.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        Ix_trans : np.array of shape (<= n_samples,)
            The indices of the source instances selected for transfer.

        Comments
        --------
        Needs the target data.
        """

        n, _ = X.shape

        # nearest neighbor search structures
        self.source_tree_ = BallTree(X, leaf_size=32, metric='euclidean')
        _, Ixs = self.source_tree_.query(X, k=self.psi+1)
        _, Ixt = self.target_tree_.query(X, k=self.psi)

        # construct feature vectors
        X_feat = np.zeros((n, 2), dtype=float)
        
        for i in range(n):
            # local mean and covariance matrix in the source domain
            NN_s = X[Ixs[i, 1:self.psi+1], :]
            mu_s = np.mean(NN_s, axis=0)
            C_s = np.cov(NN_s.T)

            # local mean and covariance matrix in the target domain
            NN_t = self.Xt_trans_[Ixt[i, :self.psi], :]
            mu_t = np.mean(NN_t, axis=0)
            C_t = np.cov(NN_t.T)

            # feature vector
            f = np.array([float(np.linalg.norm(mu_s - mu_t)), float(
                np.linalg.norm(C_s - C_t)) / float(np.linalg.norm(C_s) + self.tol)])
            X_feat[i, :] = f

        # nan to num
        X_feat = np.nan_to_num(X_feat)

        # scaling + predict
        X_feat = self.scaler_.transform(X_feat)
        labels = self.clf.predict(X_feat)
        Ix_trans = np.where(labels == 1.0)[0]

        return Ix_trans

    def _optimal_transfer_classifier(self, X, y):
        """ Optimal transfer classifier based on SVC.
        """
        
        # parameters to tune
        tuned_parameters = [{'C': [0.01, 0.1, 0.5, 1, 10, 100],
                           'gamma': [0.01, 0.1, 0.5, 1, 10, 100],
                           'kernel': ['rbf']},
                           {'kernel': ['linear'],
                           'C': [0.01, 0.1, 0.5, 1, 10, 100]}]
        
        # grid search
        svc = SVC(probability=True)
        clf = GridSearchCV(svc, tuned_parameters, cv=3, refit=True)
        clf.fit(X, y)
        
        return clf