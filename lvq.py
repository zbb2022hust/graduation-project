from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import math
from math import log
from scipy.optimize import minimize
from sklearn.utils import validation

"""===============================lvqbase==========================="""
class _LvqBaseModel(BaseEstimator, ClassifierMixin):

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 max_iter=2500, gtol=1e-5, lr=0.0001, display=False, random_state=None):
        self.random_state = random_state
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.display = display
        self.max_iter = max_iter
        self.gtol = gtol
        self.lr = lr

    def _validate_train_parms(self, train_set, train_lab):
        random_state = validation.check_random_state(self.random_state)
        if not isinstance(self.display, bool):
            raise ValueError("display must be a boolean")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an positive integer")
        if not isinstance(self.gtol, float) or self.gtol <= 0:
            raise ValueError("gtol must be a positive float")
        train_set, train_lab = validation.check_X_y(train_set, train_lab)

        self.classes_ = unique_labels(train_lab)
        nb_classes = len(self.classes_)
        nb_samples, nb_features = train_set.shape  # nb_samples unused

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            if self.prototypes_per_class < 0 or not isinstance(
                    self.prototypes_per_class, int):
                raise ValueError("prototypes_per_class must be a positive int")
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))
        # initialize prototypes
        if self.initial_prototypes is None:
            self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
            self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClass in range(nb_classes):
                nb_prot = nb_ppc[actClass]
                mean = np.mean(
                    train_set[train_lab == self.classes_[actClass], :], 0)
                self.w_[pos:pos + nb_prot] = mean + (
                        random_state.rand(nb_prot, nb_features) * 2 - 1)
                self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        return train_set, train_lab, random_state

    def fit(self, x, y):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)

        Returns
        --------
        self
        """
        x, y, random_state = self._validate_train_parms(x, y)
        if len(np.unique(y)) == 1:
            raise ValueError("fitting " + type(
                self).__name__ + " with only one class is not possible")
        self._optimize(x, y, random_state)
        return self

    def project(self, x, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of trained
        model to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        if print_variance_covered:
            print('not implemented!')
        return x[:, :dims]

"""===============================glvq==========================="""
def _squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


class GlvqModel(_LvqBaseModel):
    """Generalized Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.

    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of bfgs.

    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))

    C : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    See also
    --------
    GrlvqModel, GmlvqModel, LgmlvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 max_iter=2500, gtol=1e-5, beta=2, C=None,
                 display=False, random_state=None, lr=0.0001):
        super(GlvqModel, self).__init__(prototypes_per_class=prototypes_per_class,
                                        initial_prototypes=initial_prototypes,
                                        max_iter=max_iter, gtol=gtol, display=display,
                                        random_state=random_state, lr=lr)
        self.beta = beta
        self.c = C
        self.lr = lr

    def phi(self, x):
        """
        Parameters
        ----------

        x : input value

        """
        return 1 / (1 + np.math.exp(-self.beta * x))

    def phi_prime(self, x):
        """
        Parameters
        ----------

        x : input value

        """
        return self.beta * np.math.exp(-self.beta * x) / (
                1 + np.math.exp(-self.beta * x)) ** 2

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(prototypes.shape)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
            g[i] = dcd.dot(training_data[idxw]) - dwd.dot(
                training_data[idxc]) + (dwd.sum(0) -
                                        dcd.sum(0)) * prototypes[i]
        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + self.lr * (random_state.rand(*g.shape) - 0.5))
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        [self._map_to_int(x) for x in self.c_w_[label_equals_prototype.argmax(1)]]
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]  # y_real, y_pred

        return np.vectorize(self.phi)(mu).sum(0)

    def _validate_train_parms(self, train_set, train_lab):
        if not isinstance(self.beta, int):
            raise ValueError("beta must a an integer")

        ret = super(GlvqModel, self)._validate_train_parms(train_set, train_lab)

        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))
        if self.c is not None:
            self.c = validation.check_array(self.c)
            if self.c.shape != (2, 3):
                raise ValueError("C must be shape (2,3)")
            for k1, k2, v in self.c:
                self.c_[tuple(zip(*product(self._map_to_int(k1), self._map_to_int(k2))))] = float(v)
        return ret

    def _map_to_int(self, item):
        return np.where(self.c_w_ == item)[0]

    def _optimize(self, x, y, random_state):
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        res = minimize(
            fun=lambda vs: self._optfun(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype,
                random_state=random_state),
            method='l-bfgs-b', x0=self.w_,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        self.w_ = res.x.reshape(self.w_.shape)
        self.n_iter_ = res.nit

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_
        return cdist(x, w, 'euclidean')

    def predict(self, x):
        """Predict class membership index for each input sample.

        This function does classification on an array of
        test vectors X.


        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]


        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)
        return (self.c_w_[dist.argmin(1)])

"""===============================gmlvq框架==========================="""
class GmlvqModel(GlvqModel):
    """Generalized Matrix Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different numbers
        per class.

    initial_prototypes : array-like,
     shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype

    initial_matrix : array-like, shape = [dim, n_features], optional
        Relevance matrix to start with.
        If not given random initialization for rectangular matrix and unity
        for squared matrix.

    regularization : float, optional (default=0.0)
        Value between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    dim : int, optional (default=nb_features)
        Maximum rank or projection dimensions

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful
        termination of l-bfgs-b.

    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))

    C : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    dim_ : int
        Maximum rank or projection dimensions

    omega_ : array-like, shape = [dim, n_features]
        Relevance matrix

    See also
    --------
    GlvqModel, GrlvqModel, LgmlvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 initial_matrix=None, regularization=0.0, dim=None,
                 max_iter=2500, gtol=1e-5, beta=2, C=None, display=False,
                 random_state=None, lr=0.0001):
        super(GmlvqModel, self).__init__(prototypes_per_class,
                                         initial_prototypes, max_iter,
                                         gtol, beta, C, display, random_state, lr)
        self.regularization = regularization
        self.initial_matrix = initial_matrix
        self.initialdim = dim
        self.lr = lr

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state, lr_relevances=0, lr_prototypes=1):
        n_data, n_dim = training_data.shape
        variables = variables.reshape(variables.size // n_dim, n_dim)
        nb_prototypes = self.c_w_.shape[0]
        omega_t = variables[nb_prototypes:].conj().T
        # dist = _squared_euclidean(training_data.dot(omega_t),
        #                           variables[:nb_prototypes].dot(omega_t))
        dist = self._compute_distance(training_data, variables[:nb_prototypes],
                                      omega_t.T)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = np.vectorize(self.phi_prime)(mu)
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]

        g = np.zeros(variables.shape)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        if lr_relevances > 0:
            gw = np.zeros(omega_t.T.shape)

        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
            if lr_relevances > 0:
                difc = training_data[idxc] - variables[i]
                difw = training_data[idxw] - variables[i]
                gw -= np.dot(difw * dcd[np.newaxis].T, omega_t).T.dot(difw) - \
                      np.dot(difc * dwd[np.newaxis].T, omega_t).T.dot(difc)
                if lr_prototypes > 0:
                    g[i] = dcd.dot(difw) - dwd.dot(difc)
            elif lr_prototypes > 0:
                g[i] = dcd.dot(training_data[idxw]) - \
                       dwd.dot(training_data[idxc]) + \
                       (dwd.sum(0) - dcd.sum(0)) * variables[i]
        f3 = 0
        if self.regularization:
            f3 = np.linalg.pinv(omega_t.conj().T).conj().T
        if lr_relevances > 0:
            g[nb_prototypes:] = 2 / n_data \
                                * lr_relevances * gw - self.regularization * f3
        if lr_prototypes > 0:
            g[:nb_prototypes] = 1 / n_data * lr_prototypes \
                                * g[:nb_prototypes].dot(omega_t.dot(omega_t.T))
        g = g * (1 + self.lr * (random_state.rand(*g.shape) - 0.5))
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        variables = variables.reshape(variables.size // n_dim, n_dim)
        nb_prototypes = self.c_w_.shape[0]
        omega_t = variables[nb_prototypes:]  # .conj().T

        # dist = _squared_euclidean(training_data.dot(omega_t),
        #                           variables[:nb_prototypes].dot(omega_t))
        dist = self._compute_distance(training_data, variables[:nb_prototypes],
                                      omega_t)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong

        if self.regularization > 0:
            reg_term = self.regularization * log(
                np.linalg.det(omega_t.conj().T.dot(omega_t)))
            return np.vectorize(self.phi)(mu).sum(0) - reg_term  # f
        return np.vectorize(self.phi)(mu).sum(0)

    def _optimize(self, x, y, random_state):
        if not isinstance(self.regularization,
                          float) or self.regularization < 0:
            raise ValueError("regularization must be a positive float ")
        nb_prototypes, nb_features = self.w_.shape
        if self.initialdim is None:
            self.dim_ = nb_features
        elif not isinstance(self.initialdim, int) or self.initialdim <= 0:
            raise ValueError("dim must be an positive int")
        else:
            self.dim_ = self.initialdim

        if self.initial_matrix is None:
            if self.dim_ == nb_features:
                self.omega_ = np.eye(nb_features)
            else:
                self.omega_ = random_state.rand(self.dim_, nb_features) * 2 - 1
        else:
            self.omega_ = validation.check_array(self.initial_matrix)
            if self.omega_.shape[1] != nb_features:  # TODO: check dim
                raise ValueError(
                    "initial matrix has wrong number of features\n"
                    "found=%d\n"
                    "expected=%d" % (self.omega_.shape[1], nb_features))

        variables = np.append(self.w_, self.omega_, axis=0)
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        method = 'l-bfgs-b'
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
                          random_state=random_state,
                          lr_prototypes=1, lr_relevances=0),
            method=method, x0=variables,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = res.nit
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
                          random_state=random_state,
                          lr_prototypes=0, lr_relevances=1),
            method=method, x0=res.x,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        res = minimize(
            fun=lambda vs:
            self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs:
            self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
                          random_state=random_state,
                          lr_prototypes=1, lr_relevances=1),
            method=method, x0=res.x,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        out = res.x.reshape(res.x.size // nb_features, nb_features)
        self.w_ = out[:nb_prototypes]
        self.omega_ = out[nb_prototypes:]
        self.omega_ /= math.sqrt(
            np.sum(np.diag(self.omega_.T.dot(self.omega_))))
        self.n_iter_ = n_iter

    def _compute_distance(self, x, w=None, omega=None):
        if w is None:
            w = self.w_
        if omega is None:
            omega = self.omega_
        nb_samples = x.shape[0]
        nb_prototypes = w.shape[0]
        distance = np.zeros([nb_prototypes, nb_samples])
        for i in range(nb_prototypes):
            distance[i] = np.sum((x - w[i]).dot(omega.T) ** 2, 1)
        return distance.T

    def project(self, x, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of trained
        model to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        v, u = np.linalg.eig(self.omega_.conj().T.dot(self.omega_))
        idx = v.argsort()[::-1]
        if print_variance_covered:
            print('variance coverd by projection:',
                  v[idx][:dims].sum() / v.sum() * 100)
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))

"""===============================lgmlvq==========================="""
class LgmlvqModel(GlvqModel):
    """Localized Generalized Matrix Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different numbers
        per class.

    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

    initial_matrices : list of array-like, optional
        Matrices to start with. If not given random initialization

    regularization : float or array-like, shape = [n_classes/n_prototypes],
     optional (default=0.0)
        Values between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    dim : int, optional
        Maximum rank or projection dimensions

    classwise : boolean, optional
        If true, each class has one relevance matrix.
        If false, each prototype has one relevance matrix.

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of l-bfgs-b.

    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))

    C : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    omegas_ : list of array-like
        Relevance Matrices

    dim_ : list of int
        Maximum rank of projection

    regularization_ : array-like, shape = [n_classes/n_prototypes]
        Values between 0 and 1

    See also
    --------
    GlvqModel, GrlvqModel, GmlvqModel
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 initial_matrices=None, regularization=0.0,
                 dim=None, classwise=False, max_iter=2500, gtol=1e-5,
                 beta=2, C=None, display=False, random_state=None,
                 lr=0.0001):
        super(LgmlvqModel, self).__init__(prototypes_per_class,
                                          initial_prototypes, max_iter,
                                          gtol, beta, C, display, random_state, lr)
        self.regularization = regularization
        self.initial_matrices = initial_matrices
        self.classwise = classwise
        self.initialdim = dim
        self.lr = lr

    def _g(self, variables, training_data, label_equals_prototype,
           random_state, lr_relevances=0,
           lr_prototypes=1):
        # print("g")
        nb_samples, nb_features = training_data.shape
        nb_prototypes = self.c_w_.shape[0]
        variables = variables.reshape(variables.size // nb_features,
                                      nb_features)
        # dim to indices
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        psis = np.split(variables[nb_prototypes:], indices[:-1])  # .conj().T

        dist = self._compute_distance(training_data, variables[:nb_prototypes],
                                      psis)  # change dist function ?
        # dist = cdist(training_data, prototypes, 'sqeuclidean')
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(variables.shape)
        normfactors = 4 / distcorrectpluswrong ** 2

        if lr_relevances > 0:
            gw = []
            for i in range(len(psis)):
                gw.append(np.zeros(psis[i].shape))
        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong
            if self.classwise:
                right_idx = np.where((self.c_w_[i] == self.classes_) == 1)[0][
                    0]  # test if works
            else:
                right_idx = i
            dcd = mu[idxw] * distcorrect[idxw] * normfactors[idxw]
            dwd = mu[idxc] * distwrong[idxc] * normfactors[idxc]

            difc = training_data[idxc] - variables[i]
            difw = training_data[idxw] - variables[i]
            if lr_prototypes > 0:
                g[i] = (dcd.dot(difw) - dwd.dot(difc)).dot(
                    psis[right_idx].conj().T).dot(psis[right_idx])
            if lr_relevances > 0:
                gw[right_idx] -= (difw * dcd[np.newaxis].T).dot(psis[right_idx].conj().T).T.dot(difw) - \
                                 (difc * dwd[np.newaxis].T).dot(psis[right_idx].conj().T).T.dot(difc)
        if lr_relevances > 0:
            if sum(self.regularization_) > 0:
                regmatrices = np.zeros([sum(self.dim_), nb_features])
                for i in range(len(psis)):
                    regmatrices[sum(self.dim_[:i + 1]) - self.dim_[i]:sum(
                        self.dim_[:i + 1])] = \
                        self.regularization_[i] * np.linalg.pinv(psis[i])
                g[nb_prototypes:] = 2 / nb_samples * lr_relevances * \
                                    np.concatenate(gw) - regmatrices
            else:
                g[nb_prototypes:] = 2 / nb_samples * lr_relevances * \
                                    np.concatenate(gw)
        if lr_prototypes > 0:
            g[:nb_prototypes] = 1 / nb_samples * \
                                lr_prototypes * g[:nb_prototypes]
        g = g * (1 + self.lr * (random_state.rand(*g.shape) - 0.5))
        return g.ravel()

    def _f(self, variables, training_data, label_equals_prototype):
        # print("f")
        nb_samples, nb_features = training_data.shape
        nb_prototypes = self.c_w_.shape[0]
        variables = variables.reshape(variables.size // nb_features,
                                      nb_features)
        # dim to indices
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        psis = np.split(variables[nb_prototypes:], indices[:-1])  # .conj().T

        dist = self._compute_distance(training_data, variables[:nb_prototypes],
                                      psis)  # change dist function ?
        # dist = cdist(training_data, prototypes, 'sqeuclidean')
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]

        if sum(self.regularization_) > 0:
            def test(x):
                return np.log(np.linalg.det(x.dot(x.conj().T)))

            t = np.array([test(x) for x in psis])
            reg_term = self.regularization_ * t
            return np.vectorize(self.phi)(mu) - 1 / nb_samples * reg_term[
                pidxcorrect] - 1 / nb_samples * reg_term[pidxwrong]
        return np.vectorize(self.phi)(mu).sum(0)

    def _optimize(self, x, y, random_state):
        nb_prototypes, nb_features = self.w_.shape
        nb_classes = len(self.classes_)
        if not isinstance(self.classwise, bool):
            raise ValueError("classwise must be a boolean")
        if self.initialdim is None:
            if self.classwise:
                self.dim_ = nb_features * np.ones(nb_classes, dtype=np.int)
            else:
                self.dim_ = nb_features * np.ones(nb_prototypes, dtype=np.int)
        else:
            self.dim_ = validation.column_or_1d(self.initialdim)
            if self.dim_.size == 1:
                if self.classwise:
                    self.dim_ = self.dim_[0] * np.ones(nb_classes,
                                                       dtype=np.int)
                else:
                    self.dim_ = self.dim_[0] * np.ones(nb_prototypes,
                                                       dtype=np.int)
            elif self.classwise and self.dim_.size != nb_classes:
                raise ValueError("dim length must be number of classes")
            elif self.dim_.size != nb_prototypes:
                raise ValueError("dim length must be number of prototypes")
            if self.dim_.min() <= 0:
                raise ValueError("dim must be a list of positive ints")

        # initialize psis (psis is list of arrays)
        if self.initial_matrices is None:
            self.omegas_ = []
            for d in self.dim_:
                self.omegas_.append(
                    random_state.rand(d, nb_features) * 2.0 - 1.0)
        else:
            if not isinstance(self.initial_matrices, list):
                raise ValueError("initial matrices must be a list")
            self.omegas_ = list(map(lambda v: validation.check_array(v),
                                    self.initial_matrices))
            if self.classwise and len(self.omegas_) != nb_classes:
                raise ValueError("length of matrices wrong\n"
                                 "found=%d\n"
                                 "expected=%d" % (
                                     len(self.omegas_), nb_classes))
            elif len(self.omegas_) != nb_prototypes:
                raise ValueError("length of matrices wrong\n"
                                 "found=%d\n"
                                 "expected=%d" % (
                                     len(self.omegas_), nb_classes))
            elif any(self.omegas_[i].shape != (self.dim_[i], nb_features)
                     for i in range(len(self.omegas_))):
                raise ValueError(
                    "each matrix must have shape (%d,dim)" % nb_features)

        if isinstance(self.regularization, float):
            if self.regularization < 0:
                raise ValueError('regularization must be a positive float')
            self.regularization_ = np.repeat(self.regularization,
                                             len(self.omegas_))
        else:
            self.regularization_ = validation.column_or_1d(self.regularization)
            if self.classwise:
                if self.regularization_.size != nb_classes:
                    raise ValueError(
                        "length of regularization must be number of classes")
            else:
                if self.regularization_.size != self.w_.shape[0]:
                    raise ValueError(
                        "length of regularization "
                        "must be number of prototypes")

        variables = np.append(self.w_, np.concatenate(self.omegas_), axis=0)
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        res = minimize(
            fun=lambda vs: self._f(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._g(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=1, lr_relevances=0, random_state=random_state),
            method='L-BFGS-B',
            x0=variables, options={'disp': self.display, 'gtol': self.gtol,
                                   'maxiter': self.max_iter})
        n_iter = res.nit
        res = minimize(
            fun=lambda vs: self._f(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._g(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=0, lr_relevances=1, random_state=random_state),
            method='L-BFGS-B',
            x0=res.x, options={'disp': self.display, 'gtol': self.gtol,
                               'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        res = minimize(
            fun=lambda vs: self._f(
                vs, x, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._g(
                vs, x, label_equals_prototype=label_equals_prototype,
                lr_prototypes=1, lr_relevances=1, random_state=random_state),
            method='L-BFGS-B',
            x0=res.x, options={'disp': self.display, 'gtol': self.gtol,
                               'maxiter': self.max_iter})
        n_iter = max(n_iter, res.nit)
        out = res.x.reshape(res.x.size // nb_features, nb_features)
        self.w_ = out[:nb_prototypes]
        indices = []
        for i in range(len(self.dim_)):
            indices.append(sum(self.dim_[:i + 1]))
        self.omegas_ = np.split(out[nb_prototypes:], indices[:-1])  # .conj().T
        self.n_iter_ = n_iter

    def _compute_distance(self, x, w=None, psis=None):
        if w is None:
            w = self.w_
        if psis is None:
            psis = self.omegas_
        nb_samples = x.shape[0]
        if len(w.shape) == 1:
            nb_prototypes = 1
        else:
            nb_prototypes = w.shape[0]
        distance = np.zeros([nb_prototypes, nb_samples])
        if len(psis) == nb_prototypes:
            for i in range(nb_prototypes):
                distance[i] = np.sum(np.dot(x - w[i], psis[i].conj().T) ** 2,
                                     1)
            return np.transpose(distance)
        for i in range(nb_prototypes):
            matrix_idx = np.where(self.classes_ == self.c_w_[i])[0][0]
            distance[i] = np.sum(
                np.dot(x - w[i], psis[matrix_idx].conj().T) ** 2, 1)
        return np.transpose(distance)

    def project(self, x, prototype_idx, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of the
        prototype specified by prototype_idx to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        prototype_idx : int
          index of the prototype
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        nb_prototypes = self.w_.shape[0]
        if len(self.omegas_) != nb_prototypes \
                or self.prototypes_per_class != 1:
            print('project only possible with classwise relevance matrix')
        # y = self.predict(X)
        v, u = np.linalg.eig(
            self.omegas_[prototype_idx].T.dot(self.omegas_[prototype_idx]))
        idx = v.argsort()[::-1]
        if print_variance_covered:
            print('variance coverd by projection:',
                  v[idx][:dims].sum() / v.sum() * 100)
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))

from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.utils import validation

def plot2d(model, x, y, figure, title=""):
    """
    Projects the input data to two dimensions and plots it. The projection is
    done using the relevances of the given glvq model.

    Parameters
    ----------
    model : GlvqModel that has relevances
        (GrlvqModel,GmlvqModel,LgmlvqModel)
    x : array-like, shape = [n_samples, n_features]
        Input data
    y : array, shape = [n_samples]
        Input data target
    figure : int
        the figure to plot on
    title : str, optional
        the title to use, optional
    """
    x, y = validation.check_X_y(x, y)
    dim = 2
    f = plt.figure(figure)
    f.suptitle(title)
    pred = model.predict(x)

    if hasattr(model, 'omegas_'):
        nb_prototype = model.w_.shape[0]

        d = sorted([(model._compute_distance(x[y == model.c_w_[i]],
                                             model.w_[i]).sum(), i) for i in
                    range(nb_prototype)], key=itemgetter(0))
        idxs = list(map(itemgetter(1), d))
        for i in idxs:
            x_p = model.project(x, i, dim, print_variance_covered=True)
            w_p = model.project(model.w_[i], i, dim)
            ax = f.add_subplot(1, nb_prototype, idxs.index(i) + 2)
            ax.scatter(x_p[:, 0], x_p[:, 1], c=_to_tango_colors(y, 0),
                       alpha=0.2)
            # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
            ax.scatter(w_p[0], w_p[1],
                       c=_tango_color('aluminium', 5), marker='D')
            ax.scatter(w_p[0], w_p[1],
                       c=_tango_color(i, 0), marker='.')
            ax.axis('equal')
    else:
        ax = f.add_subplot(121)
        ax.scatter(x[:, 0], x[:, 1], c=_to_tango_colors(y), alpha=0.5)
        ax.scatter(x[:, 0], x[:, 1], c=_to_tango_colors(pred), marker='.')
        ax.scatter(model.w_[:, 0], model.w_[:, 1],
                   c=_tango_color('aluminium', 5), marker='D')
        ax.scatter(model.w_[:, 0], model.w_[:, 1],
                   c=_to_tango_colors(model.c_w_, 0), marker='.')
        ax.axis('equal')
        x_p = model.project(x, dim, print_variance_covered=True)
        w_p = model.project(model.w_, dim)

        ax = f.add_subplot(122)
        ax.scatter(x_p[:, 0], x_p[:, 1], c=_to_tango_colors(y, 0), alpha=0.5)
        # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
        ax.scatter(w_p[:, 0], w_p[:, 1],
                   c=_tango_color('aluminium', 5), marker='D')
        ax.scatter(w_p[:, 0], w_p[:, 1], s=60,
                   c=_to_tango_colors(model.c_w_, 0), marker='.')
        ax.axis('equal')
    f.show()


colors = {
    "skyblue": ['#729fcf', '#3465a4', '#204a87'],
    "scarletred": ['#ef2929', '#cc0000', '#a40000'],
    "orange": ['#fcaf3e', '#f57900', '#ce5c00'],
    "plum": ['#ad7fa8', '#75507b', '#5c3566'],
    "chameleon": ['#8ae234', '#73d216', '#4e9a06'],
    "butter": ['#fce94f', 'edd400', '#c4a000'],
    "chocolate": ['#e9b96e', '#c17d11', '#8f5902'],
    "aluminium": ['#eeeeec', '#d3d7cf', '#babdb6', '#888a85', '#555753',
                  '#2e3436']
}

color_names = list(colors.keys())


def _tango_color(name, brightness=0):
    if type(name) is int:
        if name >= len(color_names):
            name = name % len(color_names)
        name = color_names[name]
    if name in colors:
        return colors[name][brightness]
    else:
        raise ValueError('{} is not a valid color'.format(name))


def _to_tango_colors(elems, brightness=0):
    elem_set = list(set(elems))
    return [_tango_color(elem_set.index(e), brightness) for e in elems]
