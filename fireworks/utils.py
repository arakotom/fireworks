#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:43:36 2022

@author: alain
"""

import numpy as np

def generate_random_gaussian(n_samples=50, n_features=100, n_informative=2,
                             sigma_bruit=0, scal=1, const=0):
    """Build the toy problem."""
    X = (np.random.randn(n_samples, n_features))*0.1 + const
    #X /= np.linalg.norm(X, axis=0)
    ind = np.random.permutation(n_features)
    wopt = np.zeros(n_features)
    aux = np.random.randn(n_informative)
    wopt[ind[0:n_informative]] = aux + 0.1 * np.sign(aux)
    y = X.dot(wopt) + np.random.randn(n_samples) * sigma_bruit

    return X, y, wopt




# def subdiff_concavelsp(w, lbd, theta):
#     """Compute the subgradient of the concave part.

#     pen is : |w| - log(|w| + theta) including the regularizer strenght lbd.
#     """
#     return lbd * (1 - 1 / (abs(w) + theta))


# def prox_l1(w, thresh):
#     """Soft-thresholding function."""
#     absw = np.abs(w)
#     return np.where((absw - thresh) > 0, (absw - thresh) * np.sign(w), 0)


# def reg_lsp(w, theta):
#     """Compute the regularizer objective value."""
#     return sum(np.log(1 + np.abs(w) / theta))
# def approx_lsp(w, theta):
#     """Vector of the linear approximation of log(1+w/theta)."""
#     return 1. / (np.abs(w) + theta)


# def prox_lsp(w, lbd, theta):
#     """Compute proximal operator of non-cvx regularizer (closed-form)."""
#     absw = np.abs(w)
#     z = absw - theta
#     v = z * z - 4 * (lbd - absw * theta)
#     v = np.maximum(v, 0)

#     sqrtv = np.sqrt(v)
#     w1 = np.maximum((z + sqrtv) / 2, 0)
#     w2 = np.maximum((z - sqrtv) / 2, 0)
#     # Evaluate the proximal at this solution
#     y0 = 0.5 * w**2
#     y1 = 0.5 * (w1 - absw)**2 + lbd * np.log(1 + w1 / theta)
#     y2 = 0.5 * (w2 - absw)**2 + lbd * np.log(1 + w2 / theta)

#     sel1 = (y1 < y2) & (y1 < y0)
#     sel2 = (y2 < y1) & (y2 < y0)
#     wopt = w1 * sel1 + w2 * sel2
#     return np.sign(w) * wopt

# def check_opt_lsp(grad, w, lbd, theta, tol=1e-3,  verbose=False):

#     """Evaluate first-order optimality conditions ."""

#     absw = np.abs(w)
#     ind_nz = absw.nonzero()[0]
#     ind_zero = [i for i in range(w.shape[0]) if i not in ind_nz]

#     # ind_zero = np.where(absw < tol_val)[0]
#     if len(ind_zero) > 0:
#         opt_ind_zero = np.all(abs(grad[ind_zero]) <= (lbd / theta + tol))   
#         test_val = np.max(np.abs(grad[ind_zero]) - lbd)

#         if verbose:
#             print('Opt_Z', test_val, opt_ind_zero)

#     else:
#         opt_ind_zero = True
#         test_val = 0
#     if ind_nz.shape[0] > 0:

#         test = abs(grad[ind_nz] +
#                    lbd * np.sign(w[ind_nz]) / (theta + np.abs(w[ind_nz])))

#         opt_ind_nz = np.all(test < tol)
#         if verbose:
#             print('Opt_NZ', np.max(test), opt_ind_nz)

#     else:
#         opt_ind_nz = True

#     return (opt_ind_zero and opt_ind_nz), test_val
