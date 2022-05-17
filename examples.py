#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:52:30 2022

@author: alain
"""

import numpy as np
from time import process_time as time
import matplotlib.pyplot as plt

from fireworks.utils import generate_random_gaussian
from fireworks.ncvx_penalties import LSP, MCPenalty
from fireworks.ncvx_solvers import GIST, BCD_noncvxlasso, act_noncvxlasso, fireworks_noncvxlasso

np.random.seed(76)  # Fix seed for reproducibility / debugging.

# the small toy dataset
n_samples = 100
n_features = 1000
n_informative = 10
method = 'BCD'  # method for inner solver -- you can choose either BCD or GIST
nb_feat_2_add = 30  # number of features to add at each iteration
tol = 1e-3          # tolerance on optimality condition
lbd = 0.1           # regularizer strengh  max [X.T^y]*lbd

# toy data set options
sigma_bruit = 0.01
X, y, w_opt = generate_random_gaussian(n_samples, n_features, n_informative,
                                       sigma_bruit)

#  parameters for the penalty
normX = np.linalg.norm(X, axis=0, ord=2)
lbd = np.max(np.abs(np.dot(X.T, y)))*lbd


# choice of penalties  -- LogSum Penalty or MCP Penalty
theta = 1
penalty = LSP(lbd, theta)

# alpha = 10
# penalty= MCPenalty(lbd,alpha)


tic = time()
w_gist, _ = GIST(X, y, penalty, verbose=True)
toc_gist = time() - tic

tic = time()
w_act, indices, = act_noncvxlasso(X, y, penalty, method=method,
                                  max_iter=500, tol=tol,
                                  max_iter_inner=10000,
                                  nb_feat_2_add=nb_feat_2_add,
                                  tol_inner=tol,
                                  verbose=True)
toc_act = time() - tic

tic = time()
w_blitz, indice_blitz = fireworks_noncvxlasso(X, y,
                                              method=method,
                                              max_iter=500,
                                              tol=tol,
                                              max_iter_inner=10000,
                                              nb_feat_2_add=nb_feat_2_add,
                                              tol_inner=tol, inexact=False,
                                              verbose=True, penalty=penalty)
toc_blitz = time() - tic


# this may be a bit long if you increase feature
tic = time()
w_bcd, stop_crit = BCD_noncvxlasso(
    X, y, tol=tol, penalty=penalty, verbose=True, max_iter=200)
toc_bcd = time() - tic


print(f"computational gain of FireWorks vs BCD: {toc_bcd/toc_blitz}")
print(f"computational gain of FireWorks vs MVC: {toc_act/toc_blitz}")
print(f"computational gain of FireWorks vs GIST: {toc_gist/toc_blitz}")


plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(w_opt, label='True weights')
plt.ylabel('Amplitude')

plt.legend()
plt.subplot(2, 1, 2)
plt.plot(w_blitz, 'k', label='Fireworks estimation')
plt.xlabel('Feature Index')
plt.ylabel('Amplitude')
plt.legend()
