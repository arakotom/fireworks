#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:40:16 2022

@author: alain
"""

from base_penalty import BasePenalty
import numpy as np
from numba.types import bool_

#@jitclass(spec_MCP)
class MCPenalty(BasePenalty):
    """Minimax Concave Penalty (MCP), a non-convex sparse penalty.
    Notes
    -----
    With x >= 0
    pen(x) =
    alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
    gamma * alpha 2 / 2           if x > gamma * alpha
    value = sum_{j=1}^{n_features} pen(abs(w_j))
    """

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def value(self, w):
        """Compute the value of MCP."""
        s0 = np.abs(w) < self.gamma * self.alpha
        value = np.full_like(w, self.gamma * self.alpha ** 2 / 2.)
        value[s0] = self.alpha * np.abs(w[s0]) - w[s0]**2 / (2 * self.gamma)
        return np.sum(value)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of MCP."""
        tau = self.alpha * stepsize
        g = self.gamma / stepsize  # what does g stand for ?
        if np.abs(value) <= tau:
            return 0.
        if np.abs(value) > g * tau:
            return value
        return np.sign(value) * (np.abs(value) - tau) / (1. - 1./g)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of -grad to alpha * [-1, 1]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            elif np.abs(w[j]) < self.alpha * self.gamma:
                # distance of -grad_j to (alpha - abs(w[j])/gamma) * sign(w[j])
                subdiff_dist[idx] = np.abs(
                    grad[idx] + self.alpha * np.sign(w[j])
                    - w[j] / self.gamma)
            else:
                # distance of grad to 0
                subdiff_dist[idx] = np.abs(grad[idx])
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))



class LSP(BasePenalty):
    """ LogSum Penalty, a non-convex sparse penalty.
         
            \sum_k lbd * log(1 + |w_k|/theta)
    
    """

    def __init__(self, lbd,theta):
        self.theta = theta
        self.lbd = lbd
    def value(self, w):
        """Compute the value of LSP."""
        return self.lbd*np.sum(sum(np.log(1 + np.abs(w) / self.theta)))

    def prox_1d(self, w,stepsize,j):
        """Compute the proximal operator of MCP."""
        lbd= self.lbd*stepsize
        
        absw = np.abs(w)
        z = absw - self.theta
        v = z * z - 4 * (lbd - absw * self.theta)
        v = np.maximum(v, 0)
    
        sqrtv = np.sqrt(v)
        w1 = np.maximum((z + sqrtv) / 2, 0)
        w2 = np.maximum((z - sqrtv) / 2, 0)
        # Evaluate the proximal at this solution
        y0 = 0.5 * w**2
        y1 = 0.5 * (w1 - absw)**2 + lbd * np.log(1 + w1 / self.theta)
        y2 = 0.5 * (w2 - absw)**2 + lbd * np.log(1 + w2 / self.theta)
    
        sel1 = (y1 < y2) & (y1 < y0)
        sel2 = (y2 < y1) & (y2 < y0)
        wopt = w1 * sel1 + w2 * sel2
        return np.sign(w) * wopt
    # def prox_grad(self,w,grad,stepsize):
    #     return self.prox_1d(w - grad*stepsize,stepsize)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of -grad to alpha * [-1, 1]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.lbd/self.theta)
            else:
                subdiff_dist[idx] = np.abs(
                    grad[idx] + self.lbd * np.sign(w[j])/(self.theta + np.abs(w[j])))

        return subdiff_dist
    def subdiff_at_0(self):
        return self.lbd/self.theta
    def check_opt(self,w,grad,tol):
        #verbose=True
        absw = np.abs(w)
        ind_nz = absw.nonzero()[0]
        ind_zero = [i for i in range(w.shape[0]) if i not in ind_nz]
    
        # ind_zero = np.where(absw < tol_val)[0]
        #print(len(ind_zero))
        if len(ind_zero) > 0:
            opt_ind_zero = np.all(abs(grad[ind_zero]) <= (self.lbd / self.theta + tol))   
            test_val = np.max(np.abs(grad[ind_zero]) - self.lbd)
    
            #if verbose:
            #    print('Opt_Z', test_val, opt_ind_zero)
    
        else:
            opt_ind_zero = True
            test_val = 0
        if ind_nz.shape[0] > 0:
    
            test = abs(grad[ind_nz] +
                       self.lbd * np.sign(w[ind_nz]) / (self.theta + np.abs(w[ind_nz])))
    
            opt_ind_nz = np.all(test < tol)
            #if verbose:
            #    print('Opt_NZ', np.max(test), opt_ind_nz)
    
        else:
            opt_ind_nz = True
    
        return (opt_ind_zero and opt_ind_nz), test_val

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))


if __name__ == "__main__":
    w = np.array([0,-2,3])
    lsp = LSP(1,1)
    print(lsp.value(w))
    print(lsp.prox_1d(w))
    

