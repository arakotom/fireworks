#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:16:53 2019

"""
import numpy as np
from time import process_time as time
from numpy.linalg import norm
from scipy.sparse import csc_matrix as spmatrix, isspmatrix
from utils import generate_random_gaussian


def current_cost(X, y, w,penalty=None):
    """Compute the following objective function.

    min_w 0.5 || y - X w||_2^2 +  \sum_k pen(|w_k|),

    where pen is then nonconvex regularizer.
    """
    normres2 = norm(y - X.dot(w))**2
    return 0.5 * normres2 + penalty.value(w)


def GIST(X, y, penalty, tol=1e-3, max_iter=50000, w_init=None, 
         tmax=1e10,eta=1.5, sigma=0.1,verbose=False):
    """
    
    Solve the following problem.

    min_w 0.5 || y - Xw||_2^2 + \sum_k pen(|w_k|),
    
    where penalty is a non-convex penalty using the algorithm proposed by Gong 
    et al "A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
        Regularized Optimization Problems".
    https://proceedings.mlr.press/v28/gong13a.html

    Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
            
        y : array, shape (n_samples,)
            Target values.
        
        penalty : instance of Penalty class,
            Penalty used in the model.
        
        w_init : array, shape (n_features,) optional
            Coefficient vector.

        max_iter : int, optional
            The maximum number of iterations
        tol : float, optional
            The tolerance for the optimization.

        sigma : float, optional
              parameter in the line-search criterion
              default : 0.1
        eta : float, optional
            multiplicative term of the inverse stepsize in backtracking 
              default : 1.5
        tmax : float, optional 
              maximum value for the inverse stepsize in backtracking 
              default : 1e10
        verbose : bool or int, optional
            Amount of verbosity. False is silent.    
 


    Returns
    -------
  
    w : array, shape (n_feature)
        Coefficients 
    stop_crit : float
        Value of stopping criterion at convergence
    """


    n_features = X.shape[1]
    all_feats = np.arange(n_features)
  
    if w_init is None:
        wp = np.zeros(n_features,dtype=np.float64)
    else:
        wp = w_init
    cout = []
    coutnew = current_cost(X, y, wp,penalty=penalty)
    wp_aux =  np.zeros(n_features,dtype=np.float64)
    not_converged = False
    for i in range(max_iter):
        t = 1
        #----------------------- gradient update  ------------------------
        grad = - X.T.dot(y - X.dot(wp))
        for j in range(n_features):
            wp_aux[j] = penalty.prox_1d(wp[j] - grad[j]/t,1/t,0)
        #wp_aux = penalty.prox_1d(wp - grad/t,1/t,0)
        
        #----------------------- backtracking stepsize ------------------------
        coutold = coutnew
        coutnew = current_cost(X, y, wp_aux,penalty=penalty)
        while coutnew - coutold > - sigma / 2 * t * norm(wp - wp_aux)**2:
            t = t * eta
            #wp_aux = penalty.prox_1d(wp - grad/t,1/t,0)
            for j in range(n_features):
                wp_aux[j] = penalty.prox_1d(wp[j] - grad[j]/t,1/t,0)
            coutnew = current_cost(X, y, wp_aux,penalty=penalty)

            if t > tmax:
                print('Not converging. steps size too small!', t, tol)
                not_converged = True
                break
        cout.append(coutnew)
        wp = wp_aux.copy()
        #------------------- Testing optimality ------------------------------      
        opt = penalty.subdiff_distance(wp, grad, all_feats)
        stop_crit = np.max(opt)
        if stop_crit < tol:
            break
        
        if verbose:
            if i%20== 0:
                print('------------------------')
                print('|  GIST |  dist to opt    |')
                print('-----------------------')
                print(f" {i:}   {stop_crit:2.7f}")

            else:
                print(f" {i:}   {stop_crit:2.7f}")


        if not_converged:
             break

            
    return wp, stop_crit


def BCD_noncvxlasso(X, y, penalty, max_iter=50000, tol=1e-6,
                        w_init=None,  verbose=False):
    """
    
    Solve the following problem.
    
    min_w 0.5 || y - Xw||_2^2 + \sum_k pen(|w_k|),
    
    where penalty is a non-convex penalty using a coordinate descent approach
    based on a 1d proximal operator of the penalty
    
    Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
            
        y : array, shape (n_samples,)
            Target values.
        
        penalty : instance of Penalty class,
            Penalty used in the model.
        
        w_init : array, shape (n_features,) optional
            Coefficient vector.
    
        max_iter : int, optional
                The maximum number of iterations
                default : 50000
        tol : float, optional
            The tolerance for the optimization.
            default : 1e-6
        verbose : bool or int, optional
            Amount of verbosity. False is silent.    
    
    
    
    Returns
    -------
    
    w : array, shape (n_feature)
        Coefficients 
    stop_crit : float
        Value of stopping criterion at convergence
    """

    
    n_features = X.shape[1]
    if w_init is None:
        w = np.zeros(n_features)
    else:
        w = w_init
    all_feats = np.arange(n_features)

    i = 0
    rho = y - X.dot(w)
    while i < max_iter:
        for j in np.random.permutation(n_features) : 
            xj = X[:, j]
            s = rho + xj * w[j]
            xts = xj.T.dot(s)

            # do a prox gradient step 
            stepsize = 1/np.sqrt((xj.T@xj)) # lipschitz cte at xt 
            grads = - xts + w[j]/(stepsize**2)
            wp = penalty.prox_1d(w[j] - stepsize*grads ,stepsize=stepsize,j=0)


            rho = rho - xj * (wp - w[j])
            w[j] = wp 
        
        # --------------- check optimality ---------------------
        grad = -X.T.dot(y - X.dot(w))
        opt = penalty.subdiff_distance(w, grad, all_feats)
        stop_crit = np.max(opt)
        if stop_crit < tol:
            break
        
        if verbose:
            if i%20== 0:
                print('------------------------')
                print('|  CD |  dist to opt    |')
                print('-----------------------')
                print(f" {i:}   {stop_crit:2.7f}")

            else:
                print(f" {i:}   {stop_crit:2.7f}")
        # if compute_cost:
        #     cost.append(current_cost(X, y, w, penalty))
        i += 1

    return w, stop_crit
    


def act_noncvxlasso(X, y, penalty, method='BCD', max_iter=1000, tol=1e-6,
                    w_init=None, nb_feat_init=10, nb_feat_2_add=10,
                    max_iter_inner=50000,
                    tol_inner=1e-3,
                    pruning=False,
                    verbose=False):
    """
    
    Solve the following problem.
    
    min_w 0.5 || y - Xw||_2^2 + \sum_k pen(|w_k|),
    
    where penalty is a non-convex penalty using a active set (working set) strategy
    by solving iteratively a smaller inner problem using either a GIST or BCD algorithm
    
    Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
            
        y : array, shape (n_samples,)
            Target values.
        
        penalty : instance of Penalty class,
            Penalty used in the model.
        
        method : str, optional
            the algorithm to use in the inner solver
            default : BCD (coordinate descent)
        
        w_init : array, shape (n_features,) optional
            Coefficient vector.
            default : the algorithm starts with the 'nb_feat_init' features the most
            correlated by the target y 
            
        max_iter : int, optional
            The maximum number of iterations of the outer problem. it is also the
            maximum of time some features are added to the working set
            
        tol : float, optional
            The tolerance for the optimization on the full problem
            default: 1e-6
        
        nb_feat_init : int, optional
            The number of features to use in the first working set
            
        nb_feat_2_add : in, optional
            The number of features to add to the active set at each iteration 
            the strategy is to add those that are the most correlated to the 
            residual
            default : 10
            
        max_iter_inner : int, optional
            The maximum number of iterations of the outer problem. it is also the
            maximum of time some features are added to the working set
            default : 50000
            
        tol_inner : float, optional
            The tolerance for the optimization on the full problem
            default : 1e-3
            
        pruning : bool, optional
            Prune the working set of the zero elements. This makes the working
            set smaller and thus accelerates the approach but breaks convergence 
            guarantee.
        
        verbose : bool or int, optional
            Amount of verbosity. False is silent.    
    
    
    
    Returns
    -------
    
    w : array, shape (n_feature)
        Coefficients 
    stop_crit : float
        Value of stopping criterion at convergence
        
    """   
    n_samples, n_features = X.shape
    
    # Select first active set corresponding to largest correlated features
    if w_init is None:
        ind = np.argsort(-abs(X.T.dot(y)))[:nb_feat_init]
    else:
        ind = np.where(np.abs(w_init))[0]

    all_feats = np.arange(n_features)

    w = np.zeros((n_features))
    for i in range(max_iter):
        Xaux = X[:, ind]
        if i == 0 and w_init is not None:
            w_init = w_init[ind]
        if isspmatrix(Xaux):
            Xaux = Xaux.toarray()
        if method == 'BCD':
            w_inter,iter_cost = BCD_noncvxlasso(Xaux, y, penalty,
                                      max_iter=max_iter_inner,
                                      tol=tol_inner,
                                      w_init=w_init)
        elif method == 'GIST':
            w_inter, _ = GIST(Xaux, y,penalty=penalty, eta=1.5, sigma=0.1, tol=tol_inner,
                           max_iter=max_iter_inner, w_init=w_init)
            
        # pruning
        if pruning:
            w_inter, ind, Xaux= w_inter[w_inter.nonzero()[0]], ind[w_inter.nonzero()[0]], Xaux[:,w_inter.nonzero()[0]]

            
        res = (y - Xaux.dot(w_inter))
        grad = - X.T.dot(res)

        
        # --------------- check optimality ---------------------
        opt = penalty.subdiff_distance(w, grad, all_feats)
        stop_crit = np.max(opt)
        wssize = Xaux.shape[1]

        if verbose:
            if i%20== 0:
                print('----------------------------------')
                print('|  MVC |  dist to opt   |   ws size |')
                print('---------------------------------')
                print(f" {i:}   {stop_crit:2.7f}   {wssize:}")

            else:
                print(f" {i:}   {stop_crit:2.7f}   {wssize:}")


        if stop_crit < tol:
            break
        else:
            w[ind] = np.zeros(len(ind))
            candidate = np.argsort(-abs(grad))
            nb_add = 0
            for cand in candidate:
                if cand not in ind:
                    ind = np.hstack([ind, cand])
                    nb_add += 1
                    if nb_add == nb_feat_2_add:
                        break

            w_init = np.hstack([w_inter, np.zeros(nb_add)])
    
    # ------------------  gathering vectors and outputs ---------------------- 
    w_act = np.zeros(n_features)
    w_act[ind] = w_inter.copy()

    return w_act,stop_crit

def fireworks_noncvxlasso(X, y, penalty,method='BCD', max_iter=1000,
                          tol=1e-6, max_iter_inner=50000, w_init=None,
                          nb_feat_2_add=10, tol_inner=1e-3,
                          nb_feat_init=10,inexact=False,verbose=False):
    """
   
    Solve the following problem.
   
    min_w 0.5 || y - Xw||_2^2 + \sum_k pen(|w_k|),
   
    where penalty is a non-convex penalty using a active set (working set) strategy
    by solving iteratively a smaller inner problem using either a GIST or BCD algorithm
    the working set is built using the FireWorks strategy which ensures convergence
    of the algorithm.
   
    https://proceedings.mlr.press/v151/rakotomamonjy22a.html
   
    Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
           we assume that columns are norm 1
           
        y : array, shape (n_samples,)
            Target values.
       
        penalty : instance of Penalty class,
            Penalty used in the model.
       
        method : str, optional
            the algorithm to use in the inner solver
            default : BCD (coordinate descent)
       
        w_init : array, shape (n_features,) optional
            Coefficient vector.
            default : the algorithm starts with the 'nb_feat_init' features the most
            correlated by the target y 
           
        max_iter : int, optional
            The maximum number of iterations of the outer problem. it is also the
            maximum of time some features are added to the working set
            default : 1000
           
        tol : float, optional
            The tolerance for the optimization on the full problem
            default: 1e-6
       
        nb_feat_init : int, optional
            The number of features to use in the first working set
            default : 10
           
        nb_feat_2_add : in, optional
            The number of features to add to the active set at each iteration 
            the strategy is to add those that are the most correlated to the 
            residual
            default : 10
           
        max_iter_inner : int, optional
            The maximum number of iterations of the outer problem. it is also the
            maximum of time some features are added to the working set
            default : 50000
           
        tol_inner : float, optional
            The tolerance for the optimization on the full problem
            default : 1e-3
           
        inexact : bool, optional
            use inexact inner solver with initial max_iter_inner = 50 doubling
            at each iteration
            default : False 
       
        verbose : bool or int, optional
            Amount of verbosity. False is silent.    
   
   
   
    Returns
    -------
   
    w : array, shape (n_feature)
        Coefficients 
    stop_crit : float
        Value of stopping criterion at convergence
       
    """ 
    
    n_samples, n_features = X.shape
    all_feats = np.arange(n_features)
    penalty_subdiff_at_0 = 100.0 - penalty.subdiff_distance([0], [100.0], [0])
    if w_init is None:
        ind = np.argsort(-abs(X.T.dot(y)))[:nb_feat_init]
    else:
        ind = np.where(np.abs(w_init))[0]
        
    sk = np.zeros(n_samples)  # feasible dual
    w = np.zeros((n_features))
    if inexact:
        max_iter_inner=50
    
    for i in range(max_iter):
    
        #------------------------------ subproblem
        Xaux = X[:, ind]
        if i == 0 and w_init is not None:
            w_init = w_init[ind]
        if isspmatrix(Xaux):
            Xaux = Xaux.toarray()
        #---------------------- solving the inner problem --------------------
        if method == 'BCD':
            w_inter,iter_cost = BCD_noncvxlasso(Xaux, y, penalty,
                                      max_iter=max_iter_inner,
                                      tol=tol_inner, w_init=w_init
                                      )
        elif method == 'GIST':
            w_inter, iter_cost = GIST(Xaux, y,penalty,
                            eta=1.5, sigma=0.1, tol=tol_inner,
                            max_iter=max_iter_inner, w_init=w_init)
    
        
        # ------------------- Pruning the current solution --------------------
        ind_nnz = w_inter.nonzero()[0]
        w_inter, ind, Xaux= w_inter[ind_nnz], ind[ind_nnz], Xaux[:,ind_nnz]
       
        # -----  computing the current residual (dual) and gradient -----------
        rk = y - Xaux.dot(w_inter)
        grad = - X.T.dot(rk)
     
        
        #--------------- checking optimality of the full problem --------------
        w[ind] = w_inter.copy()
        opt = penalty.subdiff_distance(w, grad, all_feats)     
        w[ind] = 0
        stop_crit = np.max(opt)
        if stop_crit < tol or i == max_iter - 1:
            break
        wssize = Xaux.shape[1]

        if verbose:
            if i%20== 0:
                print('----------------------------------')
                print('|  FireWorks |  dist to opt   |   ws size |')
                print('---------------------------------')
                print(f" {i:}   {stop_crit:2.7f}   {wssize:}")
    
            else:
                print(f" {i:}   {stop_crit:2.7f}   {wssize:}")
    
    
        # ------  computing s_k  for updating the working set --------------
    
    
        Xt_rk = - grad  # X.T @ r_k
        Xt_sk = X.T.dot(sk)
      
        # -------------- looking for the best alpha using binary search
        alpha_left, alpha_right= 0,1
        while alpha_right - alpha_left > 1e-6:
            alpha_mid = (alpha_right  + alpha_left)/2
            score_mid = np.max(np.abs(alpha_mid * Xt_rk + (1 - alpha_mid) * Xt_sk)) - penalty_subdiff_at_0  
            if score_mid > 0:
                alpha_right = alpha_mid
            else:
                alpha_left = alpha_mid
                
        best_alpha = alpha_left
        # ------- computing s_k given alpha and the distance of sk to all hyperplanes
        sk = best_alpha * rk + (1 - best_alpha) * sk  # 
        dist = (penalty_subdiff_at_0 - np.abs(X.T.dot(sk)))  
        # ---------------------------------------------------------------------
    
    
        # update list of variables and remove duplicates
        candidate = np.argsort(dist)
        # add feature and remove duplicates
        nb_add = 0
        for cand in candidate:
            if cand not in ind:
                ind = np.hstack([ind, cand])
                nb_add += 1
                if nb_add == nb_feat_2_add:
                    break
                
    
        if inexact:
                max_iter_inner *=2
        
        w_init = np.hstack([w_inter, np.zeros(nb_add)])
    
    #------------ final_output and optimality test ---------------------------
    w_blitz = np.zeros(n_features)
    w_blitz[ind] = w_inter.copy()
    
    
    
    return w_blitz,stop_crit






#%%
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from ncvx_penalties import LSP, MCPenalty
    #np.random.seed(76)  # Fix seed for reproducibility / debugging.

    # the small toy dataset
    n_samples = 100
    n_features = 1000
    n_informative = 30
    method = 'BCD' 
    nb_feat_2_add = 30
    tol = 1e-3
    lbd = 0.05

    # toy data set options    
    sigma_bruit = 0.01
    X, y, w_opt = generate_random_gaussian(n_samples, n_features, n_informative,
                                          sigma_bruit)

    #  parameters for the penalty
    normX = np.linalg.norm(X, axis=0, ord=2)
    lbd = np.max(np.abs(np.dot(X.T,y)))*lbd
    
    
    # penalties  
    theta = 1
    penalty = LSP(lbd,theta)
    
    alpha = 10
    penalty= MCPenalty(lbd,alpha)
    

    
    tic = time()
    w_gist, _ = GIST(X, y,penalty,verbose=True)
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
    w_blitz, indice_blitz= fireworks_noncvxlasso(X, y,
                                            method=method,
                                            max_iter=500,
                                            tol=tol,
                                            max_iter_inner=10000,
                                            nb_feat_2_add=nb_feat_2_add,
                                            tol_inner=tol,inexact=False,
                                            verbose=True,penalty=penalty)
    toc_blitz = time() - tic

    
    


    # this may be a bit long if you increase feature
    tic = time()
    w_bcd, stop_crit   = BCD_noncvxlasso(X, y,tol=tol,penalty=penalty,verbose=True,max_iter=200)
    toc_bcd = time() - tic
    
    
    print(f"computational gain of FireWorks vs BCD: {toc_bcd/toc_blitz}")
    print(f"computational gain of FireWorks vs MVC: {toc_act/toc_blitz}")
    print(f"computational gain of FireWorks vs GIST: {toc_gist/toc_blitz}")

  