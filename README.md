# Convergent Working Set Algorithm for Lasso with non-convex regularizers

This is a numpy implementation of our paper "Convergent Working Set Algorithm for Lasso with non-convex regularizers"  appearead at AISTATS 2022.
Details of algorithm experimental results can be found in our following paper https://proceedings.mlr.press/v151/rakotomamonjy22a.html.


The examples.py file shows how the solver can be called and  plots an example of estimation for a small dataset.




````
@InProceedings{pmlr-v151-rakotomamonjy22a,
  title = 	 { Convergent Working Set Algorithm for Lasso with Non-Convex Sparse Regularizers },
  author =       {Rakotomamonjy, Alain and Flamary, R\'emi and Salmon, Joseph and Gasso, Gilles},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {5196--5211},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/rakotomamonjy22a/rakotomamonjy22a.pdf},
  url = 	 {https://proceedings.mlr.press/v151/rakotomamonjy22a.html},
  abstract = 	 { Non-convex sparse regularizers are common tools for learning with high-dimensional data. For accelerating convergence for Lasso problem involving those regularizers, a working set strategy addresses the optimization problem through an iterative algorithm by gradually incrementing the number of variables to optimize until the identification of the solution support. We propose in this paper the first Lasso working set algorithm for non-convex sparse regularizers with convergence guarantees. The algorithm, named FireWorks, is based on a non-convex reformulation of a recent duality-based approach and leverages on the geometry of the residuals. We provide theoretical guarantees showing that convergence is preserved even when the inner solver is inexact, under sufficient decay of the error across iterations. Experimental results demonstrate strong computational gain when using our working set strategy compared to full problem solvers for both block-coordinate descent or a proximal gradient solver. }
}

````
## Note

The structure of the penalty comes from the one of skglm (https://github.com/scikit-learn-contrib/skglm/).

