# Smoothing Based Optimization

Source code for the numerical results presented in the paper "[Gaussian smoothing gradient descent for minimizing high-dimensional non-convex functions](https://arxiv.org/abs/2311.00521)".

Benchmark different types of optimization algorithms on various test functions.
Experiment results will be saved to the `./images/` folder that will be created.

![opt](https://github.com/sukiboo/smoothing_based_optimization/assets/38059493/21bbdf76-3242-4c4f-9a48-03d1aa87b46d)

### File Organization

Currently includes the following files:
* `target_functions.py` -- set up a target function and sample an initial guess
* `benchmark_algorithms.py` -- compare optimization algoritms on series of tests
* `hyperparameter_search.py` -- test different hyperparameters for each function and algorithm
* `visualization.py` -- plot optimization values from the logged data

Implemented algorithms (in `./algorithms/`):
* `adam.py` -- Adam optimizer
* `rmsprop.py` -- RMSProp optimizer
* `nag.py` -- Nesterov's Accelerated Gradient Descent
* `dgs.py` -- Directional Gaussian Smoothing
* `adgs.py` -- DGS with exponential decay on sigma
* `lsgd.py` -- Laplacian Smooth Gradient Descent
* `mcgs.py` -- Monte Carlo Gaussian Smoothing
* `slgh.py` -- Single Loop Gaussian Homotopy

Old files that are now in `./extra_scripts/`:
* `main.py` -- use to launch numerical optimization
* `main_tf.py` -- use to launch network training (very slow)
* `bfgs_dgs.py` -- define BFGS+DGS algorithm
* `sbfgs.py` -- define Smoothed BFGS (it doesn't really work though)
* `smoothing_visualization.py` -- create interactive smoothing plot

