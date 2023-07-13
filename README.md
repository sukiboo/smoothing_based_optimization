# Smoothing Based Optimization

~~A bunch of scripts to play with smoothing-based optimization techniques.~~
Actually, we're now making it a legitimate paper.

Benchmark different types of optimization algorithms on various test functions.
Experiment results will be saved to the `./images/` folder that will be created.

### File Organization

Currently includes the following files:
* `target_functions.py` -- set up a target function and sample an initial guess
* `benchmark_algorithms.py` -- compare optimization algoritms on series of tests
* `hyperparameter_search.py` -- test different hyperparameters for each function and algorithm

Implemented algorithms (in `./algorithms/`):
* `adam.py` -- Adam optimizer
* `rmsprop.py` -- RMSProp optimizer
* `nag.py` -- Nesterov's Accelerated Gradient Descent
* `dgs.py` -- Directional Gaussian Smoothing
* `adgs.py` -- DGS with exponential decay on sigma
* `lsgd.py` -- Laplacian Smooth Gradient Descent
* `mcgs.py` -- Monte Carlo Gaussian Smoothing
* `slgh.py` -- Single Loop Gaussian Homotopy (Andrew why are you doing this to me)

Old files that are now in `./extra_scripts/`:
* `main.py` -- use to launch numerical optimization
* `main_tf.py` -- use to launch network training (very slow)
* `bfgs_dgs.py` -- define BFGS+DGS algorithm
* `sbfgs.py` -- define Smoothed BFGS (it doesn't really work though)
* `smoothing_visualization.py` -- create interactive smoothing plot

