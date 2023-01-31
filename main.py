
import numpy as np

from dgs import DirectionalGaussianSmoothing


if __name__ == '__main__':

    # choose target function and initial guess
    dim = 100
    x0 = np.random.randn(dim)
    fun = lambda x: -20 * np.exp(-.2 * np.sqrt(np.sum(x**2) / dim))\
            - np.exp(np.sum(np.cos(2*np.pi*x)) / dim) + 20 + np.exp(1)

    # setup smoothing-based optimization
    params = {'sigma': .5, 'learning_rate': .1, 'quad_points': 5, 'num_iters': 500}
    dgs = DirectionalGaussianSmoothing(params)
    dgs.minimize(fun, x0, plot=True)

