
import numpy as np
import scipy
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dgs import DirectionalGaussianSmoothing

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


if __name__ == '__main__':

    # target function and noise
    target = lambda x: np.sum(x**2)
    noise = lambda x: np.sin(np.sum(10*np.abs(x)))
    dim = 100

    # optimization setting
    fun = lambda x: target(x) + noise(x)
    x0 = np.random.randn(dim)
    num_iters = 100

    # perform BFGS+DGS optimization
    x = x0.copy()
    vals = [fun(x)]
    for _ in tqdm.trange(num_iters//2, ascii=True):

        # step of BFGS
        bfgs_res = scipy.optimize.minimize(fun, x, method='BFGS', options={'maxiter': 1})
        x = bfgs_res.x
        vals.append(fun(x))

        # step of DGS
        dgs_params = {'sigma': .1, 'learning_rate': .01, 'quad_points': 7, 'num_iters': 1}
        dgs = DirectionalGaussianSmoothing(dgs_params)
        dgs.minimize(fun, x, plot=False, disable_pbar=True)
        x = dgs.x
        vals.append(fun(x))

    # perform BFGS optimization
    bfgs_vals = [fun(x0)]
    bfgs_res = scipy.optimize.minimize(fun, x0, method='BFGS', options={'maxiter': num_iters},
                                       callback=lambda x: bfgs_vals.append(fun(x)))

    # perform DGS optimization
    dgs_params = {'sigma': .1, 'learning_rate': .01, 'quad_points': 7, 'num_iters': num_iters}
    dgs = DirectionalGaussianSmoothing(dgs_params)
    dgs.minimize(fun, x0, plot=False)
    dgs_vals = dgs.vals


    # plot function values
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(vals, linewidth=3, label='BFGS+DGS')
    ax.plot(bfgs_vals, linewidth=3, label='BFGS')
    ax.plot(dgs_vals, linewidth=3, label='DGS')
    plt.legend()
    plt.tight_layout()
    plt.show()

