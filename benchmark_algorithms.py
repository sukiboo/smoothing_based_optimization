
import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dgs import DirectionalGaussianSmoothing
from adam import Adam
from rmsprop import RMSProp
from nag import NAG


sns.set_theme(style='darkgrid', palette='muted', font='monospace')

##import warnings
##warnings.filterwarnings('ignore')


class BenchmarkAlgorithms:

    def __init__(self, algos, dim, num_tests, random_seed):
        self.algos = algos
        self.dim = dim
        self.num_tests = num_tests
        np.random.seed(random_seed)
        self.random_seed_list = np.random.randint(1e+9, size=num_tests)
        num_iters = 100
        self.test_algorithms()

    def test_algorithms(self):
        '''generate and solve an optimization problem'''
        self.vals = {alg: [] for alg in self.algos}
        for alg, params in self.algos.items():
            for t in tqdm.trange(self.num_tests, desc=f'Testing {alg:>7s}', ascii=True):
                np.random.seed(self.random_seed_list[t])
                self.generate_function()
                self.sample_initial_guess()
                self.run_minimization(alg, params)

    def generate_function(self):
        '''generate optimization target by perturbing sphere function'''
        dilation_sphere = 10 * np.random.rand(self.dim)
        dilation_noise = 10 * np.random.randn(self.dim)
        target = lambda x: np.sum(dilation_sphere * x**2)
        ##noise = lambda x: np.sin(np.sum(np.abs(dilation_noise * x)))
        noise = lambda x: 0.000001 * np.random.randn()
        self.fun = lambda x: target(x) + noise(x)

    def sample_initial_guess(self):
        '''randomly sample an initial guess from the domain'''
        domain = np.array([[-1,1]] * self.dim)
        self.x0 = np.random.rand(self.dim) * (domain[:,1] - domain[:,0]) + domain[:,0]

    def run_minimization(self, alg, params):
        '''perform minimization with a given algorithm'''
        if alg in ['DGS']:
            vals = self.sbo_minimize(alg, params)
        else:
            vals = self.scipy_minimize(alg, params)
        self.vals[alg].append(vals)

    def sbo_minimize(self, method, params):
        '''minimize target function with smoothing-based optimization'''
        if method == 'DGS':
            sbo = DirectionalGaussianSmoothing(params)
            sbo.minimize(self.fun, self.x0, plot=False, disable_pbar=True)
        return sbo.vals

    def scipy_minimize(self, method, params):
        '''minimize target function with a given scipy method'''
        vals = [self.fun(self.x0)]
        # TODO: fix this
        if method == 'Adam':
            method = Adam
        elif method == 'RMSProp':
            method = RMSProp
        elif method == 'NAG':
            method = NAG
        scipy.optimize.minimize(self.fun, self.x0, method=method, options=params,
                                callback=lambda x: vals.append(self.fun(x)))
        return vals

    def visualize(self, percentile=(.25,.5,.75)):
        '''visualize optimization results'''
        fig, ax = plt.subplots(figsize=(8,5))
        for alg in self.algos:
            alg_vals = pd.DataFrame(self.vals[alg])
            alg_min, alg_avg, alg_max = alg_vals.quantile(percentile, axis=0).values
            plt.plot(alg_avg, linewidth=3, label=alg)
            plt.fill_between(range(alg_min.size), alg_min, alg_max, alpha=.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./images/{self.dim}_{self.num_tests}.png', dpi=300, format='png')
        plt.show()


if __name__ == '__main__':

    dim = 100
    num_tests = 10
    random_seed = 0
    num_iters = 100

    algos = {
            'DGS': {'learning_rate': .01, 'sigma': .1, 'quad_points': 7, 'num_iters': num_iters},
           'Adam': {'learning_rate': .1, 'maxiter': num_iters},
        'RMSProp': {'learning_rate': .1, 'maxiter': num_iters},
            ##'NAG': {'learning_rate': .001, 'maxiter': num_iters},
           'BFGS': {'maxiter': num_iters},
             'CG': {'maxiter': num_iters},
         'Powell': {'maxiter': num_iters},
        }

    # run benchmarking
    ba = BenchmarkAlgorithms(algos, dim, num_tests, random_seed)
    ba.visualize()

