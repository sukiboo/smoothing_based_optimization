
import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from smoothing_based_optimization import SmoothingBasedOptimization

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class BenchmarkAlgorithms:

    def __init__(self, target, noise, dim):
        self.fun = lambda x: target(x) + noise(x)
        self.dim = dim

    def test(self, num_tests, random_seed):
        '''solve an optimization problem with various initial guesses'''
        self.setup_optimization_params(num_tests, random_seed)
        self.vals = {alg: [] for alg in self.algos}
        for alg in self.algos:
            for t in tqdm.trange(num_tests, desc=f'Testing {alg}', ascii=True):
                self.sample_initial_guess(t)
                self.run_minimization(alg)

    def setup_optimization_params(self, num_tests, random_seed):
        '''configure benchmarking parameters'''
        np.random.seed(random_seed)
        self.random_seed_list = np.random.randint(1e+9, size=num_tests)
        num_iters = 100
        self.algos = ['SBO', 'BFGS', 'CG', 'Powell', 'Nelder-Mead']
        self.sbo_params = {'sigma': .1, 'learning_rate': .01,
                           'quad_points': 7, 'num_iters': num_iters}
        self.scipy_params = {'maxiter': num_iters}

    def sample_initial_guess(self, t):
        '''randomly sample an initial guess'''
        np.random.seed(self.random_seed_list[t])
        self.x0 = np.random.randn(self.dim)

    def run_minimization(self, alg):
        '''perform minimization with a given algorithm'''
        if alg == 'SBO':
            vals = self.sbo_minimize()
        else:
            vals = self.scipy_minimize(alg)
        self.vals[alg].append(vals)

    def sbo_minimize(self):
        '''minimize target function with smoothing-based optimization'''
        sbo = SmoothingBasedOptimization(self.sbo_params)
        sbo.minimize(self.fun, self.x0, plot=False, disable_pbar=True)
        return sbo.vals

    def scipy_minimize(self, method):
        '''minimize target function with a given scipy method'''
        vals = [self.fun(self.x0)]
        scipy.optimize.minimize(self.fun, self.x0, method=method, options=self.scipy_params,
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
        plt.show()


if __name__ == '__main__':

    # configure benchmark parameters
    target = lambda x: np.sum(x**2)
    ##noise = lambda x: 0
    ##noise = lambda x: np.prod(1 + np.sin(x))
    noise = lambda x: np.sin(np.sum(10*np.abs(x)))
    ##noise = lambda x: np.random.randn()
    dim = 100
    num_tests = 10
    random_seed = 0

    # run benchmarking
    ba = BenchmarkAlgorithms(target, noise, dim)
    ba.test(num_tests, random_seed)
    ba.visualize()

