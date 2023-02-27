
import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dgs import DirectionalGaussianSmoothing

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class BenchmarkAlgorithms:

    def __init__(self, dim, num_tests, random_seed):
        self.dim = dim
        np.random.seed(random_seed)
        self.random_seed_list = np.random.randint(1e+9, size=num_tests)
        num_iters = 100
        self.algos = ['DGS', 'BFGS', 'CG', 'Powell', 'Nelder-Mead']
        self.sbo_params = {'sigma': .1, 'learning_rate': .01,
                           'quad_points': 7, 'num_iters': num_iters}
        self.scipy_params = {'maxiter': num_iters}
        self.test_algorithms()

    def test_algorithms(self):
        '''generate and solve an optimization problem'''
        self.vals = {alg: [] for alg in self.algos}
        for alg in self.algos:
            for t in tqdm.trange(num_tests, desc=f'Testing {alg}', ascii=True):
                np.random.seed(self.random_seed_list[t])
                self.generate_function()
                self.sample_initial_guess()
                self.run_minimization(alg)

    def generate_function(self):
        '''generate optimization target by perturbing sphere function'''
        dilation_sphere = 10 * np.random.rand(self.dim)
        dilation_noise = 10 * np.random.randn(self.dim)
        target = lambda x: np.sum(dilation_sphere * x**2)
        noise = lambda x: np.sin(np.sum(np.abs(dilation_noise * x)))
        ##noise = lambda x: np.random.randn()
        self.fun = lambda x: target(x) + noise(x)

    def sample_initial_guess(self):
        '''randomly sample an initial guess from the domain'''
        domain = np.array([[-1,1]] * self.dim)
        self.x0 = np.random.rand(self.dim) * (domain[:,1] - domain[:,0]) + domain[:,0]

    def run_minimization(self, alg):
        '''perform minimization with a given algorithm'''
        if alg in ['DGS']:
            vals = self.sbo_minimize(alg)
        else:
            vals = self.scipy_minimize(alg)
        self.vals[alg].append(vals)

    def sbo_minimize(self, method):
        '''minimize target function with smoothing-based optimization'''
        if method == 'DGS':
            sbo = DirectionalGaussianSmoothing(self.sbo_params)
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

    # run benchmarking
    ba = BenchmarkAlgorithms(dim, num_tests, random_seed)
    ba.visualize()

