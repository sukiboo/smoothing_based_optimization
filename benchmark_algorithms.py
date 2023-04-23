
import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from algorithms.dgs import DGS
from algorithms.adam import Adam
from algorithms.rmsprop import RMSProp
from algorithms.nag import NAG
from algorithms.lsgd import LSGD
from target_functions import setup_optimization

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class BenchmarkAlgorithms:

    def __init__(self, algorithms, target_function, num_tests, random_seed):
        self.algorithms = algorithms
        self.target_function = target_function
        self.num_tests = num_tests
        np.random.seed(random_seed)
        self.random_seed_list = np.random.randint(1e+9, size=num_tests)
        self.test_algorithms()

    def test_algorithms(self):
        '''generate and solve an optimization problem'''
        self.vals = {alg: [] for alg in self.algorithms}
        for alg, params in self.algorithms.items():
            for t in tqdm.trange(self.num_tests, desc=f'Testing {alg:>7s}', ascii=True):
                np.random.seed(self.random_seed_list[t])
                self.fun, self.x0 = setup_optimization(self.target_function,
                                                       self.random_seed_list[t])
                self.run_minimization(alg, params)

    # TODO: combine this function with the next
    def run_minimization(self, alg, params):
        '''perform minimization with a given algorithm'''
        vals = self.scipy_minimize(alg, params)
        self.vals[alg].append(vals)

    def scipy_minimize(self, method, params):
        '''minimize target function with a given scipy method'''
        vals = [self.fun(self.x0)]
        # TODO: fix this
        if method == 'DGS':
            method = DGS
        if method == 'Adam':
            method = Adam
        elif method == 'RMSProp':
            method = RMSProp
        elif method == 'NAG':
            method = NAG
        elif method == 'LSGD':
            method = LSGD
        scipy.optimize.minimize(self.fun, self.x0, method=method, options=params,
                                callback=lambda x: vals.append(self.fun(x)))
        return vals

    def visualize(self, percentile=(.25,.5,.75), show=True):
        '''visualize optimization results'''
        function_name = self.target_function['function_name']
        function_dim = self.target_function['dim']
        fig, ax = plt.subplots(figsize=(8,5))
        for alg in self.algorithms:
            alg_vals = pd.DataFrame(self.vals[alg])
            alg_min, alg_avg, alg_max = alg_vals.quantile(percentile, axis=0).values
            plt.plot(alg_avg, linewidth=3, label=alg)
            plt.fill_between(range(alg_min.size), alg_min, alg_max, alpha=.25)
        ax.set_title(f'{function_name} {function_dim}d')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        os.makedirs('./images', exist_ok=True)
        ##plt.savefig(f'./images/{function_name}_{function_dim}.png', dpi=300, format='png')
        # very ugly workaround for the time being
        save_name = f'{function_name}_{function_dim}'
        if 'noise_magnitude' in self.target_function:
            save_name += f'_{self.target_function["noise_magnitude"]}'
        if 'noise_freq' in self.target_function:
            save_name += f'_{self.target_function["noise_freq"]}'
        plt.savefig(f'./images/{save_name}.png', dpi=300, format='png')
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':

    num_tests = 10
    random_seed = 0
    maxiter = 100

    algorithms = {
        'DGS': {'learning_rate': .001, 'sigma': .1, 'quad_points': 7, 'maxiter': maxiter},
       'LSGD': {'learning_rate': .001, 'sigma': .1, 'maxiter': maxiter},
        'NAG': {'learning_rate': .001, 'beta': .5, 'maxiter': maxiter},
       'Adam': {'learning_rate': .1, 'beta1': .9, 'beta2': .999, 'maxiter': maxiter},
    'RMSProp': {'learning_rate': .1, 'beta': .9, 'maxiter': maxiter},
       'BFGS': {'maxiter': maxiter},
         'CG': {'maxiter': maxiter},
     ##'Powell': {'maxiter': maxiter},
            }

    '''dilated sphere with noise'''
    dim = 100
    target_functions = [
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 1., 'noise_magnitude': 1.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 10., 'noise_magnitude': 1.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 100., 'noise_magnitude': 1.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 1., 'noise_magnitude': 10.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 10., 'noise_magnitude': 10.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 100., 'noise_magnitude': 10.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 1., 'noise_magnitude': 100.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 10., 'noise_magnitude': 100.},
        {'function_name': 'sphere_noisy', 'dim': dim,
            'noise_freq': 100., 'noise_magnitude': 100.},
        {'function_name': 'sphere_random', 'dim': dim, 'noise_magnitude': 1e-6},
        {'function_name': 'sphere_random', 'dim': dim, 'noise_magnitude': 1e-5},
        {'function_name': 'sphere_random', 'dim': dim, 'noise_magnitude': 1e-4},
        ]

    '''any dimensionality functions from https://www.sfu.ca/~ssurjano/optimization.html'''
    dim = 100
    target_functions_ssa = [
        {'function_name': 'ackley', 'dim': dim},
        {'function_name': 'griewank', 'dim': dim},
        {'function_name': 'levy', 'dim': dim},
        {'function_name': 'michalewicz', 'dim': dim},
        {'function_name': 'rastrigin', 'dim': dim},
        {'function_name': 'rosenbrock', 'dim': dim},
        {'function_name': 'schwefel', 'dim': dim},
        {'function_name': 'sphere', 'dim': dim},
        ]

    '''some 2-dimensional functions from https://www.sfu.ca/~ssurjano/optimization.html'''
    target_functions_ssa_2 = [
        {'function_name': 'branin', 'dim': 2},
        {'function_name': 'cross-in-tray', 'dim': 2},
        {'function_name': 'dropwave', 'dim': 2},
        {'function_name': 'eggholder', 'dim': 2},
        {'function_name': 'holder', 'dim': 2},
        {'function_name': 'sphere', 'dim': 2},
        ]


    # run benchmarking
    for target_function in target_functions:
        ba = BenchmarkAlgorithms(algorithms, target_function, num_tests, random_seed)
        ba.visualize(show=False)
