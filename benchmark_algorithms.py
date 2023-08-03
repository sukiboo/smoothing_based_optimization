import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import os

from algorithms.dgs import DGS
from algorithms.mcgs import MCGS
from algorithms.lsgd import LSGD
from algorithms.slgh import SLGH
from algorithms.adgs import ADGS
from algorithms.adam import Adam
from algorithms.rmsprop import RMSProp
from algorithms.nag import NAG
from target_functions import setup_optimization

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=.8)


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

    def run_minimization(self, alg, params):
        '''perform minimization with a given algorithm'''
        # TODO: fix this
        if alg.startswith('DGS'):
            method = DGS
        elif alg.startswith('ADGS'):
            method = ADGS
        elif alg.startswith('MCGS'):
            method = MCGS
        elif alg.startswith('Adam'):
            method = Adam
        elif alg.startswith('RMSProp'):
            method = RMSProp
        elif alg.startswith('NAG'):
            method = NAG
        elif alg.startswith('LSGD'):
            method = LSGD
        elif alg.startswith('SLGH'):
            method = SLGH
        else:
            method = alg
        vals = [self.fun(self.x0)]
        scipy.optimize.minimize(self.fun, self.x0, method=method, options=params,
                                callback=lambda x: vals.append(self.fun(x)))
        self.vals[alg].append(vals)

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

        # make sensible plots by adjusting y-scale
        m, M = float('inf'), float('-inf')
        for alg in self.algorithms:
            alg_vals = pd.DataFrame(self.vals[alg])
            alg_min, alg_max = alg_vals.quantile((.25, .75), axis=0).values
            m = min(m, alg_min[-1])
            M = max(M, alg_max[0])
        ax.set_ylim(m-.05*(M-m), M+.05*(M-m))
        plt.legend()
        plt.tight_layout()

        # save results
        os.makedirs('./images', exist_ok=True)
        plt.savefig(f'./images/{function_name}_{function_dim}.png', dpi=300, format='png')
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':

    config_file = '100d'
    num_tests = None

    # read configs
    configs = yaml.safe_load(open(f'./hyperparameters/{config_file}.yml'))
    random_seed = configs['exp_params']['random_seed']
    dim = configs['exp_params']['dim']
    num_tests = configs['exp_params']['num_tests'] if num_tests is None else num_tests
    global_params = configs['global_params']

    # benchmark algorithms
    os.makedirs(f'./logs/{config_file}/', exist_ok=True)
    for function, algorithms in configs['functions'].items():
        function_params = {'function_name': function, 'dim': dim}
        # update each algorithm with global parameters
        for algortihm_params in algorithms.values():
            algortihm_params.update(global_params)
        ba = BenchmarkAlgorithms(algorithms, function_params, num_tests, random_seed)
        ba.visualize(show=False)
        # save logs
        with open(f'./logs/{config_file}/{function}.pkl', 'wb') as logfile:
            pickle.dump(ba.vals, logfile)

