import numpy as np
import scipy
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import os

from algorithms.mcgs import MCGS
from algorithms.dgs import DGS
from algorithms.lsgd import LSGD
from algorithms.slgh import SLGH
from algorithms.nag import NAG
from algorithms.adam import Adam
from algorithms.rmsprop import RMSProp
from target_functions import setup_optimization

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=.8)


class BenchmarkAlgorithms:

    def __init__(self, algorithms, target_function, num_tests, random_seed, noise=0):
        self.algorithms = algorithms
        self.target_function = target_function
        self.num_tests = num_tests
        np.random.seed(random_seed)
        self.random_seed_list = np.random.randint(1e+9, size=num_tests)
        self.test_algorithms(noise)

    def test_algorithms(self, noise):
        '''generate and solve an optimization problem'''
        self.vals = {alg: [] for alg in self.algorithms}
        for alg, params in self.algorithms.items():
            for t in tqdm.trange(self.num_tests, desc=f'Testing {alg:>7s}', ascii=True):
                np.random.seed(self.random_seed_list[t])
                self.fun, self.x0 = setup_optimization(self.target_function,
                                                       self.random_seed_list[t],
                                                       noise=noise)
                self.run_minimization(alg, params)

    def run_minimization(self, alg, params):
        '''perform minimization with a given algorithm'''
        if alg.startswith('MCGS'):
            method = MCGS
        elif alg.startswith('DGS'):
            method = DGS
        elif alg.startswith('LSGD'):
            method = LSGD
        elif alg.startswith('SLGH'):
            method = SLGH
        elif alg.startswith('NAG'):
            method = NAG
        elif alg.startswith('Adam'):
            method = Adam
        elif alg.startswith('RMSProp'):
            method = RMSProp
        else:
            method = alg
        vals = [self.fun(self.x0)]
        scipy.optimize.minimize(self.fun, self.x0, method=method, options=params,
                                callback=lambda x: vals.append(self.fun(x)))
        self.vals[alg].append(vals)


if __name__ == '__main__':

    config_file = '100d'
    num_tests = None
    noise = 1e-4#0e1-0#1e-4#1e-8

    # read configs
    configs = yaml.safe_load(open(f'./hyperparameters/{config_file}.yml'))
    random_seed = configs['exp_params']['random_seed']
    dim = configs['exp_params']['dim']
    num_tests = configs['exp_params']['num_tests'] if num_tests is None else num_tests
    global_params = configs['global_params']

    # benchmark algorithms
    os.makedirs(f'./logs/{config_file}_{noise:.0e}/', exist_ok=True)
    for function, algorithms in configs['functions'].items():
        function_params = {'function_name': function, 'dim': dim}
        print(f'\nOptimizing {function} function...')

        # update each algorithm with global parameters
        for algortihm_params in algorithms.values():
            algortihm_params.update(global_params)
        ba = BenchmarkAlgorithms(algorithms, function_params, num_tests, random_seed, noise=noise)

        # save logs
        with open(f'./logs/{config_file}_{noise:.0e}/{function}.pkl', 'wb') as logfile:
            pickle.dump(ba.vals, logfile)

