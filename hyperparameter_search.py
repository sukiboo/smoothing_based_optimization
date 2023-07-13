'''
This is temporary code file to search for the best hyperparameters for each algorithm/function pair.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

from benchmark_algorithms import BenchmarkAlgorithms

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=.8)


def visualize(ba, percentile=(.25,.5,.75), show=True):
    '''visualize optimization results'''
    function_name = ba.target_function['function_name']
    function_dim = ba.target_function['dim']
    fig, ax = plt.subplots(figsize=(10,6))
    for alg in ba.algorithms:
        alg_vals = pd.DataFrame(ba.vals[alg])
        alg_min, alg_avg, alg_max = alg_vals.quantile(percentile, axis=0).values
        plt.plot(alg_avg, linewidth=3, label=f'{alg} ({alg_avg[-1]:.2e})')
        plt.fill_between(range(alg_min.size), alg_min, alg_max, alpha=.25)
    ax.set_title(f'{function_name} {function_dim}d')

    # make sensible plots by adjusting y-scale
    m, M = float('inf'), float('-inf')
    for alg in ba.algorithms:
        alg_vals = pd.DataFrame(ba.vals[alg])
        alg_min, alg_max = alg_vals.quantile((.25, .75), axis=0).values
        m = min(m, alg_min[-1])
        M = max(M, alg_max[0])
    ax.set_ylim(m-.05*(M-m), M+.05*(M-m))
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.5))
    plt.tight_layout()

    # save results
    dir_name = f'./images/search/{config_file.split(".")[0]}'
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(f'./{dir_name}/{function_name}_{function_dim}.png', dpi=300, format='png')
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    config_file = 'adam.yml'
    num_tests = None

    # read configs
    configs = yaml.safe_load(open(f'./hyperparameters/search/{config_file}'))
    random_seed = configs['exp_params']['random_seed']
    dim = configs['exp_params']['dim']
    num_tests = configs['exp_params']['num_tests'] if num_tests is None else num_tests
    global_params = configs['global_params']

    # benchmark algorithms
    for function, algorithms in configs['functions'].items():
        function_params = {'function_name': function, 'dim': dim}
        # update each algorithm with global parameters
        for algortihm_params in algorithms.values():
            algortihm_params.update(global_params)
        ba = BenchmarkAlgorithms(algorithms, function_params, num_tests, random_seed)
        visualize(ba, show=False)

