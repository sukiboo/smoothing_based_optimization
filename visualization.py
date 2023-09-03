import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=1.2)


style = {
         'ackley': {'scale': 'linear', 'ylim': (-1, 23)},
           'levy': {'scale': 'cbrt', 'ylim': (50, 1500)},
    'michalewicz': {'scale': 'linear', 'ylim': (-60, -5)},
      'rastrigin': {'scale': 'linear', 'ylim': (-50, 2000)},
     'rosenbrock': {'scale': 'symlog', 'ylim': (-1, 1e+8)},
       ##'schwefel': {'scale': 'cbrt', 'ylim': (13000, 44000)},
       'schwefel': {'scale': 'cbrt', 'ylim': (0, 50000)},
    }


def visualize(data, function_name, dim, noise_level, percentile=(.25,.5,.75), show=True):
    '''Visualize optimization results.'''
    fig, ax = plt.subplots(figsize=(8,5))

    # plot median and confidence interval for each algorithm
    for alg in list(data.keys()):
        alg_vals = pd.DataFrame(data[alg]).fillna(method='ffill', axis=1)
        alg_min, alg_avg, alg_max = alg_vals.quantile(percentile, axis=0).values
        plt.plot(alg_avg, linewidth=4, alpha=.9, label=alg)
        plt.fill_between(range(alg_min.size), alg_min, alg_max, alpha=.25)

    # configure scale and axis depending on the function
    ##ax.set_title(f'{function_name} {dim}d')
    ax.set_ylim(style[function_name]['ylim'])
    if style[function_name]['scale'] == 'sqrt':
        ax.set_yscale('function', functions=(np.sqrt, lambda x: x**2))
    elif style[function_name]['scale'] == 'cbrt':
        ax.set_yscale('function', functions=(np.cbrt, lambda x: x**3))
    else:
        ax.set_yscale(style[function_name]['scale'])
    plt.legend()
    plt.tight_layout()

    # save the results
    os.makedirs('./images', exist_ok=True)
    plt.savefig(f'./images/{function_name}_{dim}_{noise_level}.png', dpi=300, format='png')
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    # load logs
    logs_dir = '100d_0e+00'
    for function_log in sorted(os.listdir(f'./logs/{logs_dir}')):
        with open(f'./logs/{logs_dir}/{function_log}', 'rb') as logfile:
            data = pickle.load(logfile)

        # parse parameters
        function_name = function_log.split('.')[0]
        dim = logs_dir.split('d')[0]
        noise_level = logs_dir.split('_')[-1]
        visualize(data, function_name, dim, noise_level, show=True)

