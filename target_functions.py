"""
    A list of optimization benchmark functions from
    https://www.sfu.ca/~ssurjano/optimization.html
"""

import numpy as np


def target_function(function_params, random_seed=None):
    """Setup benchmark function"""
    function_name = function_params['function_name']
    dim = function_params['dim']
    if random_seed:
        np.random.seed(random_seed)

    # noisy sphere -- change this stupid name later
    if function_name == 'sphere_noisy':
        dilation_sphere = 100 * np.random.rand(dim)
        dilation_noise = function_params['noise_freq'] * np.random.rand(dim)
        target = lambda x: np.sum(dilation_sphere * x**2)
        noise = lambda x: function_params['noise_magnitude']\
            * (1 + np.sin(np.sum(np.abs(dilation_noise * x))))
        fun = lambda x: target(x) + noise(x)
        x_dom = [[-1, 1]] * dim
        x_min = [0] * dim

    # random sphere -- see above regarding the name
    elif function_name == 'sphere_random':
        dilation_sphere = 100 * np.random.rand(dim)
        target = lambda x: np.sum(dilation_sphere * x**2)
        noise = lambda x: function_params['noise_magnitude'] * np.random.rand()
        fun = lambda x: target(x) + noise(x)
        x_dom = [[-1, 1]] * dim
        x_min = [0] * dim


    # Ackley function
    elif function_name == 'ackley':
        fun = lambda x: -20 * np.exp(-.2 * np.sqrt(np.sum(x**2) / dim))\
            - np.exp(np.sum(np.cos(2*np.pi*x)) / dim) + 20 + np.exp(1)
        x_dom = [[-32.768, 32.768]] * dim
        x_min = [0] * dim

    # Griewank function
    elif function_name == 'griewank':
        fun = lambda x: np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(dim)+1))) + 1
        x_dom = [[-600, 600]] * dim
        x_min = [0] * dim

    # Levy function
    elif function_name == 'levy':
        w = lambda x: (x + 3) / 4
        fun = lambda x: np.sin(np.pi*w(x)[0])**2 \
            + np.sum((w(x)[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w(x)[:-1] + 1)**2)) \
            + (w(x)[-1] - 1)**2 * (1 + np.sin(2*np.pi*w(x)[-1])**2)
        x_dom = [[-10, 10]] * dim
        x_min = [1] * dim

    # Michalewicz function
    elif function_name == 'michalewicz':
        m = 10 # default parameter
        ##fun = lambda x: np.sum(-np.sin(x[d]) * np.sin(d * x[d]**2 / np.pi)**m\
                               ##for d in range(dim))
        fun = lambda x: -np.sum(np.sin(x) * np.sin(np.arange(1,dim+1) * x**2 / np.pi)**(2*m))
        x_dom = [[0, np.pi]] * dim
        x_min = [None] * dim

    # Rastrigin function
    elif function_name == 'rastrigin':
        fun = lambda x: 10*dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim

    # Rosenbrock function
    elif function_name == 'rosenbrock':
        ##fun = lambda x: np.sum(100 * (x[d+1] - x[d]**2)**2 + (x[d] - 1)**2 for d in range(dim-1))
        fun = lambda x: np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
        x_dom = [[-5, 10]] * dim
        x_min = [1] * dim

    # Schwefel function
    elif function_name == 'schwefel':
        ##fun = lambda x: 418.9829*dim\
            ##- np.sum(x[d] * np.sin(np.sqrt(np.abs(x[d]))) for d in range(dim))
        fun = lambda x: 418.9829*dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        x_dom = [[-500, 500]] * dim
        x_min = [420.9687] * dim

    # sphere function
    elif function_name == 'sphere':
        fun = lambda x: np.sum(x**2)
        x_dom = [[-5.12, 5.12]] * dim
        x_min = [0] * dim


    # Branin function
    elif function_name == 'branin':
        fun = lambda x: 10 + 10*(1 - 1/(8*np.pi)) * np.cos(x[0]) \
            + (x[1] - 5.1/(4*np.pi**2) * x[0]**2 + 5/np.pi * x[0] - 6)**2
        x_dom = [[-5, 10], [0, 15]]
        x_min = [np.pi, 2.275]

    # cross-in-tray function
    elif function_name == 'cross-in-tray':
        fun = lambda x: -.0001*(np.abs(np.sin(x[0]) * np.sin(x[1]) \
            * np.exp(np.abs(100 - np.sqrt(np.sum(x**2))/np.pi))) + 1)**.1
        x_dom = [[-10, 10], [-10, 10]]
        x_min = [1.3491, 1.3491]

    # dropwave function
    elif function_name == 'dropwave':
        fun = lambda x: 1 - (1 + np.cos(12*np.sqrt(np.sum(x**2)))) / (.5*np.sum(x**2) + 2)
        x_dom = [[-5.12, 5.12], [-5.12, 5.12]]
        x_min = [0, 0]

    # eggholder function
    elif function_name == 'eggholder':
        fun = lambda x: -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0]/2 + 47)))\
            - x[0] * np.sin(np.sqrt(np.abs(x[0] - x[1] - 47)))
        x_dom = [[-512, 512], [-512, 512]]
        x_min = [512, 404.2319]

    # Holder table function
    elif function_name == 'holder':
        fun = lambda x: -np.abs(np.sin(x[0]) * np.cos(x[1])\
            * np.exp(np.abs(1 - np.sqrt(np.sum(x**2)) / np.pi)))
        x_dom = [[-10, 10], [-10, 10]]
        x_min = [8.05502, 9.66459]

    else:
        raise SystemExit('function {:s} is not defined...'.format(function_name))

    return fun, np.array(x_min), np.array(x_dom).T


def initial_guess(x_dom, random_seed):
    """Randomly sample initial guess"""
    np.random.seed(random_seed)
    dim = x_dom.shape[-1]
    x0 = (x_dom[1] - x_dom[0]) * np.random.rand(dim) + x_dom[0]
    return x0


def setup_optimization(function_params, random_seed, noise=False):
    """Return a target function and an initial guess"""
    fun, x_min, x_dom = target_function(function_params, random_seed)
    x0 = initial_guess(x_dom, random_seed)
    if not noise:
        return fun, x0
    else:
        # add random noise
        fun2 = lambda x: fun(x) * (.9999 + .0002*np.random.rand())
        return fun2, x0
