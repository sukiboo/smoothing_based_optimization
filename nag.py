"""
    Implementation of Nesterov's Accelerated Gradient algorithm introduced
    in https://www.mathnet.ru/links/23f7189434c68ec3bee31436bee6cb69/dan46009.pdf
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def NAG(fun, x0, args=(), learning_rate=.01, beta=.9,
         epsilon=1e-7, maxiter=1000, callback=None, **options):
    """
    Minimize a scalar function using the NAG optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the NAG optimizer. Default is 0.01.
    beta : float, optional
        The momentum parameter for the NAG optimizer. Default is 0.9.
    epsilon : float, optional
        The epsilon parameter for the NAG optimizer. Default is 1e-7.
    maxiter : int, optional
        The maximum number of iterations for the optimizer. Default is 1000.
    callback : callable, optional
        A function to be called after each iteration of the optimizer.
        The function is called as callback(xk), where xk is the current parameter vector.
    options : dict, optional
        Additional options to pass to the optimizer.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a dictionary.
    """
    # initialize NAG variables
    xk = x0.copy()
    v = np.zeros_like(xk)
    t = 0

    def step(x):
        '''Perform a step of NAG optimizer'''
        nonlocal t, v
        t += 1
        # use 2-point finite differences to estimate the gradient
        # this is the default gradient estimation for scipy.minimize methods, see
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py#L362
        grad = approx_derivative(fun, x, method='2-point')
        v = beta * v - learning_rate * grad
        x += beta * v - learning_rate * grad
        return x, grad

    # initialize the optimization result to be returned
    result = OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False

    # iteratively optimize target function
    for _ in range(maxiter):
        xk, dfk = step(xk)
        # update optimization result
        result.x = xk
        result.fun = fun(xk)
        result.nit += 1
        result.nfev += 1
        if callback is not None:
            callback(xk)
        # terminate if optimization is successful
        if np.linalg.norm(dfk) < np.sqrt(epsilon):
            result.success = True
            break

    return result


if __name__ == '__main__':

    fun = lambda x: np.sum(x**2)
    x0 = np.random.randn(100)
    vals = [fun(x0)]
    res = NAG(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
