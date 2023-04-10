"""
    Implementation of Adam optimization algorithm
    introduced in https://arxiv.org/abs/1412.6980
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def Adam(fun, x0, args=(), learning_rate=.01, beta1=.9, beta2=.999,
         epsilon=1e-7, maxiter=1000, callback=None, **options):
    """
    Minimize a scalar function using the Adam optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the Adam optimizer. Default is 0.01.
    beta1 : float, optional
        The beta1 parameter for the Adam optimizer. Default is 0.9.
    beta2 : float, optional
        The beta2 parameter for the Adam optimizer. Default is 0.999.
    epsilon : float, optional
        The epsilon parameter for the Adam optimizer. Default is 1e-7.
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
    # initialize Adam variables
    xk = x0.copy()
    m = np.zeros_like(xk)
    v = np.zeros_like(xk)
    t = 0

    def step(x):
        '''Perform a step of Adam optimizer'''
        nonlocal t, m, v
        t += 1
        # use 2-point finite differences to estimate the gradient
        # this is the default gradient estimation for scipy.minimize methods, see
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py#L362
        grad = approx_derivative(fun, x, method='2-point')
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        m_hat = m / (1 - np.power(beta1,t))
        v_hat = v / (1 - np.power(beta2,t))
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
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
    res = Adam(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
