"""
    Implementation of RMSProp optimization algorithm introduced
    in http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def RMSProp(fun, x0, args=(), learning_rate=.01, beta=.9, epsilon=1e-7,
            maxiter=1000, xtol=1e-6, ftol=1e-4, gtol=1e-4, callback=None, **options):
    """
    Minimize a scalar function using the RMSProp optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the RMSProp optimizer. Default is 0.01.
    beta : float, optional
        The beta parameter for the RMSProp optimizer. Default is 0.9.
    epsilon : float, optional
        The epsilon parameter for the RMSProp optimizer. Default is 1e-7.
    maxiter : int, optional
        The maximum number of iterations for the optimizer. Default is 1000.
    xtol : float
        Absolute error in solution xk acceptable for convergence.
    ftol : float
        Relative error in fun(xk) acceptable for convergence.
    gtol : float, optional
        Terminate successfully if gradient inf-norm is less than `gtol`.
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
    # initialize RMSProp variables
    xk = x0.copy()
    fk = fun(xk)
    s = np.zeros_like(xk)
    t = 0

    def step(x):
        '''Perform a step of RMSProp optimizer'''
        nonlocal t, s
        t += 1
        # use 2-point finite differences to estimate the gradient
        # this is the default gradient estimation for scipy.minimize methods, see
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py#L362
        grad = approx_derivative(fun, x, method='2-point')
        s = beta * s + (1 - beta) * np.square(grad)
        x -= learning_rate * grad / (np.sqrt(s) + epsilon)
        return x, grad

    # iteratively optimize target function
    success = False
    for _ in range(maxiter):
        x = xk.copy()
        fval = fk
        xk, gfk = step(x.copy())
        fk = fun(xk)
        if callback is not None:
            callback(xk)

        # check termination conditions
        if np.linalg.norm(gfk, np.inf) < gtol:
            msg = 'Optimization terminated succesfully.'
            success = True
            break
        if np.linalg.norm(x - xk, np.inf) < xtol:
            msg = 'Optimization terminated due to x-tolerance.'
            break
        if np.abs((fval - fk) / (fval + 1e-8)) < ftol:
            msg = 'Optimization terminated due to f-tolerance.'
            break
        if t >= maxiter:
            msg = 'The maximum number of iterations is reached.'
            break

    return OptimizeResult(x=xk, fun=fk, jac=gfk, nit=t, nfev=3*t,
                          success=success, msg=msg)


if __name__ == '__main__':

    fun = lambda x: np.sum(x**2)
    x0 = np.random.randn(100)
    vals = [fun(x0)]
    res = RMSProp(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
