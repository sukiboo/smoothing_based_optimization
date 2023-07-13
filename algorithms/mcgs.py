"""
    Implementation of the Monte Carlo Gaussian Smoothing algorithm
    introduced in ??
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def MCGS(fun, x0, args=(), learning_rate=.1, sigma=.1, num_points=1000,
         maxiter=1000, xtol=1e-6, ftol=1e-4, gtol=1e-4, callback=None, **options):
    """
    Minimize a scalar function using the MCGS optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the MCGS optimizer. Default is 0.01.
    sigma : float, optional
        The smoothing parameter for the MCGS optimizer. Default is 0.1.
    num_points : int, optional
        The number of Monte Carlo sample points to use. Default is 1000.
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
    # initialize MCGS variables
    dim = len(x0)
    xk = x0.copy()
    fk = fun(xk)
    t = 0

    def step(x):
        '''Perform a step of MCGS optimizer'''
        nonlocal t
        t += 1

        # estimate smoothed gradient via Monte Carlo
        grad_sigma = np.zeros(dim)
        for i in range(num_points):
            eps = np.random.randn(dim)
            grad_sigma += (fun(x + sigma*eps) - fun(x - sigma*eps)) * eps / (2*sigma*num_points)

        # update minimizer
        x -= learning_rate * grad_sigma
        return x, grad_sigma

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

    return OptimizeResult(x=xk, fun=fk, jac=gfk, nit=t, nfev=(num_points+1)*t,
                          success=success, msg=msg)


if __name__ == '__main__':

    fun = lambda x: np.sum(x**2)
    x0 = np.random.randn(100)
    vals = [fun(x0)]
    res = MCGS(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
