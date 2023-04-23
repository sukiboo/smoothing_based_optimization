"""
    Implementation of Laplacian Smooth Gradient Descent algorithm
    introduced in https://arxiv.org/abs/1806.06317
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def LSGD(fun, x0, args=(), learning_rate=.01, sigma=.1,
         maxiter=1000, xtol=1e-6, ftol=1e-4, gtol=1e-4, callback=None, **options):
    """
    Minimize a scalar function using the LSGD optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the LSGD optimizer. Default is 0.01.
    sigma : float, optional
        The smoothing parameter for the LSGD optimizer. Default is 0.1.
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
    # initialize LSGD variables
    xk = x0.copy()
    fk = fun(xk)
    v = np.zeros_like(xk, dtype=np.complex)
    v[[0,1,-1]] = (-2,1,1)
    coef = 1 / (np.ones_like(v) - sigma * np.fft.fft(v))
    t = 0

    def step(x):
        '''Perform a step of LSGD optimizer'''
        nonlocal t, v, coef
        t += 1
        # use 2-point finite differences to estimate the gradient
        # this is the default gradient estimation for scipy.minimize methods, see
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py#L362
        grad = approx_derivative(fun, x, method='2-point')
        grad_sigma = np.fft.ifft(coef * np.fft.fft(grad)).real
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

    return OptimizeResult(x=xk, fun=fk, jac=gfk, nit=t, nfev=3*t,
                          success=success, msg=msg)


if __name__ == '__main__':

    fun = lambda x: np.sum(x**2)
    x0 = np.random.randn(100)
    vals = [fun(x0)]
    res = LSGD(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
