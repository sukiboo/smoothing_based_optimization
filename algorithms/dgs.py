"""
    Implementation of the Directional Gaussian Smoothing algorithm
    introduced in https://arxiv.org/abs/2002.03001
    The algorithm is implemented as a scipy.optimize method
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative


def DGS(fun, x0, args=(), learning_rate=.01, sigma=.1, quad_points=5,
        maxiter=1000, xtol=1e-6, ftol=1e-4, gtol=1e-4, callback=None, **options):
    """
    Minimize a scalar function using the DGS optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        The initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function. (Ignored in this code)
    learning_rate : float, optional
        The learning rate for the DGS optimizer. Default is 0.01.
    sigma : float, optional
        The smoothing parameter for the DGS optimizer. Default is 0.1.
    quad_points : int, optional
        The number of Gauss-Hermite quadrature points to use. Default is 7.
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
    # initialize DGS variables
    dim = len(x0)
    xk = x0.copy()
    fk = fun(xk)
    t = 0

    # establish search directions and quadrature points
    basis = np.eye(dim)
    gh_roots, gh_weights = np.polynomial.hermite.hermgauss(quad_points)

    def step(x):
        '''Perform a step of DGS optimizer'''
        nonlocal t
        t += 1
        # estimate smoothed directional derivative along each basis direction
        df_sigma_basis = np.zeros(dim)
        for d in range(dim):
            # estimate directional derivative via Gauss--Hermite quadrature
            f_d = lambda t: fun(x + t*basis[d])
            f_d_vals = np.array([f_d(sigma * p) for p in gh_roots])
            df_sigma_basis[d] = np.sum(gh_weights * gh_roots * f_d_vals)\
                                / (sigma * np.sqrt(np.pi)/2)
        # assemble smoothed gradient and update minimizer
        grad_sigma = np.matmul(basis, df_sigma_basis)
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

    return OptimizeResult(x=xk, fun=fk, jac=gfk, nit=t, nfev=quad_points*xk.size*t,
                          success=success, msg=msg)


if __name__ == '__main__':

    fun = lambda x: np.sum(x**2)
    x0 = np.random.randn(100)
    vals = [fun(x0)]
    res = DGS(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
