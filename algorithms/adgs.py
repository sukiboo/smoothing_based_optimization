"""
    Implementation of the Directional Gaussian Smoothing algorithm
    introduced in https://arxiv.org/abs/2002.03001
    The algorithm is implemented as a scipy.optimize method

    This particular algorithm will be modified to change sigma
"""

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative
from collections import deque


def ADGS(fun, x0, args=(), learning_rate=.01, sigma=10., gamma=.995, quad_points=7,
        maxiter=1000, xtol=1e-6, ftol=1e-4, gtol=1e-4, callback=None, **options):
    """
    Minimize a scalar function using the ADGS optimizer.
    It is DGS but with exponential decay on sigma
    """
    # initialize DGS variables
    dim = len(x0)
    xk = x0.copy()
    fk = fun(xk)
    t = 0
    ##fun_vals = deque([0]*10, maxlen=10)

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

        # check if sigma should be reduced
        sigma *= gamma
        ##fun_vals.append(fk)
        ##if all(abs(fk - fun_val) < 1e-3 for fun_val in fun_vals):
            ##print(f'iteration {t}, sigma is reduced: {sigma:.4f} --> {sigma/2:.4f}')
            ##print(fk, fun_vals, '\n')
            ##sigma = max(sigma/2, 1e-6)
            ##fun_vals = deque([0]*10, maxlen=10)

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
    res = ADGS(fun, x0, callback=lambda x: vals.append(fun(x)))
    print(res)
