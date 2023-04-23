"""
    the original BFGS implementation is stolen from https://github.com/trsav/bfgs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


def grad(f, x):
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d):
        x_for = np.copy(x)
        x_back = np.copy(x)
        x_for[i] += h
        x_back[i] -= h
        nabla[i] = (f(x_for) - f(x_back))/(2*h)
    return nabla

def line_search(f, x, p, nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    x_new = x + a * p
    nabla_new = grad(f,x_new)
    while (a > 1e-6) and (f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p):
        a *= 0.5
        x_new = x + a * p
        nabla_new = grad(f,x_new)
    return a

def BFGS(f, x0, max_it):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.
    INPUTS:
    f:      function to be optimised
    x0:     intial guess
    max_it: maximum iterations
    plot:   if the problem is 2 dimensional, returns
            a trajectory plot of the optimisation scheme.
    OUTPUTS:
    x:      the optimal solution of the function f
    '''
    vals = [f(x0)]
    d = len(x0) # dimension of problem
    nabla = grad(f,x0) # initial gradient
    H = np.eye(d) # initial hessian
    x = x0[:]
    it = 0
    print('\n BFGS optimization:')
    while (it < max_it) and (np.linalg.norm(nabla) > 1e-5):
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,x,p,nabla) # line search
        s = a * p
        x_new = x + a * p
        nabla_new = grad(f,x_new)
        y = nabla_new - nabla
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:]
        x = x_new[:]
        vals.append(f(x))
        print(f' BFGS iteration {it:2d}: function value = {vals[-1]: .2e}')
    return vals



def smooth_grad(fun, x, sigma=.1, quad_points=7):
    '''estimate smoothed gradient via Gauss-Hermite quadrature'''
    # establish search directions and quadrature points
    dim = len(x)
    basis = np.eye(dim)
    gh_roots, gh_weights = np.polynomial.hermite.hermgauss(quad_points)

    # estimate smoothed directional derivative along each basis direction
    df_sigma_basis = np.zeros(dim)
    for d in range(dim):

        # estimate directional derivative via Gauss--Hermite quadrature
        f_d = lambda t: fun(x + t*basis[d])
        f_d_vals = np.array([f_d(sigma * p) for p in gh_roots])
        df_sigma_basis[d] = np.sum(gh_weights * gh_roots * f_d_vals)\
                            / (sigma * np.sqrt(np.pi)/2)

    # assemble smoothed gradient and update minimizer
    df_sigma = np.matmul(basis, df_sigma_basis)

    return df_sigma

def smooth_line_search(f, x, p, nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    x_new = x + a * p
    nabla_new = smooth_grad(f,x_new)
    while (a > 1e-6) and (f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p):
        a *= 0.5
        x_new = x + a * p
        nabla_new = smooth_grad(f,x_new)
    return a

def SBFGS(f, x0, max_it):
    '''
    BFGS but with smoothed gradient instead
    '''
    vals = [f(x0)]
    d = len(x0) # dimension of problem
    nabla = smooth_grad(f,x0) # initial gradient
    H = np.eye(d) # initial hessian
    x = x0[:]
    it = 0
    print('\nSBFGS optimization:')
    while (it < max_it) and (np.linalg.norm(nabla) > 1e-5):
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = smooth_line_search(f,x,p,nabla) # line search
        s = a * p
        x_new = x + a * p
        nabla_new = smooth_grad(f,x_new)
        y = nabla_new - nabla
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:]
        x = x_new[:]
        vals.append(f(x))
        print(f'SBFGS iteration {it:2d}: function value = {vals[-1]: .2e}')
    return vals



def smooth_line_search_1d(fun, x, p, nabla):
    '''
    same as the usual line search but in the smoothed 1d slice of f
    '''
    # smoothing of 1d slice
    f_1d = lambda t: fun(x + t*p)
    gh_roots, gh_weights = np.polynomial.hermite.hermgauss(7)
    def f(t):
        f_vals = np.array([f_1d(t + .1*r) for r in gh_roots])
        f_sigma = np.matmul(gh_weights, f_vals) / np.sqrt(np.pi)
        return f_sigma
    # usual line search
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(0) # fun(x)
    x_new = x + a * p
    nabla_new = smooth_grad(fun, x_new)
    while (a > 1e-6) and (f(a) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p):
        a *= 0.5
        x_new = x + a * p
        nabla_new = smooth_grad(fun, x_new)
    return a

def SBFGS1d(f, x0, max_it):
    '''
    BFGS but with smoothed gradient and line search over 1d slice
    '''
    vals = [f(x0)]
    d = len(x0) # dimension of problem
    nabla = smooth_grad(f,x0) # initial gradient
    H = np.eye(d) # initial hessian
    x = x0[:]
    it = 0
    print('\nSBFGS1d optimization:')
    while (it < max_it) and (np.linalg.norm(nabla) > 1e-5):
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = smooth_line_search_1d(f,x,p,nabla) # line search
        s = a * p
        x_new = x + a * p
        nabla_new = smooth_grad(f,x_new)
        y = nabla_new - nabla
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:]
        x = x_new[:]
        vals.append(f(x))
        print(f'SBFGS1d iteration {it:2d}: function value = {vals[-1]: .2e}')
    return vals



if __name__ == '__main__':

    ##np.random.seed(3)

    # target function and noise
    target = lambda x: np.sum(x**2)
    noise = lambda x: np.sin(np.sum(10*np.abs(x)))
    dim = 100

    # optimization setting
    fun = lambda x: target(x) + noise(x)
    x0 = np.random.randn(dim)
    num_iters = 25

    # optimize
    vals_bfgs = BFGS(fun, x0, num_iters)
    vals_sbfgs = SBFGS(fun, x0, num_iters)
    vals_sbfgs1d = SBFGS1d(fun, x0, num_iters)

    # plot function values
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(vals_bfgs, linewidth=3, label='BFGS')
    ax.plot(vals_sbfgs, linewidth=3, label='SBFGS')
    ax.plot(vals_sbfgs1d, linewidth=3, label='SBFGS1d')
    plt.legend()
    plt.tight_layout()
    plt.show()

