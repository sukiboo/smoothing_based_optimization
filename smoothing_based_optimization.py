
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class SmoothingBasedOptimization:

    def __init__(self, params):
        self.__dict__.update(params)

    def minimize(self, fun, x0, plot=False, disable_pbar=False):
        '''minimize the given function fun with the initial guess x0'''
        self.fun = fun
        self.x = x0.copy()
        self.dim = x0.size

        # establish search directions and quadrature points
        basis = np.eye(self.dim)
        gh_roots, gh_weights = np.polynomial.hermite.hermgauss(self.quad_points)

        # iteratively minimize the function
        self.vals = [self.fun(self.x)]
        pbar = tqdm.trange(self.num_iters, ascii=True,
                           postfix=f'value={self.vals[-1]: .2e}', disable=disable_pbar)
        for _ in pbar:

            # estimate smoothed directional derivative along each basis direction
            df_sigma_basis = np.zeros(self.dim)
            for d in range(self.dim):

                # estimate directional derivative via Gauss--Hermite quadrature
                f_d = lambda t: self.fun(self.x + t*basis[d])
                f_d_vals = np.array([f_d(self.sigma * p) for p in gh_roots])
                df_sigma_basis[d] = np.sum(gh_weights * gh_roots * f_d_vals)\
                                    / (self.sigma * np.sqrt(np.pi)/2)

            # assemble smoothed gradient and update minimizer
            df_sigma = np.matmul(basis, df_sigma_basis)
            self.x -= self.learning_rate * df_sigma

            # report progress
            self.vals.append(self.fun(self.x))
            pbar.set_postfix(value=f'{self.vals[-1]: .2e}')

        if plot:
            self.plot_values()

    def plot_values(self):
        '''plot function values throughout the optimization'''
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(self.vals, linewidth=3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.show()

