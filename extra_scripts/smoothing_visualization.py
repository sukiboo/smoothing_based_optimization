
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace')


class InteractiveSmoothing:

    def __init__(self, params):
        self.__dict__.update(params)
        self.fun = lambda x: self.target(x) + self.noise_magnitude * self.noise(x)

    def gaussian_smoothing(self):
        """Compute gaussian smoothing of the target function"""
        gh_roots, gh_weights = np.polynomial.hermite.hermgauss(self.num_quad)
        fun_vals = np.array([self.fun(self.x + p * self.sigma) for p in gh_roots])
        fun_sigma = np.matmul(gh_weights, fun_vals) / np.sqrt(np.pi)
        return fun_sigma

    def plot(self):
        """Create interactive plot"""
        self.fig, ax = plt.subplots(figsize=(8,6))
        plt.subplots_adjust(left=.1, right=.9, bottom=.24, top=.98)
        # initial plot
        self.fun_plot, = plt.plot(self.x, self.fun(self.x), linewidth=3, label='target function')
        self.fun_sigma_plot, = plt.plot(self.x, self.gaussian_smoothing(),
                                        linewidth=3, label=f'{self.sigma:.2f}-smoothing')
        # configure slider for num_quad
        quad_axes = plt.axes([.2, .125, .6, .04])
        self.quad_slider = Slider(quad_axes, 'num_quad', 1, 51, valinit=self.num_quad, valfmt='%d')
        self.quad_slider.on_changed(self.plot_update)
        # configure slider for sigma
        sigma_axes = plt.axes([.2, .075, .6, .04])
        self.sigma_slider = Slider(sigma_axes, 'sigma', 0., 1., valinit=self.sigma)
        self.sigma_slider.on_changed(self.plot_update)
        # configure slider for noise
        noise_axes = plt.axes([.2, .025, .6, .04])
        self.noise_slider = Slider(noise_axes, 'noise', .0, .1, valinit=self.noise_magnitude)
        self.noise_slider.on_changed(self.plot_update)
        plt.show()

    def plot_update(self, _):
        """Update the plot with the current parameter values"""
        self.sigma = self.sigma_slider.val
        self.num_quad = int(self.quad_slider.val)
        self.noise_magnitude = self.noise_slider.val
        self.fun_sigma_plot.set_ydata(self.gaussian_smoothing())
        self.fun_plot.set_ydata(self.fun(self.x))
        self.fig.canvas.draw_idle()


if __name__ == '__main__':

    params = {
        'target': lambda x: x**2,
        'x': np.linspace(-1, 1, 1000),

        # noise parameters
        'noise': lambda x: np.sin(50*np.pi*x),
        'noise_magnitude': .0,
        'random_seed': 0,

         # smoothing parameters
        'sigma': .1,
        'num_quad': 11,
        }

    IS = InteractiveSmoothing(params)
    IS.plot()

