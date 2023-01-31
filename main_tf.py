
import numpy as np
import tensorflow as tf

from dgs import DirectionalGaussianSmoothing


def tf_to_opt(model, data):
    '''reframe model training as an optimization problem'''
    # translate weights to array
    weights_to_array = lambda weights:\
        np.concatenate([weight.flatten() for weight in weights])

    # define initial guess
    weights = model.get_weights()
    x0 = weights_to_array(weights)

    # translate array to weights
    shapes = [weight.shape for weight in weights]
    sizes = [weight.size for weight in weights]
    array_to_weights = lambda array:\
        [a.reshape(shapes[i]) for i,a in enumerate(np.split(array, np.cumsum(sizes)[:-1]))]

    # define target function
    def fun(array):
        model.set_weights(array_to_weights(array))
        mse = model.evaluate(*data, verbose=0)
        return mse

    return fun, x0


if __name__ == '__main__':

    # sample training data
    x = np.linspace(-np.pi, np.pi, 1001)
    y = np.sin(x) * np.exp(x)

    # setup model
    model = tf.keras.Sequential([
        tf.keras.layers.Input((None,1)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)])
    model.compile(loss='mse')

    # get target function and initial guess
    fun, x0 = tf_to_opt(model, (x,y))

    # setup smoothing-based optimization
    params = {'sigma': .1, 'learning_rate': .01, 'quad_points': 5, 'num_iters': 100}
    dgs = DirectionalGaussianSmoothing(params)
    dgs.minimize(fun, x0, plot=True)

