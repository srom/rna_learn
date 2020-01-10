import argparse
import os
import logging
import string
import json
import time

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .training import train_conv1d_with_hyperparameters


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iterations', type=int)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    n_iterations = args.n_iterations
    verbose = args.verbose

    run_id = generate_random_run_id()

    output_dir = os.path.join(os.getcwd(), f'hyperparameters/{run_id}')
    path_model = os.path.join(output_dir, 'best_model.h5')
    path_output_best = os.path.join(output_dir, 'best_hyperparameters.json')
    path_trace = os.path.join(output_dir, 'trace.json')

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    logger.info(f'Run ID: {run_id}')

    gaussian_process_optimisation(n_iterations, path_model, path_output_best, path_trace, verbose)


def gaussian_process_optimisation(n_iter, model_path, path_output_best, path_trace, verbose):
    make_float = lambda x: float(x)
    round_to_int = lambda x: int(round(x))
    inverse_log = lambda x: 10 ** x

    optimization_rules = [
        ('n_epochs', (2, 20), round_to_int),
        #('batch_size', (16, 256), round_to_int),
        #('learning_rate', (-4.5, -3.5), inverse_log),
        #('adam_epsilon', (-8, 0), inverse_log),
        ('n_conv_1', (1, 5), round_to_int),
        ('n_filters_1', (1, 100), round_to_int), 
        ('kernel_size_1', (2, 100), round_to_int),
        #('l2_reg_1', (0., 0.1), make_float),
        ('n_conv_2', (1, 5), round_to_int),
        ('n_filters_2', (1, 100), round_to_int), 
        ('kernel_size_2', (2, 100), round_to_int),
        #('l2_reg_2', (0., 0.1), make_float),
        #('dropout', (0., 0.8), make_float),
    ]
    bounds = np.array([r[1] for r in optimization_rules], dtype='float32')
    transform_functions = [r[2] for r in optimization_rules]

    X_init_def = [
        dict(
            n_epochs=10,
            #batch_size=128,
            learning_rate=-3.6,
            #adam_epsilon=-7,
            n_conv_1=1,
            n_filters_1=20, 
            kernel_size_1=2,
            #l2_reg_1=0.1,
            n_conv_2=1,
            n_filters_2=10, 
            kernel_size_2=2,
            #l2_reg_2=0.01,
            #dropout=0.,
        ),
        dict(
            n_epochs=5,
            #batch_size=32,
            learning_rate=-4,
            #adam_epsilon=-7,
            n_conv_1=2,
            n_filters_1=100, 
            kernel_size_1=10,
            #l2_reg_1=0.,
            n_conv_2=2,
            n_filters_2=100, 
            kernel_size_2=10,
            #l2_reg_2=0.,
            #dropout=0.5,
        ),
    ]
    X_init = np.array([
        [x[r[0]] for r in optimization_rules]
        for x in X_init_def
    ], dtype='float32')

    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    m52_s = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

    gpr = GaussianProcessRegressor(kernel=m52)
    gpr_s = GaussianProcessRegressor(kernel=m52_s)

    def f(x):
        x_args = [transform_functions[i](v) for i, v in enumerate(x)]

        logger.info('Hyperparameters')
        for i, rule in enumerate(optimization_rules):
            param = rule[0]
            if isinstance(x_args[i], float):
                logger.info(f'{param}: {x_args[i]:.2e}')
            else:
                logger.info(f'{param}: {x_args[i]}')

        return train_conv1d_with_hyperparameters(*x_args, verbose=verbose)

    logger.info('Initializing with first two hyperparameters set')
    Y_init = []
    T_init = []
    best_loss = -np.inf
    for i, x in enumerate(X_init):
        logger.info(f'Initialization {i+1}')
        loss, elapsed, model = f(x)
        Y_init.append(loss)
        T_init.append(elapsed)
        if loss > best_loss:
            best_loss = loss
            store_best_params(
                0,
                x, 
                best_loss, 
                elapsed, 
                optimization_rules,
                path_output_best,
            )
            model.save(model_path)

    X_sample = X_init
    Y_sample = np.array(Y_init)[..., np.newaxis]
    T_sample = np.array(np.log(T_init))[..., np.newaxis]
    
    acq_fn = expected_improvement_per_second
    # acq_fn = expected_improvement

    logger.info('Starting hyperparameters optimisation')

    trace = []
    for i in range(n_iter):
        logger.info(f'Iteration {i+1}')

        gpr.fit(X_sample, Y_sample)
        gpr_s.fit(X_sample, T_sample)

        x_next, ei = propose_location(acq_fn, X_sample, Y_sample, gpr, gpr_s, bounds)

        loss, elapsed, model = f(x_next)

        trace.append({
            'x': x_next,
            'ei': ei,
            'loss': loss,
            'elapsed': elapsed,
            'timestamp': time.time(),
        })
        store_trace(trace, optimization_rules, path_trace)

        if loss > best_loss:
            best_loss = loss
            store_best_params(
                i + 1,
                x_next, 
                loss, 
                elapsed, 
                optimization_rules,
                path_output_best,
            )
            model.save(model_path)

        X_next = x_next[np.newaxis, ...]
        Y_next = np.array([loss])[..., np.newaxis]
        T_next = np.array([np.log(elapsed)])[..., np.newaxis]

        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
        T_sample = np.vstack((T_sample, Y_next))

    logger.info('DONE')


def expected_improvement_per_seconds(X, X_sample, Y_sample, gpr, gpr_s, xi=0.01):
    return expected_improvement(X, X_sample, Y_sample, gpr, gpr_s, xi, per_second=True)


def expected_improvement(X, X_sample, Y_sample, gpr, gpr_s, xi=0.01, per_second=False):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model, then divides 
    by the expected duration in seconds from a second Gaussian process 
    surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor of f fitted to samples.
        gpr_s: A GaussianProcessRegressor of the duration fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    expected_duration_ln = gpr_s.predict(X)
    expected_duration = np.exp(expected_duration_ln)

    sigma = sigma[..., np.newaxis]

    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='ignore'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    if per_second:
        return ei / expected_duration
    else:
        return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, gpr_s, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor of f fitted to samples.
        gpr_s: A GaussianProcessRegressor of the duration fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    ei_min_neg = np.inf
    x_min = None
    
    def min_obj(X):
        return -acquisition(X[np.newaxis, ...], X_sample, Y_sample, gpr, gpr_s)[0, 0]
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.success and res.fun < ei_min_neg:
            ei_min_neg = res.fun
            x_min = res.x

    if x_min is None:
        raise ValueError('Optimisation of acquisition function failed')

    return x_min, -ei_min_neg


def store_trace(trace, optimization_rules, output_path):
    params = [r[0] for r in optimization_rules]
    transform_functions = [r[2] for r in optimization_rules]

    x_args = [
        [transform_functions[i](v) for i, v in enumerate(t['x'])]
        for t in trace
    ]
    
    trace_output = [
        {
            'x': {
                param: x_args[i][j]
                for j, param in enumerate(params)
            },
            'expected_improvement': t['ei'],
            'loss': t['loss'],
            'elapsed': t['elapsed'],
            'timestamp': t['timestamp'],
        }
        for i, t in enumerate(trace)
    ]

    with open(output_path, 'w') as fd:
        json.dump(
            {'trace': trace_output, 'n_iterations': len(trace)}, 
            fd,
        )


def store_best_params(iteration, x, loss, elapsed, optimization_rules, output_path):
    params = [r[0] for r in optimization_rules]
    transform_functions = [r[2] for r in optimization_rules]
    x_args = [transform_functions[i](v) for i, v in enumerate(x)]

    output = {
        'iteration': iteration,
        'x': {
            param: x_args[i]
            for i, param in enumerate(params)
        },
        'loss': loss,
        'elapsed_seconds': elapsed,
    }

    with open(output_path, 'w') as fd:
        json.dump(output, fd)


def generate_random_run_id():
    chars = [c for c in string.ascii_lowercase + string.digits]
    random_slug_chars = np.random.choice(chars, size=5, replace=True)
    random_slug = ''.join(random_slug_chars)
    return f'run_{random_slug}'


if __name__ == '__main__':
    main()
