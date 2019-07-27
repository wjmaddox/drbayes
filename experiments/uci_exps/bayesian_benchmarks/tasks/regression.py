"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))

Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.

"""

import argparse
import numpy as np
import time
from scipy.stats import norm

from bayesian_benchmarks.data import get_regression_data
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.get_model import get_regression_model

def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='linear', nargs='?', type=str)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()

def run(ARGS, data=None, model=None, is_test=False):

    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)
    res = {}

    print('data standard deviation is: ', data.Y_std)
    start = time.time()
    model.fit(data.X_train, data.Y_train)
    fit_time = time.time() - start
    res['fit_time'] = fit_time

    start = time.time()
    m, v = model.predict(data.X_test)
    infer_time = time.time() - start
    res['infer_time'] = infer_time
    

    l = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
    res['test_loglik'] = np.average(l)

    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v**0.5) * data.Y_std)
    res['test_loglik_unnormalized'] = np.average(lu)

    d = data.Y_test - m
    std = v**0.5
    cal = (d < 1.96 * std) * (d > -1.96 * std)
    
    du = d * data.Y_std

    res['test_mae'] = np.average(np.abs(d))
    res['test_mae_unnormalized'] = np.average(np.abs(du))

    res['test_rmse'] = np.average(d**2)**0.5
    res['test_rmse_unnormalized'] = np.average(du**2)**0.5

    res['test_calibration'] = np.average(cal)

    res.update(ARGS.__dict__)
    
    if not is_test:  # pragma: no cover
        with Database(ARGS.database_path) as db:
            db.write('regression', res)

    return res


if __name__ == '__main__':
    run(parse_args())
