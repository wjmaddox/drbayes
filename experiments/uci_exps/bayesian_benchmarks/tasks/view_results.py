import sys
#sys.path.append('../../')

import numpy as np
import pandas
import pickle
#from matplotlib import pyplot as plt
from scipy.stats import rankdata

from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.data import classification_datasets, _ALL_REGRESSION_DATATSETS, _ALL_CLASSIFICATION_DATATSETS
ALL_DATATSETS = {}
ALL_DATATSETS.update(_ALL_REGRESSION_DATATSETS)
ALL_DATATSETS.update(_ALL_CLASSIFICATION_DATATSETS)
#from bayesian_benchmarks.data import regression_datasets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default=None, required=True, help='path to database (default: None)')
args = parser.parse_args()


def rankarray(A):
    ranks = []
    for a in A:
        ranks.append(rankdata(a))
    return np.array(ranks)


def read_regression_classification(db_loc, fs, models_names, datasets, task):
    fields = ['dataset', 'N', 'D'] + [m[1] for m in models_names]

    results = {}
    for f in fs:
        results[f] = {'table':{f:[] for f in fields}, 'vals':[]}

    with Database(db_loc) as db:

        for dataset in datasets:
            for f in fs:
                results[f]['table']['dataset'].append(dataset[:10])
                results[f]['table']['N'].append(ALL_DATATSETS[dataset].N)
                results[f]['table']['D'].append(ALL_DATATSETS[dataset].D)

            row = {f:[] for f in fs}
            for model, name in models_names:
                res = db.read(task, fs, {'model':model, 
                                         'dataset':dataset})

                if len(res) == 0:
                    for f in fs:
                        results[f]['table'][name].append('')
                        row[f].append(np.nan)
                else:
                    print('{} {} {}'.format(model, dataset, len(res)))
                    for i, f in enumerate(fs):
                        L = [np.nan if l[i] is None else float(l[i]) for l in res]
                        m = np.nanmean(L)
                        std = np.nanstd(L) if len(L) > 1 else np.nan
                        if m < 1000 and m > -1000:
                            r = '{:.3f}({:.3f})'.format(m, std)
                            row[f].append(m)
                        else:
                            r = 'nan'
                            row[f].append(np.nan)

                        results[f]['table'][name].append(r)

            for f in fs:   
                results[f]['vals'].append(row[f])


    for f in fs:
        if 'unnormalized' not in f:
            vals = np.array(results[f]['vals'])

            avgs = np.nanmean(vals, 0)
            meds = np.nanmedian(vals, 0)
            rks = np.nanmean(rankarray(vals), 0)

            for s, n in [[avgs, 'avg'], [meds, 'median'], [rks, 'avg rank']]:
                results[f]['table']['dataset'].append(n)
                results[f]['table']['N'].append('')
                results[f]['table']['D'].append('')
                if task == 'classification':
                    results[f]['table']['K'].append('')
                for ss, name in zip(s, [m[1] for m in models_names]):
                    results[f]['table'][name].append('{:.3f}'.format(ss))
    
    return results, fields


models_names = [['RegNet', 'SGD'], ['RegNetNL_LP', 'NL'], ['RegNetpcaess', 'PCA+ESS'], ['RegNetpcavi', 'PCA+VI'],
               ['RegNetpcalow_rank_gaussian', 'SWAG']]
#regression_datasets = ['wilson_elevators', 'wilson_keggdirected', 'wilson_keggundirected', 'wilson_protein',
#		'wilson_pol', 'wilson_skillcraft'
#		]
regression_datasets = ['boston', 'concrete', 'energy', 'naval', 'yacht']

fs = ['test_loglik', 'test_rmse', 'test_loglik_unnormalized', 'test_rmse_unnormalized', 'test_calibration']

results, fields = read_regression_classification(args.database, fs, models_names, regression_datasets, 'regression')


print('normalized test loglikelihood')
print(pandas.DataFrame(results['test_loglik']['table'], columns=fields))

print('test loglikelihood')
print(pandas.DataFrame(results['test_loglik_unnormalized']['table'], columns=fields))


print('normalised test rmse')
print(pandas.DataFrame(results['test_rmse']['table'], columns=fields))

print('test rmse')
print(pandas.DataFrame(results['test_rmse_unnormalized']['table'], columns=fields))

print('test calibration')
print(pandas.DataFrame(results['test_calibration']['table'], columns=fields))

output_file = args.database[:-3]
with open(output_file+'.pkl', 'wb') as f:
	pickle.dump({'results':results, 'fields':fields}, f, pickle.HIGHEST_PROTOCOL)
