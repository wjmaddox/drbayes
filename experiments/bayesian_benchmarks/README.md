# Bayesian Benchmarks

**This is a clone of Hugh Salimbeni's Bayesian benchmarks repo, found [here](https://github.com/hughsalimbeni/bayesian_benchmarks)** 

## Experiments

[Model definitions](models/nnet/models.py)

[Scripts](tasks/swag_regression.py) 

[Results](results/)

## Other stuff

This is a set of tools for evaluating Bayesian models, together with benchmark implementations and results.

Motivations:
* There is a lack of standardized tasks that meaningfully assess the quality of uncertainty quantification for Bayesian black-box models.
* Variations between tasks in the literature make a direct comparison between methods difficult.
* Implementing competing methods takes considerable effort, and there little incentive to do a good job.
* Published papers may not always provide complete details of implementations due to space considerations.

Aims:
* Curate a set of benchmarks that meaningfully compare the efficacy of Bayesian models in real-world tasks.
* Maintain a fair assessment of benchmark methods, with full implementations and results.

Tasks:
* Classification and regression

Current implementations:
* SWAG and DR Bayes
* Sparse variational GP, for [Gaussian](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf) and [non-Gaussian](http://proceedings.mlr.press/v38/hensman15.pdf) likelihoods
* Sparse variational GP, with [minibatches](https://arxiv.org/pdf/1309.6835.pdf)
* 2 layer Deep Gaussian process, with [doubly-stochastic variational inference](http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes.pdf)
* A variety of sklearn models
