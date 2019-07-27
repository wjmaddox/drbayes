# UCI Regression Experiments

**This is a clone of Hugh Salimbeni's Bayesian benchmarks repo, found [here](https://github.com/hughsalimbeni/bayesian_benchmarks)** 


## Data Preparation

In order to run experiments on the UCI-large datasets from the paper, please run `mkdir data` 
from `experiments/uci_exps` folder. 
Then, download `uci.tar.gz` from [Google Drive](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view) and put
it in the `data` folder.
Run `cd data; tar -xzvf uci.tar.gz`.

## Running Experiments

The scripts for reproducing results on UCI problems from the paper can be found at 
`experiments/uci_exps/bayesian_benchmarks/tasks/run_ucilarge.sh` (UCI-large) and
`experiments/uci_exps/bayesian_benchmarks/tasks/run_ucismall.sh` (UCI-small).
You can run these scripts as follows
```bash
cd experiments/uci_exps/bayesian_benchmarks/tasks/
chmod u+x run_ucismall.sh 
./run_ucismall.sh 
```

In the paper we report results averaged over splits 1,...,20, and use splits 21,...,23 for validation.

## Viewing Results
The results of scripts `run_ucismall.sh` and `run_ucilarge.sh` will be saved in databases 
`results_small.db` and `results_large.db` respectively.
You can then view the results using the script `view_results.py` at `experiments/uci_exps/bayesian_benchmarks/tasks/`.
For example, to view results on UCI-small run
```bash
python3 view_results.py --database results_small.db
```
