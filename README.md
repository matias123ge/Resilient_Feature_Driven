# feature-deletion-robust-regression

Author: Matias Kühnau 

Special thanks to Akylas Stratigakos for the base code and mentoring. 

Repository for resilient energy forecasting against missing features at test time.

- ``FDR_regressor.py``: the main script for feature deletion-robust regression. The user specifies the selected quantile, robustness budget, and the solution method (reformulation, affinely adjustable reformulation).
- ``FDR_group_regressor.py``: additional functionality to declare groups of features. Robustness budget must be selected accordingly.

## Applications

Four prevalent energy forecasting applications are examined: electricity price, load, wind production, and solar production forecasting. For load and wind, features are deleted in groups.

### How to run

Each script includes a set of configuration parameters in function  `params`:
- ``params['scale']``: train the FDRR model (else will try to load a trained model).
- ``params['save']``: save trained FDRR model, results, and plots.
- ``params['impute']``: whether to use imputation for the benchmarks  (`True` everywhere).
- ``params['scale']``: whether to scale everything prior to training (`True` everywhere).

See  ``requirements.txt``  for dependencies.

### Electricity prices

*Data*: ENTO-E Transparency Platform (```market_data_download.py``)

*Scripts*:

- ``main_DA_trading.py``: compares FDRR against a number of benchmark models for day ahead trading.

- ``main_DA_trading_allfeats.py``: compares FDRR against a number of benchmark models for day ahead trading using all features.

- ``main_ID_trading_dual.py``: compares FDRR against a number of benchmark models for intra day trading using all features.

The Preliminaries folder contains preliminary case studies. 

Main research results are in Da-case, where baseline is a day-ahead case study with two price balancing. 
No trained models are available in the repo, reach out to me for these. 
