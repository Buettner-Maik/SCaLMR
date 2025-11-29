This is a part rework of the caafa framework https://github.com/Buettner-Maik/caafa-stream rewritten to use river https://github.com/online-ml/river as a baseline for true incremental data streams.

* /caafa/ contains the framework itself
* /paper/ contains further information and material for the submission such as graphs, tables and the consolidated and aggregated results of the experiments.
  * /paper/all_acquisition_tables.pdf for the remaining two result tables
  * /paper/graphs/ for the remaining pareto fronts and cd-plots
  * /paper/run_results/ for the summarized large results and the detailed f_selection and imputer pre_runs

Recreation of the results is pending; RNG seeds:
* miss_mask seeds: `{:100*iteration+i+1 for i, f in enumerate(dataset.features)} | {LABEL_DICT_KEY:10000+iteration}`, iterations are 1 to 10
* random_rbf dataset: `river.synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=2, n_features=10, n_centroids=50) # take 10000 instances`, included in /data/csv/random_rbf

