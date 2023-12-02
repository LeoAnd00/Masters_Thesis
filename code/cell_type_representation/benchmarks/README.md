# benchmarks
Contains all code used in this project, including preprocessing, visualization, machine learning models and more.

## Structur
- **results:** Contains all benchmarking results from the different data sets.
- **benchmark.py:** Code for benchmarking models on specified data set. Methods that are benchmarked: *Unintegrated*, *PCA*, *Scanorama*, *Harmony*, *scVI*, *scANVI*, *scGen*, *ComBat*, *DESC*, *FastMNN* and *In-House Models*.
- **benchmark_on_testdata.py:** Code for evaluating generalizability of models on benchmark metrics when varying the number of patients data for training.