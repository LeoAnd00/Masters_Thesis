# benchmarks
Contains all code used in this project, including preprocessing, visualization, machine learning models and more.

## Structur
- **results:** Contains all benchmarking results from the different data sets.
- **benchmark.py:** Code for benchmarking models on specified data set. Methods that are benchmarked: *Unintegrated*, *PCA*, *Scanorama*, *Harmony*, *scVI*, *scANVI*, *scGen*, *ComBat*, *DESC*, *FastMNN* and *In-House Models*.
- **benchmark_generalizability.py:** Code for benchmarking models on generalizability to unseen data. All data is used for training and validation
- **benchmark_generalizability_with_validation.py:** Code for benchmarking models on generalizability to unseen data. 80% of data is used for training and 20% for validation.