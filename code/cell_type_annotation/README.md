**cell_type_annotation:** Contains code for performing the cell type annotation benchmark.
1. Start by running *Make_data_splits.ipynb* to generate the training and test data folds.
2. Run all codes in */Baron*, */MacParland*, */Segerstolpe*, and */Zheng68k* for scNym, Seurat, SciBet, and CellID.
3. Code for TOSICA and the model developed in this study can be found in the *Alvis_cluster_code* repo.
4. Run *Calc_metrics.ipynb* to calculate accuracy, balanced accuracy and F1 score on the test data of each fold.
5. Manually move results of the model developed in this study and TOSICA into the *results_and_visualizations/results_benchmark.csv* file produced from step 4.
6. Run *visualize_metrics.ipynb* to make visualization plots of results. 