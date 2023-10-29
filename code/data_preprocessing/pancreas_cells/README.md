# pancreas_cells
Contains all code for preprocessing the scRNA-seq data of human pancreas cells.

## How to use notebooks
- **Step 1:** Start by runing *pancreas_preprocess.ipynb* to perform QC and normalization.
- **Step 2:** Data is then annotated with cell types using *pancreas_ScType_CellType_Labeling.Rmd* using the [ScType](https://github.com/IanevskiAleksandr/sc-type/tree/master) library.
- **Step 3:** Run *pancreas_apply_labels.ipynb* to add the cell type annotations to the AnnData object.