# data_preprocessing
Contains all code for preprocessing the scRNA-seq data.

## How to use notebooks in each subfolder *X*
- **Step 1:** Start by runing *X*_preprocess.ipynb to perform QC and normalization.
- **Step 2:** Data is then annotated with cell types using *X*_ScType_CellType_Labeling.Rmd using the [ScType](https://github.com/IanevskiAleksandr/sc-type/tree/master) library.
- **Step 3:** Run *X*_apply_labels.ipynb to add the cell type annotations to the AnnData object.