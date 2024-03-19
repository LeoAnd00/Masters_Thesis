# code
Contains all code used in this project, including preprocessing, visualization, machine learning models and more.

## Structur
- **data_preprocessing:** Contains all preprocessing code of scRNA-seq data. All processed data is stored in the *data/processed/* folder.
    - **automated_preprocess:** Code one can run in cmd that automatically preprocess the bone marrow, PBMC, pancreas, kidney and merged datasets.
    - **immune_cells:** Code for preprocessing data related to immune cells.
        - **bone_marrow_human:** Preprocessing code for scRNA-seq data of the bone marrow dataset.
        - **pbmcs_human:** Preprocessing code for scRNA-seq data of the PBMC dataset.
        - **merge:** Code to merge all AnnData objects from *bone_marrow_human* and *pbmcs_human*.
        - **automated_preprocess:** Code one can run in cmd that automatically preprocess all immune cells data.
    - **pancreas_cells:** Code for preprocessing data of the pancreas dataset.
    - **kidney_cells:** Code for preprocessing data of the kidney dataset.
    - **pathway_information:** Contains preprocessing code for human pathway information (gene sets). Run *gene_sets_filtering.ipynb* to process the gene sets.
    - **data_for_evaluating_cell_type_annotation:** Code for processing the Baron, MacParland, Segerstolpe, and Zheng68k datasets.

- **cell_type_representation:** Contains all code for training and benchmarking models, including code for the models themselves.
    -**benchmarks:** Code for runing benchmark of model performance.
    -**figures:** Contains figures of UMAPs of latent spaces from different models, colored by cell type and batch effect.
    -**functions:** Code for data preprocessing, training models and producing cell type vector representations from latent space representations.
    -**HPC_cluster:** Code for training models, suitable for HPC clusters.
    -**models:** Model scripts.
    -**trained_models:** Contains trained models for different data sets.