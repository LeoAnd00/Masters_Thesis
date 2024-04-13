**data_preprocessing:** Contains preprocessing code and visualization of scRNA-seq data. All processed data is stored in the *data/processed/* folder.
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
- **visualizations:** Code for making UMAP visualizations of data.