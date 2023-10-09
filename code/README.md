# code
Contains all code used in this project, including preprocessing, visualization, machine learning models and more.

## Structur
- **data_preprocessing:** Contains all preprocessing code of scRNA-seq data.
    - **immune_cells:** Code for preprocessing data related to immune cells.
        - **bone_marrow_human:** Preprocessing code for scRNA-seq data samples taken from bone marrow of human subjects.
        - **pbmcs_human:** Preprocessing code for scRNA-seq data samples taken from peripheral blood of human subjects.
        - **merge:** Code to merge all AnnData objects from *bone_marrow_human* and *pbmcs_human*.
        - **automated_preprocess:** Code one can run in cmd that automatically preprocess all data, including annotating cell type.
    - **pathway_information:** Contains preprocessing code for human pathway information (gene sets).