# data
All data used in this project are located in this folder, consisting of scRNA-seq data, pathway information and *Cell Painting* images.

## Structur
- **raw:** Contains all raw scRNA-seq data.
    - **immune_cells:** Data related to immune cells.
        - **bone_marrow_human:** scRNA-seq data samples taken from bone marrow of human subjects.
        - **pbmcs_human:** scRNA-seq data samples taken from peripheral blood of human subjects.
    - **pancreas_cells:** Data related to pancreas cells.
    - **kidney_cells:** Data related to kidney cells.
    - **pathway_information:** Contains information on which human genes are realted in different pathways in human cells.
- **processed:** Contains all preprocessed scRNA-seq data.
    - **immune_cells:** Data related to immune cells.
        - **bone_marrow_human:** Preprocessed scRNA-seq data samples taken from bone marrow of human subjects.
        - **pbmcs_human:** Preprocessed scRNA-seq data samples taken from peripheral blood of human subjects.
        - **merged:** Merged AnnData objects from *bone_marrow_human* and *pbmcs_human*.
    - **pancreas_cells:** Data related to pancreas cells.
    - **kidney_cells:** Data related to kidney cells.
    - **merged:** AnnData object where all data have been merged.
    - **pathway_information:** Contains information on which human genes are realted in different pathways in human cells after being filtered.