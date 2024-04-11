# code
Contains code used in this project, including preprocessing, visualization, machine learning models and more.
More code exist in the *Alvis_cluster_code* repo.

## Structur
- **data_preprocessing:** Contains preprocessing code and visualization of scRNA-seq data. All processed data is stored in the *data/processed/* folder.
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

- **cell_type_representation:** Contains results and visualizations from the scRNA-Seq embedding space benchmark. Code for running the benchmark itself can be found in the *Alvis_cluster_code* repo.
    - **benchmarks:** Contain results from the scRNA-Seq embedding space benchmark from the *Alvis_cluster_code* repo.
    - **visualizations:** Visualizations for report. Both for the generalizable embedding space and for invetigating the validity of the cell type centroid loss.
 
- **cell_type_annotation:** Contains code for performing the cell type annotation benchmark.
    1. Start by running *Make_data_splits.ipynb* to generate the training and test data folds.
    2. Run all codes in */Baron*, */MacParland*, */Segerstolpe*, and */Zheng68k* for scNym, Seurat, SciBet, and CellID.
    3. Code for TOSICA and the model developed in this study can be found in the *Alvis_cluster_code* repo.
    4. Run *Calc_metrics.ipynb* to calculate accuracy, balanced accuracy and F1 score on the test data of each fold.
    5. Add results of the model developed in this study and TOSICA into the *results_and_visualizations/results_benchmark.csv* file produced from step 4.
    6. Run *visualize_metrics.ipynb* to make visualization plots of results. 

- **cell_type_annotation_loss_comp:** Contains code for visualizing the comparison between different loss functions.
    1. Code for training the model on the different loss functions can be found in the *Alvis_cluster_code* repo.
    2. Move resulting folder for each dataset containing .csv files into the */cell_type_annotation_loss_comp* path in the *main* repo.
    3. Run *Calc_metrics.ipynb* to concatenate all cell type annotation results into */results_and_visualizations*
    4. Run *visualize_metrics.ipynb* to make visualizations.

- **novel_cell_type_detection:** Contains code for visualizing novel cell type detection function of the model.
    1. Code for training model is in the *Alvis_cluster_code* repo. 
    2. Code for extracting likelihood of all predictions of all datasets, cell type exclusions, and fold can also be found in the *Alvis_cluster_code* repo. This will give a *likelihood.json* file.
    3. Move *likelihood.json* to */novel_cell_type_detection/results* in the *main* repo.
    4. Simply run *novel_cell_type_confidence.ipynb* to make visualizations.
