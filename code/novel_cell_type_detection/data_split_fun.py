import scanpy as sc
import torch
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

def split_data(data_path: str, 
               save_path: str, 
               folds: int=5,
               batch_key: str="patientID", 
               label_key: str="cell_type",
               seed: int=42,
               HVG: bool=False,
               HVGs: int=2000):
    
    adata = sc.read(data_path, cache=True)

    adata.obs["batch"] = adata.obs[batch_key]

    # Ensure reproducibility
    def rep_seed(seed):
        # Check if a GPU is available
        if torch.cuda.is_available():
            # Set the random seed for PyTorch CUDA (GPU) operations
            torch.cuda.manual_seed(seed)
            # Set the random seed for all CUDA devices (if multiple GPUs are available)
            torch.cuda.manual_seed_all(seed)
        
        # Set the random seed for CPU-based PyTorch operations
        torch.manual_seed(seed)
        
        # Set the random seed for NumPy
        np.random.seed(seed)
        
        # Set the random seed for Python's built-in 'random' module
        random.seed(seed)
        
        # Set the random seed for TensorFlow
        tf.random.set_seed(seed)
        
        # Set CuDNN to deterministic mode for PyTorch (GPU)
        torch.backends.cudnn.deterministic = True
        
        # Disable CuDNN's benchmarking mode for deterministic behavior
        torch.backends.cudnn.benchmark = False

    rep_seed(seed)

    # Initialize Stratified K-Fold
    stratified_kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Iterate through the folds
    adata_original = adata.copy()
    test_adata_original = adata.copy()
    fold_counter = 0
    for train_index, test_index in stratified_kfold.split(adata_original.X, adata_original.obs[label_key]):
        fold_counter += 1
        
        adata = adata_original[train_index, :].copy()
        test_adata = test_adata_original[test_index, :].copy()

        if HVG:
            sc.pp.highly_variable_genes(adata, n_top_genes=HVGs, flavor="cell_ranger")
            test_adata = test_adata[:, adata.var["highly_variable"]]
            adata = adata[:, adata.var["highly_variable"]]

        # Define cell types to exclude
        exclude_cell_types = ['Mature_B_Cells', 
                            'Plasma_Cells', 
                            'alpha-beta_T_Cells', 
                            'gamma-delta_T_Cells_1', 
                            'gamma-delta_T_Cells_2']

        # Create a boolean mask to select cells that are not in the exclude list
        mask = ~adata.obs['cell_type'].isin(exclude_cell_types)

        # Apply the mask to AnnData object
        adata = adata[mask]

        # Download split data
        adata.write(f'{save_path}_train_fold_{fold_counter}.h5ad')
        test_adata.write(f'{save_path}_test_fold_{fold_counter}.h5ad')