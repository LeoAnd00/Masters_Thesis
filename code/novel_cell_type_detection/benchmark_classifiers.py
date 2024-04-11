# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
import torch
import random
import tensorflow as tf
import warnings
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import CELLULAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class classifier_train():
    """
    A class for training model1 for novel cell type detection.

    Parameters
    ----------
    exclude_cell_types : list
        A list containing cell types to exclude.
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
    dataset_name : str 
        Name of dataset.
    image_path : str, optional
        The path to save UMAP images.
    batch_key : str, optional
        The batch key to use for batch effect information (default is "patientID").
    label_key : str, optional
        The label key containing the cell type information (default is "cell_type").
    HVG : bool, optional 
        Whether to select highly variable genes (HVGs) (default is True).
    HVGs : int, optional
        The number of highly variable genes to select if HVG is enabled (default is 2000).
    num_folds : int, optional
        Number of folds for cross testing
    fold : int, optional
        Which fold to use.
    seed : int, optional
        Which random seed to use (default is 42).
    """

    def __init__(self, 
                 exclude_cell_types,
                 data_path: str, 
                 dataset_name: str,
                 image_path: str='',
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 num_folds: int=5,
                 fold: int=1,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        if dataset_name != "MacParland":
            del adata.layers['log1p_counts']

        self.adata = adata

        self.label_key = label_key
        self.image_path = image_path
        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold
        self.dataset_name = dataset_name

        self.metrics = None
        self.metrics_Model1 = None

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

        rep_seed(self.seed)

        # Initialize Stratified K-Fold
        stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Define cell types to exclude
        self.exclude_cell_types = exclude_cell_types

        # Iterate through the folds
        self.adata = self.adata.copy()
        self.test_adata = self.adata.copy()
        fold_counter = 0
        for train_index, test_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.label_key]):
            fold_counter += 1
            if fold_counter == fold:
                self.adata = self.adata[train_index, :].copy()
                self.test_adata = self.test_adata[test_index, :].copy()

                # Create a boolean mask to select cells that are not in the exclude list
                mask = ~self.adata.obs['cell_type'].isin(self.exclude_cell_types)

                # Apply the mask to AnnData object
                self.adata = self.adata[mask]

                break

        self.original_adata = self.adata.copy()
        self.original_test_adata = self.test_adata.copy()

        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.test_adata = self.test_adata[:, self.adata.var["highly_variable"]].copy()
            self.test_adata.var["highly_variable"] = self.adata.var["highly_variable"].copy()
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell type'
        self.batcheffect_title = 'Batch effect'

    def Model1_classifier(self, threshold: float, excluded_cell: str, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of model1 on single-cell RNA-seq data.

        Parameters
        ----------
        threshold : float
            Threshold value to use for novel cell type detection likelihood limit.
        excluded_cell : str
            Which cell type that is exclude during training.
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool, optional
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool, optional
            Whether to train the model (True) or use a existing model (False) (default: True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

        Returns
        -------
        None
        """
        save_path = f"{save_path}Fold_{self.fold}/"

        adata_in_house = self.original_adata.copy()

        if train:
            CELLULAR.train(adata=adata_in_house, model_path=save_path, train_classifier=True, target_key=self.label_key, batch_key="batch")
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions, pred_prob = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path, use_classifier=True, return_pred_probs=True)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions
        adata_in_house_test.obs[f"{self.label_key}_probability"] = pred_prob

        del predictions, pred_prob

        # Define to be novel if confidence is below threshold
        max_scores = adata_in_house_test.obs[f"{self.label_key}_probability"]
        for i, max_score in enumerate(max_scores):
            if max_score < threshold:
                adata_in_house_test.obs[f"{self.label_key}_prediction"].iloc[i] = "Novel"

        results_novel = adata_in_house_test[adata_in_house_test.obs[self.label_key].isin(self.exclude_cell_types)]
        results_novel.obs[self.label_key] = ["Novel"]*len(results_novel.obs[self.label_key])

        results_not_novel = adata_in_house_test[~adata_in_house_test.obs[self.label_key].isin(self.exclude_cell_types)]

        # Extract the unique labels
        unique_labels1 = np.unique(results_not_novel.obs[self.label_key])
        unique_labels2 = np.unique(results_not_novel.obs[f"{self.label_key}_prediction"])
        unique_labels3 = np.unique(results_novel.obs[f"{self.label_key}_prediction"])
        unique_labels4 = np.unique(results_novel.obs[self.label_key])
        unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2,unique_labels3,unique_labels4]))

        # Get the Tab20 color map with the number of colors you need
        tab20_colors = cm.get_cmap('gist_ncar', (len(unique_labels)+2))

        # Define your color palette
        your_color_palette = [tab20_colors(i) for i in range(len(unique_labels))]

        # Define your own colormap dictionary
        color_dict = dict(zip(unique_labels, your_color_palette))

        # Create a colormap object using LinearSegmentedColormap
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", your_color_palette)

        # Convert string labels to numerical labels
        label_encoder_temp = LabelEncoder()
        label_encoder_temp.fit(unique_labels)

        y_true = label_encoder_temp.transform(results_not_novel.obs[self.label_key])
        y_pred = label_encoder_temp.transform(results_not_novel.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy_not_novel = accuracy_score(y_true, y_pred)

        # Calculate balanced accuracy
        balanced_accuracy_not_novel = balanced_accuracy_score(y_true, y_pred)

        # Calculate F1 score
        f1_not_novel = f1_score(y_true, y_pred, average='weighted')

        y_true = label_encoder_temp.transform(results_novel.obs[self.label_key])
        y_pred = label_encoder_temp.transform(results_novel.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy_novel = accuracy_score(y_true, y_pred)

        # Calculate balanced accuracy
        balanced_accuracy_novel = balanced_accuracy_score(y_true, y_pred)

        # Calculate F1 score
        f1_novel = f1_score(y_true, y_pred, average='weighted')

        # Creating a metrics dataFrame
        self.metrics_Model1 = pd.DataFrame({"method": "Model1", 
                                    "accuracy": accuracy_not_novel, 
                                    "balanced_accuracy": balanced_accuracy_not_novel,
                                    "f1_score": f1_not_novel,
                                    "dataset": self.dataset_name,
                                    "novel": "Known",
                                    "fold": self.fold,
                                    "excluded_cell_type": excluded_cell,
                                    "threshold":threshold}, 
                                    index=[0])
        temp = pd.DataFrame({"method": "Model1", 
                            "accuracy": accuracy_novel, 
                            "balanced_accuracy": balanced_accuracy_novel,
                            "f1_score": f1_novel,
                            "dataset": self.dataset_name,
                            "novel": "Novel",
                            "fold": self.fold,
                            "excluded_cell_type": excluded_cell,
                            "threshold":threshold}, 
                            index=[0])
        self.metrics_Model1 = pd.concat([self.metrics_Model1, temp], ignore_index=True)

        random_order = np.random.permutation(adata_in_house_test.n_obs)
        adata_in_house_test = adata_in_house_test[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types")
        if save_figure:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Model1_cell_type.svg")
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Model1_batch_effect.svg")
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types", show=False, save=f"{self.image_path}Model1_predicted_cell_type.svg")
            
        del adata_in_house_test

    def make_benchamrk_results_dataframe(self):
        """
        Generates a dataframe named 'metrics' containing the performance metrics of different methods.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method consolidates performance metrics from various methods into a single dataframe.
        """

        calculated_metrics = []
        calculated_metrics_names = []
        if self.metrics_Model1 is not None:
            calculated_metrics.append(self.metrics_Model1)
            calculated_metrics_names.append("Model1")

        if len(calculated_metrics_names) != 0:
            metrics = pd.concat(calculated_metrics, axis="columns")

            if self.metrics is None:
                self.metrics = metrics
            else:
                self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()

        self.metrics = self.metrics.sort_values(by='accuracy', ascending=False)

    def save_results_as_csv(self, name: str='benchmarks/results/Benchmark_results'):
        """
        Saves the performance metrics dataframe as a CSV file.

        Parameters
        ----------
        name : str, optional
            The file path and name for the CSV file (default is 'benchmarks/results/Benchmark_results' (file name will then be Benchmark_results.csv)).

        Returns
        -------
        None

        Notes
        -----
        This method exports the performance metrics dataframe to a CSV file.
        """
        self.metrics.to_csv(f'{name}.csv', index=True, header=True)
        self.metrics = None

    def read_csv(self, name: str='benchmarks/results/Benchmark_results'):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        name : str, optional
            The file path and name of the CSV file to read (default is 'benchmarks/results/Benchmark_results').

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """
        if self.metrics is not None:
            metrics = pd.read_csv(f'{name}.csv', index_col=0)
            self.metrics = pd.concat([self.metrics, metrics], axis="rows")
        else:
            self.metrics = pd.read_csv(f'{name}.csv', index_col=0)



