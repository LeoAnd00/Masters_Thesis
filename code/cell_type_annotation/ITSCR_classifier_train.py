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
import scTRAC.scTRAC as scTRAC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class classifier_train():
    """
    A class for benchmarking single-cell RNA-seq data integration methods.

    Parameters
    ----------
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
    pathway_path: str, optional
        The path to pathway/gene set information.
    gene2vec_path: str, optional
        The path to gene2vec representations.
    image_path : str, optional
        The path to save UMAP images.
    batch_key : str, optional
        The batch key to use for batch effect information (default is "patientID").
    label_key : str, optional
        The label key containing the cell type information (default is "cell_type").
    HVG : bool, optional 
        Whether to select highly variable genes (HVGs) (default is True).
    HVGs : int, optional
        The number of highly variable genes to select if HVG is enabled (default is 4000).
    num_patients_for_training : int, optional
        The number of patients/samples to use for training.
    num_patients_for_testing : int, optional
        The number of patients/samples to use for testing.
    Scaled : bool, optional
        Whether to scale the data so that the mean of each feature becomes zero and std becomes the approximate std of each individual feature (default is False).
    seed : int, optional
        Which random seed to use (default is 42).

    Methods
    -------

    """

    def __init__(self, 
                 data_path: str, 
                 pathway_path: str='../../data/processed/pathway_information/all_pathways.json',
                 gene2vec_path: str='../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt',
                 image_path: str='',
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 num_folds: int=5,
                 fold: int=1,
                 seed: int=42,
                 select_patients_seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        self.adata = adata

        self.label_key = label_key
        self.pathway_path = pathway_path
        self.gene2vec_path = gene2vec_path
        self.image_path = image_path
        self.seed = seed
        self.HVGs = HVGs

        self.metrics = None
        self.metrics_Model1 = None
        self.metrics_Model2 = None
        self.metrics_Model3 = None
        self.metrics_TOSICA = None

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

        rep_seed(select_patients_seed)

        # Initialize Stratified K-Fold
        stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=select_patients_seed)

        # Iterate through the folds
        self.adata = self.adata.copy()
        self.test_adata = self.adata.copy()
        fold_counter = 0
        for train_index, test_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.label_key]):
            fold_counter += 1
            if fold_counter == fold:
                self.adata = self.adata[train_index, :].copy()
                self.test_adata = self.test_adata[test_index, :].copy()
                break

        self.original_adata = self.adata.copy()
        self.original_test_adata = self.test_adata.copy()

        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.test_adata = self.test_adata[:, self.adata.var["highly_variable"]].copy()
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell type'
        self.batcheffect_title = 'Batch effect'

    def tosica(self):
        """
        Evaluate and visualization on performance of TOSICA (https://github.com/JackieHanLab/TOSICA/tree/main) on single-cell RNA-seq data.

        Parameters
        ----------

        Returns
        -------
        None

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        #import TOSICA
        import TOSICA.TOSICA as TOSICA
        #from TOSICA.TOSICA import TOSICA as TOSICA

        adata_tosica = self.adata.copy()
        TOSICA.train(adata_tosica, gmt_path='human_gobp', label_name=self.label_key,project='hGOBP_TOSICA')

        model_weight_path = './hGOBP_TOSICA/model-9.pth'
        adata_tosica = self.test_adata.copy()
        new_adata = TOSICA.pre(adata_tosica, model_weight_path = model_weight_path,project='hGOBP_TOSICA')

        adata_tosica.obs[f"{self.label_key}_prediction"] = new_adata.obs['Prediction'].copy()

        del new_adata

        # Extract the unique labels
        unique_labels1 = np.unique(adata_tosica.obs[self.label_key])
        unique_labels2 = np.unique(adata_tosica.obs[f"{self.label_key}_prediction"])
        unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2]))
        np.random.shuffle(unique_labels)

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
        y_true = label_encoder_temp.transform(adata_tosica.obs[self.label_key])
        y_pred = label_encoder_temp.transform(adata_tosica.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy:", balanced_accuracy)

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        print("F1 Score:", f1)

        # Creating a metrics dataFrame
        self.metrics_TOSICA = pd.DataFrame({
                                            'Accuracy': [accuracy],
                                            'Balanced Accuracy': [balanced_accuracy],
                                            'F1 Score': [f1]
                                           })
        
        del adata_tosica

    def Model1_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_encoder.py model on single-cell RNA-seq data.

        Parameters
        ----------
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

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        adata_in_house = self.original_adata.copy()

        model = scTRAC.scTRAC(target_key=self.label_key,
                              latent_dim=100,
                              batch_key="batch",
                              model_name="Model1",
                              model_path=save_path)
        
        if train:
            model.train(adata=adata_in_house, train_classifier=True, optimize_classifier=True, num_trials=100, only_print_best=True)
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = model.predict(adata=adata_in_house_test)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions = model.predict(adata=adata_in_house_test, use_classifier=True, detect_unknowns=False)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        random_order = np.random.permutation(adata_in_house_test.n_obs)
        adata_in_house_test = adata_in_house_test[random_order, :]

        # Extract the unique labels
        unique_labels1 = np.unique(adata_in_house_test.obs[self.label_key])
        unique_labels2 = np.unique(adata_in_house_test.obs[f"{self.label_key}_prediction"])
        unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2]))
        np.random.shuffle(unique_labels)

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
        y_true = label_encoder_temp.transform(adata_in_house_test.obs[self.label_key])
        y_pred = label_encoder_temp.transform(adata_in_house_test.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy:", balanced_accuracy)

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        print("F1 Score:", f1)

        # Creating a metrics dataFrame
        self.metrics_Model1 = pd.DataFrame({
                                            'Accuracy': [accuracy],
                                            'Balanced Accuracy': [balanced_accuracy],
                                            'F1 Score': [f1]
                                           })

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

    def Model2_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_tokenized_hvg_transformer_with_pathways.py model on single-cell RNA-seq data.

        Parameters
        ----------
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

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """

        adata_in_house = self.original_adata.copy()

        HVG_buckets_ = 1000

        HVGs_num = self.HVGs

        model = scTRAC.scTRAC(target_key=self.label_key,
                              latent_dim=100,
                              batch_key="batch",
                              model_name="Model2",
                              model_path=save_path)
        
        if train:
            model.train(adata=adata_in_house, train_classifier=True, optimize_classifier=True, num_trials=100, only_print_best=True)
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = model.predict(adata=adata_in_house_test)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions = model.predict(adata=adata_in_house_test, use_classifier=True, detect_unknowns=False)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        random_order = np.random.permutation(adata_in_house_test.n_obs)
        adata_in_house_test = adata_in_house_test[random_order, :]

        # Extract the unique labels
        unique_labels1 = np.unique(adata_in_house_test.obs[self.label_key])
        unique_labels2 = np.unique(adata_in_house_test.obs[f"{self.label_key}_prediction"])
        unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2]))
        np.random.shuffle(unique_labels)

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
        y_true = label_encoder_temp.transform(adata_in_house_test.obs[self.label_key])
        y_pred = label_encoder_temp.transform(adata_in_house_test.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy:", balanced_accuracy)

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        print("F1 Score:", f1)

        # Creating a metrics dataFrame
        self.metrics_Model2 = pd.DataFrame({
                                            'Accuracy': [accuracy],
                                            'Balanced Accuracy': [balanced_accuracy],
                                            'F1 Score': [f1]
                                           })

        if umap_plot:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types")
        if save_figure:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Model2_cell_type.svg")
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Model2_batch_effect.svg")
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types", show=False, save=f"{self.image_path}Model2_predicted_cell_type.svg")
        
        del adata_in_house_test

    def Model3_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_tokenized_hvg_transformer_with_pathways.py model on single-cell RNA-seq data.

        Parameters
        ----------
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

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """

        adata_in_house = self.original_adata.copy()

        HVG_buckets_ = 1000

        HVGs_num = self.HVGs

        model = scTRAC.scTRAC(target_key=self.label_key,
                              latent_dim=100,
                              batch_key="batch",
                              model_name="Model3",
                              model_path=save_path)
        
        if train:
            model.train(adata=adata_in_house, train_classifier=True, optimize_classifier=True, num_trials=100, only_print_best=True)
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = model.predict(adata=adata_in_house_test)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions = model.predict(adata=adata_in_house_test, use_classifier=True, detect_unknowns=False)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        random_order = np.random.permutation(adata_in_house_test.n_obs)
        adata_in_house_test = adata_in_house_test[random_order, :]

        # Extract the unique labels
        unique_labels1 = np.unique(adata_in_house_test.obs[self.label_key])
        unique_labels2 = np.unique(adata_in_house_test.obs[f"{self.label_key}_prediction"])
        unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2]))
        np.random.shuffle(unique_labels)

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
        y_true = label_encoder_temp.transform(adata_in_house_test.obs[self.label_key])
        y_pred = label_encoder_temp.transform(adata_in_house_test.obs[f"{self.label_key}_prediction"])

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy:", balanced_accuracy)

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        print("F1 Score:", f1)

        # Creating a metrics dataFrame
        self.metrics_Model3 = pd.DataFrame({
                                            'Accuracy': [accuracy],
                                            'Balanced Accuracy': [balanced_accuracy],
                                            'F1 Score': [f1]
                                           })

        if umap_plot:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types")
        if save_figure:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Model3_cell_type.svg")
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Model3_batch_effect.svg")
            sc.pl.umap(adata_in_house_test, palette=color_dict, color=f"{self.label_key}_prediction", ncols=1, title=f"Predicted cell types", show=False, save=f"{self.image_path}Model3_predicted_cell_type.svg")
        
        del adata_in_house_test

    def make_benchamrk_results_dataframe(self, min_max_normalize: bool=False):
        """
        Generates a dataframe named 'metrics' containing the performance metrics of different methods.

        Parameters
        ----------
        min_max_normalize : bool, optional
            If True, performs min-max normalization on the metrics dataframe (default is False).

        Returns
        -------
        None

        Notes
        -----
        This method consolidates performance metrics from various methods into a single dataframe.
        If min_max_normalize is True, the metrics dataframe is normalized between 0 and 1.
        """

        calculated_metrics = []
        calculated_metrics_names = []
        if self.metrics_Model1 is not None:
            calculated_metrics.append(self.metrics_Model1)
            calculated_metrics_names.append("Model1")
        if self.metrics_Model2 is not None:
            calculated_metrics.append(self.metrics_Model2)
            calculated_metrics_names.append("Model2")
        if self.metrics_Model3 is not None:
            calculated_metrics.append(self.metrics_Model3)
            calculated_metrics_names.append("Model3")
        if self.metrics_TOSICA is not None:
            calculated_metrics.append(self.metrics_TOSICA)
            calculated_metrics_names.append("TOSICA")

        if len(calculated_metrics_names) != 0:
            metrics = pd.concat(calculated_metrics, axis="columns")

            metrics = metrics.set_axis(calculated_metrics_names, axis="rows")

            if self.metrics is None:
                self.metrics = metrics#.sort_values(by='Overall', ascending=False)
            else:
                self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()

        self.metrics = self.metrics.sort_values(by='Accuracy', ascending=False)

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



