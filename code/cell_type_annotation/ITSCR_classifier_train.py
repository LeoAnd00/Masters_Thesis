# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import torch.nn as nn
import torch
import random
import tensorflow as tf
import warnings
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from functions import data_preprocessing as dp
#from functions import train_with_validation as trainer
from functions import train_with_validationV2 as trainer
from sklearn.model_selection import StratifiedKFold
from models import Model1 as Model1
from models import Model2 as Model2
from models import Model3 as Model3

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

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell type'
        self.batcheffect_title = 'Batch effect'

    def Model1_classifier(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
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

        #Model
        model = Model1.Model1(input_dim=self.HVGs,
                                output_dim=100,
                                drop_out=0.2,
                                act_layer=nn.ReLU,
                                norm_layer=nn.BatchNorm1d,
                                include_classifier=False,
                                num_cell_types=len(adata_in_house.obs['cell_type'].unique()))

        train_env = trainer.train_module(data_path=adata_in_house,
                                        save_model_path=save_path,
                                        HVG=True,
                                        HVGs=self.HVGs,
                                        HVG_buckets=1000,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=self.seed,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=11,#110,
                                eval_freq=4,#1,
                                epochs=10,#100,
                                earlystopping_threshold=40)
        
        predictions = train_env.predict(data_=adata_in_house, model=model, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_encoder = scib.metrics.metrics(
            self.original_adata,
            adata_in_house,
            "batch", 
            self.label_key,
            embed="In_house",
            isolated_labels_asw_=True,
            silhouette_=True,
            hvg_score_=True,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=True,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=True,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            organism="human",
        )

        random_order = np.random.permutation(adata_in_house.n_obs)
        adata_in_house = adata_in_house[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_HVG_Encoder_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_HVG_Encoder_batch_effect.svg")

        del adata_in_house

    def Model3_classifier(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
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

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=self.pathway_path,
                                        num_pathways=500,
                                        pathway_gene_limit=10,
                                        save_model_path=save_path,
                                        HVG=True,
                                        HVGs=HVGs_num,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=True,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=True,
                                        gene2vec_path=self.gene2vec_path)
        #Model
        model = Model3.Model3(mask=train_env.data_env.pathway_mask,
                                            num_HVGs=min([HVGs_num,int(train_env.data_env.X.shape[1])]),
                                            output_dim=100,
                                            num_pathways=500,
                                            HVG_tokens=HVG_buckets_,
                                            HVG_embedding_dim=train_env.data_env.gene2vec_tensor.shape[1],
                                            use_gene2vec_emb=True,
                                            include_classifier=False,
                                            num_cell_types=len(adata_in_house.obs['cell_type'].unique()))
                                                                          
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=self.seed,
                                batch_size=236,#256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=110,#25,
                                eval_freq=1,
                                epochs=100,#20,
                                earlystopping_threshold=40,#5)
                                train_classifier=True)
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = train_env.predict(data_=adata_in_house_test, model=model, model_path=save_path, return_attention=False, use_classifier=False)
        adata_in_house.obsm["latent_space"] = predictions

        predictions = train_env.predict(data_=adata_in_house_test, model=model, model_path=save_path, return_attention=False, use_classifier=True)
        predictions = train_env.data_env_for_classification.label_encoder.inverse_transform(predictions).to_list()
        adata_in_house.obsm[f"{self.label_key}_prediction"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house, use_rep="latent_space")

        random_order = np.random.permutation(adata_in_house.n_obs)
        adata_in_house = adata_in_house[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title)
            sc.pl.umap(adata_in_house, color=f"{self.label_key}_prediction", ncols=1, title=self.celltype_title)
        if save_figure:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_ITSCR_Model_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_ITSCR_Model_batch_effect.svg")
            sc.pl.umap(adata_in_house, color=f"{self.label_key}_prediction", ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_ITSCR_Model_predicted_cell_type.svg")
            
        del adata_in_house

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



