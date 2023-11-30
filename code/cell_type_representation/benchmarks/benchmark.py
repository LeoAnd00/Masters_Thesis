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
from functions import data_preprocessing as dp
from functions import train as trainer
from models import model_encoder as model_encoder
from models import model_pathway as model_pathway
from models import model_encoder_with_pathway as model_encoder_with_pathway
from models import CustomScaler_model_transformer_encoder as model_transformer_encoder
from models import CustomScaler_model_transformer_encoder_with_pathways as model_transformer_encoder_with_pathways
from models import model_tokenized_hvg_transformer as model_tokenized_hvg_transformer
from models import model_tokenized_hvg_transformer_with_pathways as model_tokenized_hvg_transformer_with_pathways

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class benchmark():
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
                 Scaled: bool=False,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        self.adata = adata
        self.label_key = label_key
        self.pathway_path = pathway_path
        self.gene2vec_path = gene2vec_path
        self.image_path = image_path

        # Initialize variables
        self.metrics = None
        self.metrics_pca = None
        self.metrics_scanorama = None
        self.metrics_scvi = None
        self.metrics_scanvi = None
        self.metrics_harmony = None
        self.metrics_scgen = None
        self.metrics_combat = None
        self.metrics_desc = None
        self.metrics_bbknn = None
        self.metrics_fastmnn = None
        self.metrics_unscaled = None
        self.metrics_in_house_model_encoder = None
        self.metrics_in_house_model_pathways = None
        self.metrics_in_house_model_encoder_pathways = None
        self.metrics_in_house_model_transformer_encoder = None
        self.metrics_in_house_model_transformer_encoder_pathways = None
        self.metrics_in_house_model_tokenized_HVG_transformer = None
        self.metrics_in_house_model_tokenized_HVG_transformer_with_pathways = None

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

        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        if Scaled:
            self.adata.X = dp.scale_data(self.adata.X)

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell type'
        self.batcheffect_title = 'Batch effect'

    def unintegrated(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of unintegrated (Not going through any model) version of single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

        Returns
        -------
        None

        Notes
        -----
        This method computes various metrics to evaluate the quality of an unintegrated version of single-cell RNA-seq data.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the unintegrated data.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        adata_unscaled = self.adata.copy()

        #sc.tl.pca(adata_unscaled, n_comps=n_comps, use_highly_variable=True)
        adata_unscaled.obsm["Unscaled"] = adata_unscaled.X
        sc.pp.neighbors(adata_unscaled, use_rep="Unscaled")

        self.metrics_unscaled = scib.metrics.metrics(
            self.adata,
            adata_unscaled,
            "batch", 
            self.label_key,
            embed="Unscaled",
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

        random_order = np.random.permutation(adata_unscaled.n_obs)
        adata_unscaled = adata_unscaled[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_unscaled)
            #sc.pl.umap(adata_unscaled, color=[self.label_key, "batch"], ncols=1)
            sc.pl.umap(adata_unscaled, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_unscaled, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_unscaled)
            sc.pl.umap(adata_unscaled, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Unintegrated_cell_type.svg")
            sc.pl.umap(adata_unscaled, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Unintegrated_batch_effect.svg")

        del adata_unscaled

    def pca(self, n_comps: int=50, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of PCA on single-cell RNA-seq data.

        Parameters
        ----------
        n_comps : int, optional
            Number of components to retrieve from PCA.
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        adata_pca = self.adata.copy()

        sc.tl.pca(adata_pca, n_comps=n_comps, use_highly_variable=True)
        adata_pca.obsm["PCA"] = adata_pca.obsm["X_pca"]
        sc.pp.neighbors(adata_pca, use_rep="PCA")

        self.metrics_pca = scib.metrics.metrics(
            self.adata,
            adata_pca,
            "batch", 
            self.label_key,
            embed="PCA",
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

        random_order = np.random.permutation(adata_pca.n_obs)
        adata_pca = adata_pca[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_pca)
            sc.pl.umap(adata_pca, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_pca, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_pca)
            sc.pl.umap(adata_pca, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}PCA_cell_type.svg")
            sc.pl.umap(adata_pca, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}PCA_batch_effect.svg")


        del adata_pca

    def scanorama(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of SCANORAMA (version 1.7.4: https://github.com/brianhie/scanorama) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        # import package
        import scanorama

        adata_scanorama = self.adata.copy()

        # List of adata per batch
        batch_cats = adata_scanorama.obs.batch.cat.categories
        adata_list = [adata_scanorama[adata_scanorama.obs.batch == b].copy() for b in batch_cats]
        scanorama.integrate_scanpy(adata_list)

        adata_scanorama.obsm["Scanorama"] = np.zeros((adata_scanorama.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
        for i, b in enumerate(batch_cats):
            adata_scanorama.obsm["Scanorama"][adata_scanorama.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

        sc.pp.neighbors(adata_scanorama, use_rep="Scanorama")

        self.metrics_scanorama = scib.metrics.metrics(
            self.adata,
            adata_scanorama,
            "batch", 
            self.label_key,
            embed="Scanorama",
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

        random_order = np.random.permutation(adata_scanorama.n_obs)
        adata_scanorama = adata_scanorama[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scanorama)
            sc.pl.umap(adata_scanorama, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scanorama, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scanorama)
            sc.pl.umap(adata_scanorama, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Scanorama_cell_type.svg")
            sc.pl.umap(adata_scanorama, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Scanorama_batch_effect.svg")

        del adata_scanorama

    def harmony(self, n_comps: int=30, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of Harmony (version 0.1.7: https://github.com/lilab-bcb/harmony-pytorch) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        # import package
        from harmony import harmonize

        adata_harmony = self.adata.copy()

        sc.tl.pca(adata_harmony, n_comps=n_comps, use_highly_variable=True)

        adata_harmony.obsm["Harmony"] = harmonize(adata_harmony.obsm["X_pca"], adata_harmony.obs, batch_key="batch")

        sc.pp.neighbors(adata_harmony, use_rep="Harmony")

        self.metrics_harmony = scib.metrics.metrics(
            self.adata,
            adata_harmony,
            "batch", 
            self.label_key,
            embed="Harmony",
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

        random_order = np.random.permutation(adata_harmony.n_obs)
        adata_harmony = adata_harmony[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_harmony)
            sc.pl.umap(adata_harmony, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_harmony, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_harmony)
            sc.pl.umap(adata_harmony, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}Harmony_cell_type.svg")
            sc.pl.umap(adata_harmony, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}Harmony_batch_effect.svg")


        del adata_harmony

    def scvi(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of scVI (version 1.0.4: https://github.com/scverse/scvi-tools) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

        Returns
        -------
        vae : The trained scVI model
            Returns the trained scVI model, which can be used as input to the scANVI to save time if one intends to run both scVI and scANVI.

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        # import package
        import scvi

        adata_scvi = self.adata.copy()

        scvi.model.SCVI.setup_anndata(adata_scvi, layer="pp_counts", batch_key="batch")
        vae = scvi.model.SCVI(adata_scvi, gene_likelihood="nb", n_layers=2, n_latent=30)
        vae.train()
        adata_scvi.obsm["scVI"] = vae.get_latent_representation()

        sc.pp.neighbors(adata_scvi, use_rep="scVI")

        self.metrics_scvi = scib.metrics.metrics(
            self.adata,
            adata_scvi,
            "batch", 
            self.label_key,
            embed="scVI",
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

        random_order = np.random.permutation(adata_scvi.n_obs)
        adata_scvi = adata_scvi[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scvi, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}scVI_cell_type.svg")
            sc.pl.umap(adata_scvi, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}scVI_batch_effect.svg")


        del adata_scvi

        return vae

    def scanvi(self, umap_plot: bool=True, vae=None, save_figure: bool=False):
        """
        Evaluate and visualization on performance of scANVI (version 1.0.4: https://github.com/scverse/scvi-tools) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        vae : scVI model
            If you've already trained the scVI model you can use it as input to scANVI here.
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
        # import package
        import scvi

        adata_scANVI = self.adata.copy()

        if vae is None:
            scvi.model.SCVI.setup_anndata(adata_scANVI, layer="pp_counts", batch_key="batch")
            vae = scvi.model.SCVI(adata_scANVI, gene_likelihood="nb", n_layers=2, n_latent=30)
            vae.train()

        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata_scANVI,
            labels_key=self.label_key,
            unlabeled_category="UnknownUnknown",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        adata_scANVI.obsm["scANVI"] = lvae.get_latent_representation()

        del lvae, vae
        sc.pp.neighbors(adata_scANVI, use_rep="scANVI")

        self.metrics_scanvi = scib.metrics.metrics(
            self.adata,
            adata_scANVI,
            "batch", 
            self.label_key,
            embed="scANVI",
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

        random_order = np.random.permutation(adata_scANVI.n_obs)
        adata_scANVI = adata_scANVI[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scANVI)
            sc.pl.umap(adata_scANVI, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scANVI, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scANVI)
            sc.pl.umap(adata_scANVI, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}scANVI_cell_type.svg")
            sc.pl.umap(adata_scANVI, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}scANVI_batch_effect.svg")

        del adata_scANVI

    def scgen(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of scGen (version 2.1.1: https://github.com/theislab/scgen) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        from scgen import SCGEN

        adata_scgen = self.adata.copy()

        SCGEN.setup_anndata(adata_scgen, batch_key="batch", labels_key=self.label_key)
        model = SCGEN(adata_scgen)
        model.train(
            max_epochs=100,
            batch_size=128,
            early_stopping=True,
            early_stopping_patience=10,
        )
        corrected_adata = model.batch_removal()

        adata_scgen.obsm["scGen"] = corrected_adata.obsm["corrected_latent"]

        del corrected_adata
        sc.pp.neighbors(adata_scgen, use_rep="scGen")

        self.metrics_scgen = scib.metrics.metrics(
            self.adata,
            adata_scgen,
            "batch", 
            self.label_key,
            embed="scGen",
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

        random_order = np.random.permutation(adata_scgen.n_obs)
        adata_scgen = adata_scgen[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scgen)
            sc.pl.umap(adata_scgen, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scgen, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scgen)
            sc.pl.umap(adata_scgen, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}scGen_cell_type.svg")
            sc.pl.umap(adata_scgen, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}scGen_batch_effect.svg")

        del adata_scgen

    def combat(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of ComBat (Scanpy version 1.9.5: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.combat.html) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        adata_combat = self.adata.copy()
        corrected_data = sc.pp.combat(adata_combat, key="batch", inplace=False)

        adata_combat.obsm["ComBat"] = corrected_data

        del corrected_data
        sc.pp.neighbors(adata_combat, use_rep="ComBat")

        self.metrics_combat = scib.metrics.metrics(
            self.adata,
            adata_combat,
            "batch", 
            self.label_key,
            embed="ComBat",
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

        random_order = np.random.permutation(adata_combat.n_obs)
        adata_combat = adata_combat[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_combat)
            sc.pl.umap(adata_combat, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_combat, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_combat)
            sc.pl.umap(adata_combat, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}ComBat_cell_type.svg")
            sc.pl.umap(adata_combat, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}ComBat_batch_effect.svg")

        del adata_combat

    def desc(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of DESC (version 2.1.1: https://github.com/eleozzr/desc) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        import desc
        import os

        adata_desc = self.adata.copy()

        adata_out = desc.scale_bygroup(adata_desc, groupby="batch", max_value=6)

        adata_out = desc.train(
            adata_out,
            dims=[adata_desc.shape[1], 128, 32],
            tol=0.001,
            n_neighbors=10,
            batch_size=256,
            louvain_resolution=0.8,
            save_encoder_weights=False,
            #save_dir=tmp_dir,
            do_tsne=False,
            use_GPU=False,
            GPU_id=None,
            num_Cores=os.cpu_count(),
            use_ae_weights=False,
            do_umap=False,
        )

        adata_desc.obsm["DESC"] = adata_out.obsm["X_Embeded_z" + str(0.8)]

        del adata_out
        sc.pp.neighbors(adata_desc, use_rep="DESC")

        self.metrics_desc = scib.metrics.metrics(
            self.adata,
            adata_desc,
            "batch", 
            self.label_key,
            embed="DESC",
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

        random_order = np.random.permutation(adata_desc.n_obs)
        adata_desc = adata_desc[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_desc)
            sc.pl.umap(adata_desc, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_desc, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_desc)
            sc.pl.umap(adata_desc, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}DESC_cell_type.svg")
            sc.pl.umap(adata_desc, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}DESC_batch_effect.svg")

        del adata_desc

    def bbknn(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of BBKNN (version 1.6.0: https://github.com/Teichlab/bbknn) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        import bbknn

        adata_bbknn = self.adata.copy()
        sc.pp.pca(adata_bbknn, svd_solver="arpack")
        corrected = bbknn.bbknn(adata_bbknn, batch_key="batch", copy=True)
        #print(corrected)
        #print(corrected.obsp["distances"].shape)
        #print(corrected.obsp["connectivities"].shape)
        #print(corrected.obsm["X_pca"].shape)
        #sddsfsdf
        adata_bbknn = corrected.copy()
        adata_bbknn.obsm["BBKNN"] = corrected.X

        del corrected
        sc.pp.neighbors(adata_bbknn, use_rep="BBKNN")

        self.metrics_bbknn = scib.metrics.metrics(
            self.adata,
            adata_bbknn,
            "batch", 
            self.label_key,
            embed="BBKNN",
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

        random_order = np.random.permutation(adata_bbknn.n_obs)
        adata_bbknn = adata_bbknn[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_bbknn)
            sc.pl.umap(adata_bbknn, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_bbknn, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_bbknn)
            sc.pl.umap(adata_bbknn, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}BBKNN_cell_type.svg")
            sc.pl.umap(adata_bbknn, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}BBKNN_batch_effect.svg")

        del adata_bbknn

    def tosica(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of TOSICA (https://github.com/JackieHanLab/TOSICA/tree/main) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        import TOSICA

        adata_tosica = self.adata.copy()
        TOSICA.train(adata_tosica, gmt_path='human_gobp', label_name=self.label_key,epochs=3,project='hGOBP_TOSICA')

        model_weight_path = './hGOBP_TOSICA/model-0.pth'
        new_adata = TOSICA.pre(adata_tosica, model_weight_path = model_weight_path,project='hGOBP_TOSICA', laten=True)

        adata_tosica.obsm["TOSICA"] = new_adata.copy()

        del new_adata
        sc.pp.neighbors(adata_tosica, use_rep="TOSICA")

        self.metrics_tosica = scib.metrics.metrics(
            self.adata,
            adata_tosica,
            "batch", 
            self.label_key,
            embed="TOSICA",
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

        random_order = np.random.permutation(adata_tosica.n_obs)
        adata_tosica = adata_tosica[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_tosica)
            sc.pl.umap(adata_tosica, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_tosica, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_tosica)
            sc.pl.umap(adata_tosica, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}TOSICA_cell_type.svg")
            sc.pl.umap(adata_tosica, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}TOSICA_batch_effect.svg")

        del adata_tosica

    def fastmnn(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of FastMNN (version 0.1.9.5: https://github.com/chriscainx/mnnpy) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        adata_mnn = self.adata.copy()

        data_mnn = []
        for i in adata_mnn.obs["batch"].cat.categories:
            data_mnn.append(adata_mnn[adata_mnn.obs["batch"] == i].copy())

        corrected, _, _ = sc.external.pp.mnn_correct(
            *data_mnn,
            batch_key="batch",
            batch_categories=adata_mnn.obs["batch"].cat.categories,
            index_unique=None,
        )

        adata_mnn = corrected.copy()
        adata_mnn.obsm["FastMNN"] = corrected.X
        del corrected

        sc.pp.neighbors(adata_mnn, use_rep="FastMNN")

        self.metrics_fastmnn = scib.metrics.metrics(
            self.adata,
            adata_mnn,
            "batch", 
            self.label_key,
            embed="FastMNN",
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

        random_order = np.random.permutation(adata_mnn.n_obs)
        adata_mnn = adata_mnn[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_mnn)
            sc.pl.umap(adata_mnn, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_mnn, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_mnn)
            sc.pl.umap(adata_mnn, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}FastMNN_cell_type.svg")
            sc.pl.umap(adata_mnn, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}FastMNN_batch_effect.svg")

        del adata_mnn

    def trvae(self, umap_plot: bool=True, save_figure: bool=False):
        """
        trVAE: https://github.com/theislab/trvaep 
        """
        # Note: Too memory expensive to run
        '''import trvaep

        adata_trvae = adata_hvg.copy()
        n_batches = adata_trvae.obs["batch"].nunique()

        model = trvaep.CVAE(
            adata_trvae.n_vars,
            num_classes=n_batches,
            encoder_layer_sizes=[64, 32],
            decoder_layer_sizes=[32, 64],
            latent_dim=10,
            alpha=0.0001,
            use_mmd=True,
            beta=1,
            output_activation="ReLU",
        )

        # Note: set seed for reproducibility of results
        trainer = trvaep.Trainer(model, adata_trvae, condition_key="batch", seed=42)

        trainer.train_trvae(300, 1024, early_patience=50) 

        # Get the dominant batch covariate
        main_batch = adata_trvae.obs["batch"].value_counts().idxmax()

        # Get latent representation
        latent_y = model.get_y(
            adata_trvae.X,
            c=model.label_encoder.transform(np.tile(np.array([main_batch]), len(adata_trvae))),
        )
        adata_trvae.obsm["trvaep"] = latent_y

        # Get reconstructed feature space:
        #data = model.predict(x=adata_trvae.X, y=adata_trvae.obs["batch"].tolist(), target=main_batch)
        #adata_trvae.X = data'''

    def saucie(self, umap_plot: bool=True, save_figure: bool=False):
        """
        SAUCIE: https://github.com/KrishnaswamyLab/SAUCIE 
        """
        # Note: Pause for now. It requires a old version of tensorflow
        '''import SAUCIE
        import sklearn.decomposition

        adata_saucie = self.adata.copy()

        pca_op = sklearn.decomposition.PCA(100)
        data = pca_op.fit_transform(adata_saucie.X)
        saucie = SAUCIE.SAUCIE(100, lambda_b=0.1)
        loader_train = SAUCIE.Loader(data, labels=adata_saucie.obs["batch"].cat.codes, shuffle=True)
        loader_eval = SAUCIE.Loader(data, labels=adata_saucie.obs["batch"].cat.codes, shuffle=False)
        saucie.train(loader_train, steps=5000)
        adata_saucie.obsm["SAUCIE"] = saucie.get_reconstruction(loader_eval)[0]
        #adata_saucie.X = pca_op.inverse_transform(adata_saucie.obsm["SAUCIE"])

        sc.pp.neighbors(adata_saucie, use_rep="SAUCIE")'''

    def in_house_model_encoder(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
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
        adata_in_house = self.adata.copy()

        #Model
        model = model_encoder.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                              output_dim=100,
                                              drop_out=0.2,
                                              act_layer=nn.ReLU,
                                              norm_layer=nn.BatchNorm1d)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=None,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=1000,
                                        use_HVG_buckets=False,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=False,
                                        gene2vec_path=self.gene2vec_path)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_encoder = scib.metrics.metrics(
            self.adata,
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

    def in_house_model_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_pathway.py model on single-cell RNA-seq data.

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

        adata_in_house = self.adata.copy()

        #Model
        model = model_pathway.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                                attn_embed_dim=24*4,
                                                output_dim=100,
                                                num_pathways=300,
                                                num_heads=4,
                                                mlp_ratio=4,
                                                attn_bias=False,
                                                drop_ratio=0.2,
                                                attn_drop_out=0.,
                                                depth=3,
                                                norm_layer=nn.BatchNorm1d,
                                                pathway_embedding_dim=50)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=self.pathway_path,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=1000,
                                        use_HVG_buckets=False,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=False,
                                        gene2vec_path=self.gene2vec_path)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_pathways = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_Pathways_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_Pathways_batch_effect.svg")

        del adata_in_house


    def in_house_model_encoder_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_encoder_with_pathway.py model on single-cell RNA-seq data.

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

        adata_in_house = self.adata.copy()

        #Model
        model = model_encoder_with_pathway.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                                attn_embed_dim=24*4,
                                                output_dim=100,
                                                num_pathways=300,
                                                num_heads=4,
                                                mlp_ratio=4,
                                                attn_bias=False,
                                                drop_ratio=0.2,
                                                attn_drop_out=0.,
                                                proj_drop_out=0.2,
                                                depth=3,
                                                act_layer=nn.ReLU,
                                                norm_layer=nn.BatchNorm1d,
                                                pathway_embedding_dim=50)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=self.pathway_path,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=1000,
                                        use_HVG_buckets=False,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=False,
                                        gene2vec_path=self.gene2vec_path)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_encoder_pathways = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_HVG_Encoder_with_Pathways_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_HVG_Encoder_with_Pathways_batch_effect.svg")

        del adata_in_house

    def in_house_model_transformer_encoder(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the CustomScaler_model_transformer_encoder.py model on single-cell RNA-seq data.

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
        
        adata_in_house = self.adata.copy()

        #Model
        model = model_transformer_encoder.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                                            attn_embed_dim=12,
                                                            output_dim=100,
                                                            num_heads=1,
                                                            mlp_ratio=4,
                                                            drop_ratio=0.2,
                                                            attn_drop_out=0.0,
                                                            proj_drop_out=0.2,
                                                            depth=1,
                                                            act_layer=nn.ReLU,
                                                            norm_layer=nn.BatchNorm1d)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=None,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=1000,
                                        use_HVG_buckets=False,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=False,
                                        gene2vec_path=self.gene2vec_path)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=24,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_transformer_encoder = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_HVG_Transformer_Encoder_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_HVG_Transformer_Encoder_batch_effect.svg")

        del adata_in_house

    def in_house_model_transformer_encoder_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the CustomScaler_model_transformer_encoder_with_pathways.py model on single-cell RNA-seq data.

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
        
        adata_in_house = self.adata.copy()

        #Model
        model = model_transformer_encoder_with_pathways.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                                                            attn_embed_dim=12,
                                                                            num_pathways=300,
                                                                            pathway_embedding_dim=50,
                                                                            output_dim=100,
                                                                            num_heads=1,
                                                                            mlp_ratio=4,
                                                                            drop_ratio=0.2,
                                                                            attn_drop_out=0.0,
                                                                            proj_drop_out=0.2,
                                                                            depth=1,
                                                                            act_layer=nn.ReLU,
                                                                            norm_layer=nn.BatchNorm1d)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=self.pathway_path,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=1000,
                                        use_HVG_buckets=False,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=False,
                                        gene2vec_path=self.gene2vec_path)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=24,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_transformer_encoder_pathways = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_HVG_Transformer_Encoder_with_Pathways_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_HVG_Transformer_Encoder_with_Pathways_batch_effect.svg")

        del adata_in_house
    

    def in_house_model_tokenized_HVG_transformer(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of the model_tokenized_hvg_transformer.py model on single-cell RNA-seq data.

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

        adata_in_house = self.adata.copy()

        HVG_buckets_ = 300

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=None,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=True,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=True,
                                        gene2vec_path=self.gene2vec_path)
        
        #Model
        model = model_tokenized_hvg_transformer.CellType2VecModel(input_dim=min([4000,int(train_env.data_env.X.shape[1])]),
                                                        output_dim=100,
                                                        drop_out=0.2,
                                                        act_layer=nn.ReLU,
                                                        norm_layer=nn.BatchNorm1d,
                                                        attn_embed_dim=24*4,
                                                        num_heads=4,
                                                        mlp_ratio=4,
                                                        attn_bias=False,
                                                        attn_drop_out=0.,
                                                        depth=3,
                                                        nn_tokens=HVG_buckets_,
                                                        nn_embedding_dim=train_env.data_env.gene2vec_tensor.shape[1],
                                                        use_gene2vec_emb=True)
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_tokenized_HVG_transformer = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_Tokenized_HVG_Transformer_Encoder_Model_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_Tokenized_HVG_Transformer_Encoder_Model_batch_effect.svg")

        del adata_in_house

    def in_house_model_tokenized_HVG_transformer_with_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
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

        adata_in_house = self.adata.copy()

        HVG_buckets_ = 300

        train_env = trainer.train_module(data_path=adata_in_house,
                                        pathways_file_path=self.pathway_path,
                                        num_pathways=300,
                                        pathway_hvg_limit=10,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=True,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=True,
                                        gene2vec_path=self.gene2vec_path)
        
        #Model
        model = model_tokenized_hvg_transformer_with_pathways.CellType2VecModel(input_dim=4000,
                                                                    output_dim=100,
                                                                    num_pathways=300,
                                                                    drop_out=0.2,
                                                                    act_layer=nn.ReLU,
                                                                    norm_layer=nn.BatchNorm1d,
                                                                    attn_embed_dim=24*4,
                                                                    num_heads=4,
                                                                    mlp_ratio=4,
                                                                    attn_bias=False,
                                                                    attn_drop_out=0.,
                                                                    depth=3,
                                                                    pathway_embedding_dim=50,
                                                                    nn_tokens=HVG_buckets_,
                                                                    nn_embedding_dim=train_env.data_env.gene2vec_tensor.shape[1],
                                                                    use_gene2vec_emb=True)
                    
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                use_target_weights=True,
                                use_batch_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, model_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house_model_tokenized_HVG_transformer_with_pathways = scib.metrics.metrics(
            self.adata,
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
            sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}InHouse_Tokenized_HVG_Transformer_Encoder_with_Pathways_Model_cell_type.svg")
            sc.pl.umap(adata_in_house, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}InHouse_Tokenized_HVG_Transformer_Encoder_with_Pathways_Model_batch_effect.svg")

        del adata_in_house

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
        if self.metrics_pca is not None:
            calculated_metrics.append(self.metrics_pca)
            calculated_metrics_names.append("PCA")
        if self.metrics_scanorama is not None:
            calculated_metrics.append(self.metrics_scanorama)
            calculated_metrics_names.append("Scanorama")
        if self.metrics_scvi is not None:
            calculated_metrics.append(self.metrics_scvi)
            calculated_metrics_names.append("scVI")
        if self.metrics_scanvi is not None:
            calculated_metrics.append(self.metrics_scanvi)
            calculated_metrics_names.append("scANVI")
        if self.metrics_harmony is not None:
            calculated_metrics.append(self.metrics_harmony)
            calculated_metrics_names.append("Harmony")
        if self.metrics_scgen is not None:
            calculated_metrics.append(self.metrics_scgen)
            calculated_metrics_names.append("scGen")
        if self.metrics_combat is not None:
            calculated_metrics.append(self.metrics_combat)
            calculated_metrics_names.append("ComBat")
        if self.metrics_desc is not None:
            calculated_metrics.append(self.metrics_desc)
            calculated_metrics_names.append("DESC")
        if self.metrics_bbknn is not None:
            calculated_metrics.append(self.metrics_bbknn)
            calculated_metrics_names.append("BBKNN")
        if self.metrics_fastmnn is not None:
            calculated_metrics.append(self.metrics_fastmnn)
            calculated_metrics_names.append("FastMNN")
        if self.metrics_unscaled is not None:
            calculated_metrics.append(self.metrics_unscaled)
            calculated_metrics_names.append("Unintegrated")
        if self.metrics_in_house_model_encoder is not None:
            calculated_metrics.append(self.metrics_in_house_model_encoder)
            calculated_metrics_names.append("In-house HVG Encoder Model")
        if self.metrics_in_house_model_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_pathways)
            calculated_metrics_names.append("In-house Pathways Model")
        if self.metrics_in_house_model_encoder_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_encoder_pathways)
            calculated_metrics_names.append("In-house HVG Encoder with Pathways Model")
        if self.metrics_in_house_model_transformer_encoder is not None:
            calculated_metrics.append(self.metrics_in_house_model_transformer_encoder)
            calculated_metrics_names.append("In-house HVG Transformer Encoder Model")
        if self.metrics_in_house_model_transformer_encoder_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_transformer_encoder_pathways)
            calculated_metrics_names.append("In-house HVG Transformer Encoder with Pathways Model")
        if self.metrics_in_house_model_tokenized_HVG_transformer is not None:
            calculated_metrics.append(self.metrics_in_house_model_tokenized_HVG_transformer)
            calculated_metrics_names.append("In-house Tokenized HVG Transformer Encoder Model")
        if self.metrics_in_house_model_tokenized_HVG_transformer_with_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_tokenized_HVG_transformer_with_pathways)
            calculated_metrics_names.append("In-house Tokenized HVG Transformer Encoder with Tokenized Pathways Model")

        if len(calculated_metrics_names) != 0:
            metrics = pd.concat(calculated_metrics, axis="columns")

            metrics = metrics.set_axis(calculated_metrics_names, axis="columns")

            metrics = metrics.loc[
                [
                    "ASW_label",
                    "ASW_label/batch",
                    "PCR_batch",
                    "isolated_label_silhouette",
                    "graph_conn",
                    "hvg_overlap",
                    "NMI_cluster/label",
                    "ARI_cluster/label",
                    "cell_cycle_conservation",
                    "isolated_label_F1"
                ],
                :,
            ]

            metrics = metrics.T
            metrics = metrics.drop(columns=["hvg_overlap"])

            if self.metrics is None:
                self.metrics = metrics#.sort_values(by='Overall', ascending=False)
            else:
                self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()

        if min_max_normalize:
            self.metrics = (self.metrics - self.metrics.min()) / (self.metrics.max() - self.metrics.min())
        
        self.metrics["Overall Batch"] = self.metrics[["ASW_label/batch", "PCR_batch", "graph_conn"]].mean(axis=1)
        self.metrics["Overall Bio"] = self.metrics[["ASW_label", 
                                        "isolated_label_silhouette", 
                                        "NMI_cluster/label", 
                                        "ARI_cluster/label",
                                        "isolated_label_F1",
                                        "cell_cycle_conservation"]].mean(axis=1)
        self.metrics["Overall"] = 0.4 * self.metrics["Overall Batch"] + 0.6 * self.metrics["Overall Bio"] # priorities biology slightly more
        self.metrics = self.metrics.sort_values(by='Overall', ascending=False)

        self.metrics = self.metrics.sort_values(by='Overall', ascending=False)

    def visualize_results(self, bg_color: str="Blues"):
        """
        Visualizes the performance metrics dataframe using a colored heatmap.

        Parameters
        ----------
        bg_color : str, optional
            The colormap for the heatmap (default is "Blues").

        Returns
        -------
        None

        Notes
        -----
        This method creates a styled heatmap of the performance metrics dataframe for visual inspection.
        """
        styled_metrics = self.metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

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
        self.metrics = pd.read_csv(f'{name}.csv', index_col=0)




