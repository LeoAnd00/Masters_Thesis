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
from models import model_transformer_encoder as model_transformer_encoder
from models import model_transformer_encoder_with_pathways as model_transformer_encoder_with_pathways

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class benchmark():
    """
    A class for benchmarking single-cell RNA-seq data integration methods.

    Parameters:
    data_path (str): The path to the single-cell RNA-seq data file in h5ad format.
    batch_key (str): The batch key to use for batch information. Default is "patientID".
    HVG (bool, optional): Whether to select highly variable genes (HVGs). Default is True.
    HVGs (int, optional): The number of highly variable genes to select if HVG is enabled. Default is 4000.
    Scaled (bool, optional): Whether to scale the data. Default is False.
    """

    def __init__(self, 
                 data_path: str, 
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

        # Initialize variables
        self.metrics = None
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

    def unintegrated(self, n_comps: int=30, umap_plot: bool=True):
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

        if umap_plot:
            sc.tl.umap(adata_unscaled)
            sc.pl.umap(adata_unscaled, color=[self.label_key, "batch"], ncols=1)

        del adata_unscaled

    def scanorama(self, umap_plot: bool=True):
        """
        SCANORAMA version 1.7.4: https://github.com/brianhie/scanorama
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

        if umap_plot:
            sc.tl.umap(adata_scanorama)
            sc.pl.umap(adata_scanorama, color=[self.label_key, "batch"], ncols=1)

        del adata_scanorama

    def harmony(self, n_comps: int=30, umap_plot: bool=True):
        """
        Harmony version 0.1.7: https://github.com/lilab-bcb/harmony-pytorch
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

        if umap_plot:
            sc.tl.umap(adata_harmony)
            sc.pl.umap(adata_harmony, color=[self.label_key, "batch"], ncols=1)

        del adata_harmony

    def scvi(self, umap_plot: bool=True):
        """
        scVI version 1.0.4: https://github.com/scverse/scvi-tools
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

        if umap_plot:
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color=[self.label_key, "batch"], ncols=1)

        del adata_scvi

        return vae

    def scanvi(self, umap_plot: bool=True, vae=None):
        """
        scANVI version 1.0.4: https://github.com/scverse/scvi-tools
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

        if umap_plot:
            sc.tl.umap(adata_scANVI)
            sc.pl.umap(adata_scANVI, color=[self.label_key, "batch"], ncols=1)

        del adata_scANVI

    def scgen(self, umap_plot: bool=True):
        """
        scGen version 2.1.1: https://github.com/theislab/scgen 
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

        if umap_plot:
            sc.tl.umap(adata_scgen)
            sc.pl.umap(adata_scgen, color=[self.label_key, "batch"], ncols=1)

        del adata_scgen

    def combat(self, umap_plot: bool=True):
        """
        ComBat (Scanpy version 1.9.5): https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.combat.html
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

        if umap_plot:
            sc.tl.umap(adata_combat)
            sc.pl.umap(adata_combat, color=[self.label_key, "batch"], ncols=1)

        del adata_combat

    def desc(self, umap_plot: bool=True):
        """
        DESC version 2.1.1: https://github.com/eleozzr/desc
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

        if umap_plot:
            sc.tl.umap(adata_desc)
            sc.pl.umap(adata_desc, color=[self.label_key, "batch"], ncols=1)

        del adata_desc

    def bbknn(self, umap_plot: bool=True):
        """
        BBKNN version 1.6.0: https://github.com/Teichlab/bbknn
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

        if umap_plot:
            sc.tl.umap(adata_bbknn)
            sc.pl.umap(adata_bbknn, color=[self.label_key, "batch"], ncols=1)

        del adata_bbknn

    def tosica(self, umap_plot: bool=True):
        """
        TOSICA: https://github.com/JackieHanLab/TOSICA/tree/main
        """
        import TOSICA

        adata_tosica = self.adata.copy()
        TOSICA.train(adata_tosica, gmt_path='human_gobp', label_name=self.label_key,epochs=3,project='hGOBP_TOSICA')

        model_weight_path = './hGOBP_TOSICA/model-0.pth'
        new_adata = TOSICA.pre(adata_tosica, model_weight_path = model_weight_path,project='hGOBP_TOSICA', laten=True)

        adata_tosica.obsm["TOSICA"] = new_adata

        print(adata_tosica.obsm["TOSICA"].shape)
        return

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

        if umap_plot:
            sc.tl.umap(adata_tosica)
            sc.pl.umap(adata_tosica, color=[self.label_key, "batch"], ncols=1)

        del adata_tosica

    def fastmnn(self, umap_plot: bool=True):
        """
        FastMNN version 0.1.9.5: https://github.com/chriscainx/mnnpy
        """
        # Doesn't work great for some reason
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

        if umap_plot:
            sc.tl.umap(adata_mnn)
            sc.pl.umap(adata_mnn, color=[self.label_key, "batch"], ncols=1)

        del adata_mnn

    def trvae(self, umap_plot: bool=True):
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

    def saucie(self, umap_plot: bool=True):
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

    def in_house_model_encoder(self, save_path: str, umap_plot: bool=True, train: bool=True):
        """
        Model description

        Parameters:
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool
            Whether to train the model (True) or use a existing model (False) (default: True).
        """
        adata_in_house = self.adata.copy()

        #Model
        model = model_encoder.CellType2VecModel(input_dim=adata_in_house.X.shape[1],
                                              output_dim=100,
                                              drop_out=0.2,
                                              act_layer=nn.ReLU,
                                              norm_layer=nn.BatchNorm1d)

        train_env = trainer.train_module(data_path=adata_in_house,
                                        json_file_path=None,
                                        num_pathways=None,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
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

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=[self.label_key, "batch"], ncols=1)

        del adata_in_house

    def in_house_model_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True):
        """
        Model description

        Parameters:
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool
            Whether to train the model (True) or use a existing model (False) (default: True).
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
                                        json_file_path='../../data/processed/pathway_information/all_pathways.json',
                                        num_pathways=300,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
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

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=[self.label_key, "batch"], ncols=1)

        del adata_in_house

    def in_house_model_encoder_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True):
        """
        Model description

        Parameters:
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool
            Whether to train the model (True) or use a existing model (False) (default: True).
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
                                        json_file_path='../../data/processed/pathway_information/all_pathways.json',
                                        num_pathways=300,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=256,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
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

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=[self.label_key, "batch"], ncols=1)

        del adata_in_house

    def in_house_model_transformer_encoder(self, save_path: str, umap_plot: bool=True, train: bool=True):
        """
        Model description

        Parameters:
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool
            Whether to train the model (True) or use a existing model (False) (default: True).
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
                                        json_file_path=None,
                                        num_pathways=None,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=24,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
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

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=[self.label_key, "batch"], ncols=1)

        del adata_in_house

    def in_house_model_transformer_encoder_pathways(self, save_path: str, umap_plot: bool=True, train: bool=True):
        """
        Model description

        Parameters:
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool
            Whether to train the model (True) or use a existing model (False) (default: True).
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
                                        json_file_path='../../data/processed/pathway_information/all_pathways.json',
                                        num_pathways=300,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        if train:
            _ = train_env.train(model=model,
                                device=None,
                                seed=42,
                                batch_size=24,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                eval_freq=4,
                                epochs=20,
                                earlystopping_threshold=3)
        
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
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

        if umap_plot:
            sc.tl.umap(adata_in_house)
            sc.pl.umap(adata_in_house, color=[self.label_key, "batch"], ncols=1)

        del adata_in_house

    def make_benchamrk_results_dataframe(self):
        """
        Makes a dataframe called metrics that contains the performance of the different methods for multiple metrics
        """

        calculated_metrics = []
        calculated_metrics_names = []
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
            calculated_metrics_names.append("In-house Encoder Model")
        if self.metrics_in_house_model_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_pathways)
            calculated_metrics_names.append("In-house Pathways Model")
        if self.metrics_in_house_model_encoder_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_encoder_pathways)
            calculated_metrics_names.append("In-house Encoder with Pathways Model")
        if self.metrics_in_house_model_transformer_encoder is not None:
            calculated_metrics.append(self.metrics_in_house_model_transformer_encoder)
            calculated_metrics_names.append("In-house Transformer Encoder Model")
        if self.metrics_in_house_model_transformer_encoder_pathways is not None:
            calculated_metrics.append(self.metrics_in_house_model_transformer_encoder_pathways)
            calculated_metrics_names.append("In-house Transformer Encoder with Pathways Model")

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

        metrics["Overall Batch"] = metrics[["ASW_label/batch", "PCR_batch", "graph_conn"]].mean(axis=1)
        metrics["Overall Bio"] = metrics[["ASW_label", 
                                        "isolated_label_silhouette", 
                                        "NMI_cluster/label", 
                                        "ARI_cluster/label",
                                        "isolated_label_F1",
                                        "cell_cycle_conservation"]].mean(axis=1)
        metrics["Overall"] = 0.4 * metrics["Overall Batch"] + 0.6 * metrics["Overall Bio"] # priorities biology slightly more

        if self.metrics is None:
            self.metrics = metrics.sort_values(by='Overall', ascending=False)
        else:
            self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()
            self.metrics = self.metrics.sort_values(by='Overall', ascending=False)

    def visualize_results(self, bg_color: str="Blues"):
        styled_metrics = self.metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

    def save_results_as_csv(self, name: str='benchmarks/results/Benchmark_results'):
        self.metrics.to_csv(f'{name}.csv', index=True, header=True)

    def read_csv(self, name: str='benchmarks/results/Benchmark_results'):
        if self.metrics is None:
            self.metrics = pd.read_csv(f'{name}.csv', index_col=0)
        else:
            metrics = pd.read_csv(f'{name}.csv', index_col=0)
            self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()
            self.metrics = self.metrics.sort_values(by='Overall', ascending=False)




