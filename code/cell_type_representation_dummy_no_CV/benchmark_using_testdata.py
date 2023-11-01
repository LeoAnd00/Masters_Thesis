# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
import random
import tensorflow as tf
from IPython.display import display
from functions import data_preprocessing as dp
from models import cl_dummy_model4 as scRNASeq_model

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
    seed (int, optional): Random seed to use.
    folds (int, otpional): Folds for testing, example 5 folds would mean 4 folds will be used for training and 1 for testing.
    """

    def __init__(self, 
                 data_path: str, 
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 Scaled: bool=False,
                 seed: int=42,
                 folds: int=5):
        #data_path = '../../data/processed/immune_cells/merged/Oetjen_merged.h5ad'
        all_adata = sc.read(data_path, cache=True)

        all_adata.obs["batch"] = all_adata.obs[batch_key]

        self.all_adata = all_adata
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
        self.metrics_in_house = None

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
            sc.pp.highly_variable_genes(self.all_adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.all_adata = self.all_adata[:, self.all_adata.var["highly_variable"]].copy()
        if Scaled:
            self.all_adata.X = dp.scale_data(self.all_adata.X)

        # Split data into training and testing
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        encoder = LabelEncoder()
        encoded_batch = encoder.fit_transform(self.all_adata.obs["batch"])
        for fold_idx, (train_indices, test_indices) in enumerate(skf.split(self.all_adata.X, self.all_adata.obs[self.label_key], encoded_batch)):
            self.adata = self.all_adata[train_indices, :].copy()
            self.test_adata = self.all_adata[test_indices, :].copy()
            self.test_indices = test_indices
            break

    def unintegrated(self, n_comps: int=30, umap_plot: bool=True):
        adata_unscaled = self.test_adata.copy()

        #sc.tl.pca(adata_unscaled, n_comps=n_comps, use_highly_variable=True)
        adata_unscaled.obsm["Unscaled"] = adata_unscaled.X
        sc.pp.neighbors(adata_unscaled, use_rep="Unscaled")

        self.metrics_unscaled = scib.metrics.metrics(
            self.test_adata,
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

        # Scanorama requires batch_key to make predictions. We allow it to use the batch information for the test data

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

        adata_scanorama = adata_scanorama[self.test_indices,:].copy()

        sc.pp.neighbors(adata_scanorama, use_rep="Scanorama")

        self.metrics_scanorama = scib.metrics.metrics(
            self.test_adata,
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

        # Harmony requires batch_key to make predictions. We allow it to use the batch information for the test data

        # import package
        from harmony import harmonize

        adata_harmony = self.adata.copy()

        sc.tl.pca(adata_harmony, n_comps=n_comps, use_highly_variable=True)

        adata_harmony.obsm["Harmony"] = harmonize(adata_harmony.obsm["X_pca"], adata_harmony.obs, batch_key="batch")

        adata_harmony = adata_harmony[self.test_indices,:].copy()

        sc.pp.neighbors(adata_harmony, use_rep="Harmony")

        self.metrics_harmony = scib.metrics.metrics(
            self.test_adata,
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

        adata_scvi_train = self.adata.copy()

        scvi.model.SCVI.setup_anndata(adata_scvi_train, layer="pp_counts", batch_key="batch")
        vae = scvi.model.SCVI(adata_scvi_train, gene_likelihood="nb", n_layers=2, n_latent=30)
        vae.train()
        del adata_scvi_train
        #adata_scvi.obsm["scVI"] = vae.get_latent_representation()

        adata_scvi = self.test_adata.copy()
        adata_scvi.obsm["scVI"] = vae.get_latent_representation(adata_scvi)#vae.get_latent_representation()

        sc.pp.neighbors(adata_scvi, use_rep="scVI")

        self.metrics_scvi = scib.metrics.metrics(
            self.test_adata,
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

        adata_scANVI_train = self.adata.copy()

        if vae is None:
            scvi.model.SCVI.setup_anndata(adata_scANVI_train, layer="pp_counts", batch_key="batch")
            vae = scvi.model.SCVI(adata_scANVI_train, gene_likelihood="nb", n_layers=2, n_latent=30)
            vae.train()

        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata_scANVI_train,
            labels_key=self.label_key,
            unlabeled_category="UnknownUnknown",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        del adata_scANVI_train

        adata_scANVI = self.test_adata.copy()
        adata_scANVI.obsm["scANVI"] = lvae.get_latent_representation(adata_scANVI)

        del lvae, vae
        sc.pp.neighbors(adata_scANVI, use_rep="scANVI")

        self.metrics_scanvi = scib.metrics.metrics(
            self.test_adata,
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

        adata_scgen_train = self.adata.copy()

        SCGEN.setup_anndata(adata_scgen_train, batch_key="batch", labels_key=self.label_key)
        model = SCGEN(adata_scgen_train)
        model.train(
            max_epochs=100,
            batch_size=128,
            early_stopping=True,
            early_stopping_patience=10,
        )
        del adata_scgen_train

        adata_scgen = self.test_adata.copy()
        corrected_adata = model.batch_removal(adata_scgen)#model.batch_removal()

        adata_scgen.obsm["scGen"] = corrected_adata.obsm["corrected_latent"]

        del corrected_adata
        sc.pp.neighbors(adata_scgen, use_rep="scGen")

        self.metrics_scgen = scib.metrics.metrics(
            self.test_adata,
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

        # ComBat requires batch_key to make predictions. We allow it to use the batch information for the test data

        adata_combat = self.adata.copy()
        corrected_data = sc.pp.combat(adata_combat, key="batch", inplace=False)

        adata_combat.obsm["ComBat"] = corrected_data

        adata_combat = adata_combat[self.test_indices,:].copy()

        del corrected_data
        sc.pp.neighbors(adata_combat, use_rep="ComBat")

        self.metrics_combat = scib.metrics.metrics(
            self.test_adata,
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

        # DESC requires batch_key to make predictions. We allow it to use the batch information for the test data

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
        adata_desc = adata_desc[self.test_indices,:].copy()

        del adata_out
        sc.pp.neighbors(adata_desc, use_rep="DESC")

        self.metrics_desc = scib.metrics.metrics(
            self.test_adata,
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

        # BBKNN requires batch_key to make predictions. We allow it to use the batch information for the test data

        import bbknn

        adata_bbknn = self.adata.copy()
        sc.pp.pca(adata_bbknn, svd_solver="arpack")
        corrected = bbknn.bbknn(adata_bbknn, batch_key="batch", copy=True)
        adata_bbknn.obsm["BBKNN"] = corrected.X

        adata_bbknn = adata_bbknn[self.test_indices,:].copy()

        del corrected
        sc.pp.neighbors(adata_bbknn, use_rep="BBKNN")

        self.metrics_bbknn = scib.metrics.metrics(
            self.test_adata,
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

        print(new_adata)

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
        # FastMNN requires batch_key to make predictions. We allow it to use the batch information for the test data

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

        adata_mnn.obsm["FastMNN"] = corrected.X
        del corrected

        sc.pp.neighbors(adata_mnn, use_rep="FastMNN")

        self.metrics_fastmnn = scib.metrics.metrics(
            self.test_adata,
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

    def in_house_model(self, save_path: str, umap_plot: bool=True):
        adata_in_house_train = self.adata.copy()
        
        # For cl_dummy_model2
        train_env = scRNASeq_model.train_module(data_path=adata_in_house_train,
                                        save_model_path=save_path,
                                        HVG=False,
                                        HVGs=4000,
                                        Scaled=False,
                                        target_key=self.label_key,
                                        batch_keys=["batch"])
        
        # Train
        '''_, _ = train_env.train(device=None,
                                seed=42,
                                folds=5,
                                cv=False,
                                batch_size=256,
                                attn_embed_dim=24,
                                depth=4,
                                num_heads=1,
                                output_dim=100,
                                attn_drop_out=0.,
                                proj_drop_out=0.2,
                                attn_bias=False,
                                act_layer=nn.Tanh,
                                norm_layer=nn.BatchNorm1d,#nn.BatchNorm1d,#nn.LayerNorm,
                                loss_with_weights=True,
                                init_temperature=0.25,
                                min_temperature=0.1,
                                max_temperature=2.0,
                                init_lr=0.001,
                                lr_scheduler_warmup=4,
                                lr_scheduler_maxiters=25,
                                print_rate=2,
                                epochs=20,
                                earlystopping_threshold=5)'''
        _ = train_env.train(device=None,
                                seed=42,
                                batch_size=256,
                                attn_embed_dim=24,
                                depth=4,
                                num_heads=1,
                                output_dim=100,
                                attn_drop_out=0.,
                                proj_drop_out=0.2,
                                attn_bias=False,
                                act_layer=nn.ReLU,
                                norm_layer=nn.BatchNorm1d,#nn.BatchNorm1d,#nn.LayerNorm,
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
        
        adata_in_house = self.test_adata.copy()
        predictions = train_env.predict(data_=adata_in_house, out_path=save_path)
        adata_in_house.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics_in_house = scib.metrics.metrics(
            self.test_adata,
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
        if self.metrics_in_house is not None:
            calculated_metrics.append(self.metrics_in_house)
            calculated_metrics_names.append("In-house")

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

    def save_results_as_csv(self, name: str='Benchmark_results'):
        self.metrics.to_csv(f'{name}.csv', index=True, header=True)

    def read_csv(self, name: str='Benchmark_results'):
        if self.metrics is None:
            self.metrics = pd.read_csv(f'{name}.csv', index_col=0)
        else:
            metrics = pd.read_csv(f'{name}.csv', index_col=0)
            self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()
            self.metrics = self.metrics.sort_values(by='Overall', ascending=False)


