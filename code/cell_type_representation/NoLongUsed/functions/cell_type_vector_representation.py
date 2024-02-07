import torch
import torch.nn as nn
import numpy as np
from functions import train2 as trainer
import scanpy as sc
from scipy.spatial.distance import cdist
import json
import pandas as pd


class MakeCellTypeRepresentation():
    """
    A class for creating and visualizing cell type representations in a latent space obtained from a trained model.

    Parameters
    ----------
    data_path : str
        Path to the dataset. Needs to be the same as used to train the model.
    pathways_path : str
        Path to the JSON file containing pathway information. Needs to be the same as used to train the model.
    model_path : str, optional
        Path to the trained model.
    num_pathways : int, optional
        Number of pathways. Default is 300. Needs to be the same as used to train the model.
    pathway_hvg_limit : int, optional
        The minimum number of HVGs in a pathway for the pathway to be considered (default is 10). Only used if json_file_path is given. Needs to be the same as used to train the model.
    pathways_buckets : int, optional
        The number of buckets for binning pathway information (default is 100). Only used if json_file_path is given and use_pathway_buckets is set to True. Needs to be the same as used to train the model.
    use_pathway_buckets : bool, optional
        Whether to use pathway information representations as relativ percentage of active genes of a pathway and to use buckets (True). Or To simply apply the binary pathway mask on the expression levels (False) (default is False). Needs to be the same as used to train the model.
    HVG : bool, optional
        Whether to use highly variable genes. Default is True. Needs to be the same as used to train the model.
    HVGs : int, optional
        Number of highly variable genes. Default is 4000. Needs to be the same as used to train the model.
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True. Needs to be the same as used to train the model.
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization. Needs to be the same as used to train the model.
    Scaled : bool, optional
        Whether to scale the data. Default is False. Needs to be the same as used to train the model.
    target_key : str, optional
        Key specifying the target variable (e.g., cell type). Default is 'cell_type'. Needs to be the same as used to train the model.
    batch_keys : list, optional
        List of keys specifying batch variables. Default is ['patientID']. Needs to be the same as used to train the model.
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec representations when training the HVG Transformer model relying on tokenization. use_HVG_buckets must be set to True for the use of gene2vec to work.

    Methods
    -------
    CentroidRepresentation() -> dict:
        Calculates and returns centroids for each cell type cluster in the latent space.

    MedianRepresentation() -> dict:
        Calculates and returns median centroids for each cell type cluster in the latent space.

    MedoidRepresentation() -> dict:
        Calculates and returns medoid centroids for each cell type cluster in the latent space.

    UMAP_visualization(representation: dict, neighbors: int = 5, size: int = 60):
        Generates a UMAP visualization of the cell type vector representations.

    PCA_visualization(representation: dict, size: int = 60):
        Generates a PCA plot of the cell type vector representations.

    Download_representation(representation: dict, save_path: str = "file.csv"):
        Converts the representation dictionary to a Pandas DataFrame and saves it as a CSV file.
    """

    def __init__(self, 
                 data_path, 
                 pathways_path: str, 
                 model_path: str='trained_models/Encoder/', 
                 num_pathways: int=300,
                 pathway_hvg_limit: int=10,
                 pathways_buckets: int=100,
                 use_pathway_buckets: bool=True,
                 HVG: bool=True,
                 HVGs: int=4000,
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 Scaled: bool=False,
                 target_key: str="cell_type", 
                 batch_keys: list=["patientID"],
                 use_gene2vec_emb: bool = False):
        
        self.target_key = target_key

        # Load data and training environment containing the predict function
        train_env = trainer.train_module(data_path=data_path,
                                        json_file_path=pathways_path,
                                        num_pathways=num_pathways,
                                        pathway_hvg_limit=pathway_hvg_limit,
                                        save_model_path=model_path,
                                        pathways_buckets=pathways_buckets,
                                        use_pathway_buckets=use_pathway_buckets,
                                        HVG=HVG,
                                        HVGs=HVGs,
                                        HVG_buckets=HVG_buckets,
                                        use_HVG_buckets=use_HVG_buckets,
                                        Scaled=Scaled,
                                        target_key=target_key,
                                        batch_keys=batch_keys,
                                        use_gene2vec_emb=use_gene2vec_emb)
        
        #self.adata = sc.read(data_path, cache=True)
        self.adata = train_env.data_env.adata

        # Make predictions
        predictions = train_env.predict(data_=self.adata, model_path=model_path)
        self.adata.obsm["predictions"] = predictions
        #sc.tl.pca(self.adata, n_comps=50, use_highly_variable=True)
        #self.adata.obsm["predictions"] = self.adata.obsm["X_pca"]

        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)

        sc.pp.neighbors(self.adata, use_rep="predictions")
        sc.tl.umap(self.adata)
        sc.pl.umap(self.adata, color=self.target_key, ncols=1, edgecolor="none", show=False, title='Cell type embedding space')

    def CentroidRepresentation(self):
        """
        Calculates and returns centroids for each unique label in the dataset.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the centroid representation.
        """
        unique_labels = np.unique(self.adata.obs[self.target_key])
        centroids = {}  # A dictionary to store centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(self.adata.obs[self.target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = self.adata.obsm["predictions"][label_indices]

            # Calculate the centroid for the current label cluster.
            centroid = np.mean(latent_space_for_label, axis=0)

            centroids[label] = centroid

        return centroids

    def MedianRepresentation(self):
        """
        Calculates and returns median centroids for each unique label in the dataset.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the median centroid representation.
        """
        unique_labels = np.unique(self.adata.obs[self.target_key])
        median_centroids = {}  # A dictionary to store median centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(self.adata.obs[self.target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = self.adata.obsm["predictions"][label_indices]

            # Calculate the median for each feature across the data points with the current label.
            median = np.median(latent_space_for_label, axis=0)

            median_centroids[label] = median

        return median_centroids

    def MedoidRepresentation(self):
        """
        Calculates and returns medoid centroids for each unique label in the dataset.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the medoid centroid representation.
        """
        unique_labels = np.unique(self.adata.obs[self.target_key])
        medoid_centroids = {}  # A dictionary to store medoid centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(self.adata.obs[self.target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = self.adata.obsm["predictions"][label_indices]

            # Calculate pairwise distances between data points in the group.
            distances = cdist(latent_space_for_label, latent_space_for_label, 'euclidean')

            # Find the index of the medoid (index with the smallest sum of distances).
            medoid_index = np.argmin(distances.sum(axis=0))

            # Get the medoid (data point) itself.
            medoid = latent_space_for_label[medoid_index]

            medoid_centroids[label] = medoid

        return medoid_centroids
    
    def UMAP_visualization(self, representation: dict, neighbors: int=5, size: int=60):
        """
        Generates a UMAP visualization of cell type vector representations.

        Parameters
        ----------
        representation : dict
            A dictionary where each key is a label, and the corresponding value is the vector representation.
        neighbors : int, optional
            The number of neighbors for UMAP construction. Default is 5.
        size : int, optional
            Size of data points in the plot. Default is 60.
        """
        # Make AnnData object
        latent_space_array = np.array(list(representation.values()))
        labels = np.array(list(representation.keys()))
        adata = sc.AnnData(latent_space_array)
        labels = pd.Categorical(labels, categories=labels, ordered=False)
        adata.obs['labels'] = labels

        # Use Scanpy UMAP to visualize
        sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep='X')
        sc.tl.umap(adata)

        sc.pl.umap(adata, color='labels', size=size, edgecolor="none", show=False, title='Cell type vector representations')

    def PCA_visualization(self, representation: dict, size: int=60):
        """
        Generates a PCA plot of cell type vector representations.

        Parameters
        ----------
        representation : dict
            A dictionary where each key is a label, and the corresponding value is the vector representation.
        size : int, optional
            Size of data points in the plot. Default is 60.
        """
        # Make AnnData object
        latent_space_array = np.array(list(representation.values()))
        labels = np.array(list(representation.keys()))
        adata = sc.AnnData(latent_space_array)
        labels = pd.Categorical(labels, categories=labels, ordered=False)
        adata.obs['labels'] = labels

        # Perform PCA
        sc.tl.pca(adata)

        # Visualize the PCA plot
        sc.pl.pca(adata, color='labels', size=size, edgecolor="none", show=False, title='Cell type vector representations')

    def Download_representation(self, representation: dict, save_path: str="file.csv"):
        """
        Converts a dictionary of cell type vector representations to a CSV file.

        Parameters
        ----------
        representation : dict
            A dictionary where each key is a label, and the corresponding value is the vector representation.
        save_path : str, optional
            Path to save the CSV file. Default is "file.csv".
        """
        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame(representation)

        # Save the DataFrame as a CSV file
        df.to_csv(save_path, index=False)

