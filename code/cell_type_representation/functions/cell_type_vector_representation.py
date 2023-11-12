import torch
import torch.nn as nn
import numpy as np
from functions import train as trainer
import scanpy as sc
from scipy.spatial.distance import cdist
import json
import pandas as pd


class MakeCellTypeRepresentation():

    def __init__(self, 
                 data_path, 
                 pathways_path: str, 
                 model_path: str='trained_models/Encoder/', 
                 num_pathways: int=300,
                 HVG: bool=True,
                 HVGs: int=4000,
                 Scaled: bool=False,
                 target_key: str="cell_type", 
                 batch_keys: list=["patientID"]):
        
        self.target_key = target_key

        # Load data and training environment containing the predict function
        train_env = trainer.train_module(data_path=data_path,
                                        json_file_path=pathways_path,
                                        num_pathways=num_pathways,
                                        save_model_path=model_path,
                                        HVG=HVG,
                                        HVGs=HVGs,
                                        Scaled=Scaled,
                                        target_key=target_key,
                                        batch_keys=batch_keys)
        
        #self.adata = sc.read(data_path, cache=True)
        self.adata = train_env.data_env.adata

        # Make predictions
        predictions = train_env.predict(data_=self.adata, out_path=model_path)
        self.adata.obsm["predictions"] = predictions
        #sc.tl.pca(self.adata, n_comps=50, use_highly_variable=True)
        #self.adata.obsm["predictions"] = self.adata.obsm["X_pca"]

        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)

        sc.pp.neighbors(self.adata, use_rep="predictions")
        sc.tl.umap(self.adata)
        sc.pl.umap(self.adata, color=self.target_key, ncols=1, edgecolor="none", show=False, title='Cell type embedding space')

    def CentroidRepresentation(self):
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
        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame(representation)

        # Save the DataFrame as a CSV file
        df.to_csv(save_path, index=False)

