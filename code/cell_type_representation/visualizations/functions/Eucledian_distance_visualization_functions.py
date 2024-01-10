
import scanpy as sc
from functions import train as trainer
from functions import data_preprocessing as dp
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import scib
from sklearn.decomposition import PCA
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class VisualizeEnv():
    """
    A class for visualizing eucledian distance between cell type clusters when accounting for batch effect.

    Methods:
    - MakePredictions: Make predictions using a trained model on a separate dataset.
    - PCA_cell_type_centroid_distances: Calculate the centroid distances between cell types in PCA space.
    - CalculateDistanceMatrix: Calculate the distance matrix based on PCA space.
    - DownloadDistanceMatrix: Save the distance matrix to a file.
    - LoadDistanceMatrix: Load a precomputed distance matrix.
    - VisualizeCellTypeCorrelations: Visualize the distance matrices and their statistics.
    """

    def __init__(self):
        pass

    def MakePredictions(self, 
                        predict: bool,
                        train_path: str, 
                        pred_path: str, 
                        gene2vec_path: str,
                        model_path: str,
                        target_key: str,
                        batch_key: str,
                        model: nn.Module,
                        HVGs: int=2000, 
                        HVG_buckets_: int=1000,
                        Use_HVG_buckets_: bool=False,
                        use_gene2vec_emb: bool=False,
                        pathways_file_path: str=None,
                        batch_size: int=32):
        """
        Make predictions using a trained model on a separate dataset.

        Parameters:
        - predict (bool): Whether to perform predictions.
        - train_path (str): File path to the training dataset in AnnData format.
        - pred_path (str): File path to the prediction dataset in AnnData format.
        - gene2vec_path (str): File path to gene2vec embeddings.
        - model_path (str): Directory path to save the trained model and predictions.
        - target_key (str): Key for cell type labels.
        - batch_key (str): Key for batch information.
        - model (nn.Module): Trained neural network model.
        - HVGs (int): Number of highly variable genes (HVGs).
        - HVG_buckets_ (int): Number of HVG buckets.
        - Use_HVG_buckets_ (bool): Whether to use HVG buckets.
        - use_gene2vec_emb (bool): Whether to use gene2vec embeddings.
        - pathways_file_path (str): File path to pathway information.
        - batch_size (int): Batch size for predictions.

        Returns:
        None
        """
        
        self.label_key = target_key
        self.train_adata = sc.read(train_path, cache=True)
        self.train_adata.obs["batch"] = self.train_adata.obs[batch_key]

        self.train_env = trainer.train_module(data_path=self.train_adata,
                                        pathways_file_path=pathways_file_path,
                                        num_pathways=300,
                                        pathway_gene_limit=10,
                                        save_model_path="",
                                        HVG=True,
                                        HVGs=HVGs,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=Use_HVG_buckets_,
                                        Scaled=False,
                                        target_key=target_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=use_gene2vec_emb,
                                        gene2vec_path=gene2vec_path)
        
        self.data_env = self.train_env.data_env
        
        self.pred_adata = sc.read(pred_path, cache=True)
        self.pred_adata.obs["batch"] = self.pred_adata.obs[batch_key]

        if predict:
                
            self.prediction = self.train_env.predict(data_=self.pred_adata, model=model, model_path=model_path, batch_size=batch_size)

            self.pred_adata.obsm["In_house"] = self.prediction

    def PCA_cell_type_centroid_distances(self, n_components: int=100):
        """
        Calculate the average centroid distances between cell types across batch effects in PCA space.

        Parameters:
        - n_components (int): Number of principal components for PCA.

        Returns:
        - average_distance_df (pd.DataFrame): DataFrame of average centroid distances.
        - distance_std_df (pd.DataFrame): DataFrame of standard deviations of centroid distances.
        """

        # Step 1: Perform PCA on AnnData.X
        adata = self.data_env.adata.copy()  # Make a copy of the original AnnData object
        pca = PCA(n_components=n_components)
        adata_pca = pca.fit_transform(adata.X)

        # Step 2: Calculate centroids for each cell type cluster of each batch effect
        centroids = {}
        for batch_effect in adata.obs['batch'].unique():
            for cell_type in adata.obs['cell_type'].unique():
                mask = (adata.obs['batch'] == batch_effect) & (adata.obs['cell_type'] == cell_type)
                centroid = np.mean(adata_pca[mask], axis=0)
                centroids[(batch_effect, cell_type)] = centroid

        # Step 3: Calculate the average centroid distance between all batch effects
        average_distance_matrix = np.zeros((len(adata.obs['cell_type'].unique()), len(adata.obs['cell_type'].unique())))
        distance_std_matrix = np.zeros((len(adata.obs['cell_type'].unique()), len(adata.obs['cell_type'].unique())))
        for i, cell_type_i in enumerate(adata.obs['cell_type'].unique()):
            for j, cell_type_j in enumerate(adata.obs['cell_type'].unique()):
                distances = []
                for batch_effect in adata.obs['batch'].unique():
                    centroid_i = torch.tensor(centroids[(batch_effect, cell_type_i)], dtype=torch.float32, requires_grad=False)
                    centroid_j = torch.tensor(centroids[(batch_effect, cell_type_j)], dtype=torch.float32, requires_grad=False)
                    try:
                        #distance = euclidean(centroids[(batch_effect, cell_type_i)], centroids[(batch_effect, cell_type_j)])
                        distance = torch.norm(centroid_j - centroid_i, p=2)
                        if not torch.isnan(distance).any():
                            distances.append(distance)
                    except: # Continue if centroids[(batch_effect, cell_type_i)] doesn't exist
                        continue
                average_distance = np.mean(distances)
                distance_std = np.std(distances)
                distance_std_matrix[i, j] = distance_std
                average_distance_matrix[i, j] = average_distance

        # Convert average_distance_matrix into a DataFrame
        average_distance_df = pd.DataFrame(average_distance_matrix, index=adata.obs['cell_type'].unique(), columns=adata.obs['cell_type'].unique())
        
        # Replace NaN values with 0
        average_distance_df = average_distance_df.fillna(0)

        #average_distance_df = average_distance_df/average_distance_df.max().max()

        # Convert distance_std_matrix into a DataFrame
        distance_std_df = pd.DataFrame(distance_std_matrix, index=adata.obs['cell_type'].unique(), columns=adata.obs['cell_type'].unique())
        
        # Replace NaN values with 0
        distance_std_df = distance_std_df.fillna(0)

        return average_distance_df, distance_std_df

    def CalculateDistanceMatrix(self, model_output_dim: int=100):
        """
        Calculate the distance matrix based on PCA space.

        Parameters:
        - model_output_dim (int): Dimensionality of the model output.

        Returns:
        None
        """

        #X = torch.tensor(self.prediction)
        cell_type_vector = self.pred_adata.obs["cell_type"]

        # Calculate the avergae centroid distance between cell type clusters of PCA transformed data
        self.pca_cell_type_centroids_distances_matrix, self.pca_distance_std_df = self.PCA_cell_type_centroid_distances(n_components=model_output_dim)

        """# Step 1: Calculate centroids for each cell type cluster of each batch effect
        centroids = {}
        for cell_type in cell_type_vector.unique():
            mask = (cell_type_vector == cell_type)
            centroid = torch.mean(X[mask], axis=0)
            centroids[cell_type] = centroid

        # Step 2: Calculate the average centroid distance between all batch effects
        average_distance_matrix_input = torch.zeros((len(cell_type_vector.unique()), len(cell_type_vector.unique())))
        for i, cell_type_i in enumerate(cell_type_vector.unique()):
            for j, cell_type_j in enumerate(cell_type_vector.unique()):
                #if (cell_type_i == "Platelets") or (cell_type_j == "Platelets"):
                #    continue
                centroid_i = centroids[cell_type_i]
                centroid_j = centroids[cell_type_j]
                average_distance = torch.norm(centroid_j - centroid_i, p=2)
                average_distance_matrix_input[i, j] = average_distance"""

        # Replace values with 0 if they were 0 in the PCA centorid matrix
        cell_type_centroids_distances_matrix_filter = self.pca_cell_type_centroids_distances_matrix.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]
        #self.average_distance_matrix_input = average_distance_matrix_input

        self.cell_type_centroids_distances_matrix_filter = torch.tensor(cell_type_centroids_distances_matrix_filter.values, dtype=torch.float32)
        
        distance_std_df = self.pca_distance_std_df.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]
        self.distance_std_df = torch.tensor(distance_std_df.values, dtype=torch.float32)

    def DownloadDistanceMatrix(self, name: str="DistanceMatrix"):
        """
        Save the distance matrix to a file.

        Parameters:
        - name (str): Name of the distance matrix file.

        Returns:
        None
        """

        #np.save(f'distance_matrices/{name}.npy', self.average_distance_matrix_input, allow_pickle=False)
        np.save(f'distance_matrices/PCA_{name}.npy', self.cell_type_centroids_distances_matrix_filter, allow_pickle=False)

    def LoadDistanceMatrix(self, name: str="DistanceMatrix"):
        """
        Load a precomputed distance matrix.

        Parameters:
        - name (str): Name of the distance matrix file.

        Returns:
        None
        """

        #self.average_distance_matrix_input = np.load(f'distance_matrices/{name}.npy')
        self.cell_type_centroids_distances_matrix_filter = np.load(f'distance_matrices/PCA_{name}.npy')

    def VisualizeCellTypeCorrelations(self, image_path: str=None):
        """
        Visualize the distance matrices and their statistics.

        Parameters:
        - image_path (str): Directory path to save the visualizations.

        Returns:
        None
        """

        """# Create a heatmap to visualize the relative distance matrix
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(self.average_distance_matrix_input/np.max(self.average_distance_matrix_input), cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Euclidean Distance')
        plt.xticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique())
        plt.title('Cell Type Normalized Euclidean Distance')
        plt.tight_layout()  # Adjust layout for better spacing
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_Model.svg', format='svg')
        plt.show()"""

        # Create a heatmap to visualize the relative distance matrix of the PCA reference
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(self.cell_type_centroids_distances_matrix_filter/np.max(self.cell_type_centroids_distances_matrix_filter), cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Normalized Euclidean Distance')
        plt.xticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique())
        plt.title('Normalized euclidean distance between cell type centorids in PCA latent space')
        plt.tight_layout()  # Adjust layout for better spacing
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_PCA.svg', format='svg')
        plt.show()

        """# Create a heatmap to visualize the distance matrix
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(self.average_distance_matrix_input, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Euclidean Distance')
        plt.xticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique())
        plt.title('Cell Type Euclidean Distance')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()"""

        # Create a heatmap to visualize the distance matrix of the PCA reference
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(self.cell_type_centroids_distances_matrix_filter, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Euclidean Distance')
        plt.xticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique())
        plt.title('Euclidean distance between cell type centorids in PCA latent space')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(self.distance_std_df, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Std of Euclidean Distance')
        plt.xticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique())
        plt.title('Std of euclidean distance between cell type centorids in PCA latent space')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

        CV_df = (self.distance_std_df / self.cell_type_centroids_distances_matrix_filter)
        nan_mask = torch.isnan(CV_df)
        CV_df = torch.where(nan_mask, torch.tensor(0.0), CV_df)

        # Extract upper triangular part of the matrix
        upper_triangular = torch.triu(CV_df)

        # Find non-zero elements and their indices
        non_zero_indices = torch.nonzero(upper_triangular)

        # Extract non-zero elements
        non_zero_elements = CV_df[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        print(non_zero_elements)
        mean_value = torch.mean(non_zero_elements)
        std_value = torch.std(non_zero_elements)
        print("mean CV value: ", mean_value)
        print("std CV value: ", std_value)
        print("Number fo cell types: ", len(self.data_env.adata.obs['cell_type'].unique()))

        
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(CV_df, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Coefficient of variation (CV)')
        plt.xticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique(), rotation=60, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(self.data_env.adata.obs['cell_type'].unique())), self.data_env.adata.obs['cell_type'].unique())
        plt.title('CV of euclidean distance between cell type centorids in PCA latent space')
        plt.tight_layout()  # Adjust layout for better spacing
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_PCA_CV.svg', format='svg')
        plt.show()

        # Create a violin plot of CV scores
        non_zero_elements_np = non_zero_elements.numpy()
        sns.violinplot(y=non_zero_elements_np)

        # Add labels and title
        plt.xlabel("Density")
        plt.ylabel("CV score")
        plt.title("Violin plot of CV scores")
        plt.tight_layout()  # Adjust layout for better spacing
        if image_path:
            plt.savefig(f'{image_path}_PCA_CV_Violin.svg', format='svg')

        # Show the plot
        plt.show()

    def UMAPLatentSpace(self):

        adata_in_house = self.pred_adata.copy()

        sc.pp.neighbors(adata_in_house, use_rep="In_house")

        self.metrics = scib.metrics.metrics(
            self.pred_adata,
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

        print("Metrics: ", self.metrics)

        random_order = np.random.permutation(adata_in_house.n_obs)
        adata_in_house = adata_in_house[random_order, :]

        sc.tl.umap(adata_in_house)
        sc.pl.umap(adata_in_house, color=self.label_key, ncols=1, title="Cell type")
        sc.pl.umap(adata_in_house, color="batch", ncols=1, title="Batch effects")
        del adata_in_house

