
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
from scipy.spatial.distance import pdist, squareform
from models import model_tokenized_hvg_transformer as model_tokenized_hvg_transformer

class VisualizeEnv():

    def __init__(self):
        pass

    def MakeAttentionMatrix(self, 
                            attention_matrix_or_just_env: bool,
                            train_path: str, 
                            pred_path: str, 
                            gene2vec_path: str,
                            model_path: str,
                            target_key: str,
                            batch_key: str,
                            HVGs: int=2000, 
                            HVG_buckets_: int=1000,
                            batch_size: int=32):
        
        self.train_adata = sc.read(train_path, cache=True)
        self.train_adata.obs["batch"] = self.train_adata.obs[batch_key]

        self.train_env = trainer.train_module(data_path=self.train_adata,
                                        pathways_file_path=None,
                                        num_pathways=300,
                                        pathway_gene_limit=10,
                                        save_model_path="",
                                        HVG=True,
                                        HVGs=HVGs,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=True,
                                        Scaled=False,
                                        target_key=target_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=True,
                                        gene2vec_path=gene2vec_path)
        
        self.data_env = self.train_env.data_env
        
        self.pred_adata = sc.read(pred_path, cache=True)

        # Define model
        model = model_tokenized_hvg_transformer.CellType2VecModel(input_dim=min([HVGs,int(self.data_env.X.shape[1])]),
                                                        output_dim=100,
                                                        drop_out=0.2,
                                                        act_layer=nn.ReLU,
                                                        norm_layer=nn.LayerNorm,#nn.BatchNorm1d, LayerNorm
                                                        attn_embed_dim=24*4,
                                                        num_heads=4,
                                                        mlp_ratio=4,
                                                        attn_bias=False,
                                                        attn_drop_out=0.,
                                                        depth=3,
                                                        nn_tokens=HVG_buckets_,
                                                        nn_embedding_dim=self.data_env.gene2vec_tensor.shape[1],
                                                        use_gene2vec_emb=True)

        if attention_matrix_or_just_env:
            self.attention_matrices, self.latent_space_dictionary = self.predict(data_=self.pred_adata, model=model, model_path=model_path, batch_size=batch_size)

            print(list(self.attention_matrices.keys()))
        
    def predict(self, data_, model_path: str, model=None, batch_size: int=32, device: str=None):
        """
        Generate latent represntations for data using the trained model.

        Parameters
        ----------
        data_ : AnnData
            An AnnData object containing data for prediction.
        model_path : str
            The path to the directory where the trained model is saved.
        model : nn.Module
            If the model is saved as torch.save(model.state_dict(), f'{out_path}model.pt') one have to input a instance of the model. If torch.save(model, f'{out_path}model.pt') was used then leave this as None (default is None).
        batch_size : int, optional
            Batch size for data loading during prediction (default is 32).
        device : str or None, optional
            The device to run the prediction on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.

        Returns
        -------
        attn_matrices : np.array
            Matrix containing the averaged attention matrix of each cell type cluster in latent space.
        """

        data_ = prep_test_data(data_, self.data_env)

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model == None:
            model = torch.load(f'{model_path}.pt')
        else:
            model.load_state_dict(torch.load(f'{model_path}.pt'))
        # To run on multiple GPUs:
        if torch.cuda.device_count() > 1:
            model= nn.DataParallel(model)
        model.to(device)

        data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

        # Define gene2vec_tensor if gene2ve is used
        if self.data_env.use_gene2vec_emb:
            gene2vec_tensor = self.data_env.gene2vec_tensor
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    gene2vec_tensor = torch.cat((gene2vec_tensor, self.data_env.gene2vec_tensor), dim=0)
            gene2vec_tensor = gene2vec_tensor.to(device)
            #gene2vec_tensor = self.data_env.gene2vec_tensor.to(device)

        # Make dictionary for the attention matrices
        labels = data_.adata.obs[self.data_env.target_key]
        target_counts = labels.value_counts()
        unique_targets = list(set(labels))

        # Specify the size of the matrix
        matrix_size = (len(self.data_env.hvg_genes), len(self.data_env.hvg_genes)) 
        # Create a dictionary with unique targets as keys and matrices as values
        attn_matrices = {target: torch.zeros(matrix_size) for target in unique_targets}
        # Create a dictionary with unique targets as keys and latent space representations as values
        latent_space_dictionary = {target: torch.zeros(100) for target in unique_targets}

        model.eval()
        with torch.no_grad():
            for data_inputs, data_labels, data_pathways in tqdm(data_loader):

                data_inputs = data_inputs.to(device)
                data_labels = list(data_labels)
                data_pathways = data_pathways.to(device)

                latent_space, attn_matrix = model(data_inputs, data_pathways, gene2vec_tensor, return_attention=True)

                for label in (list(set(data_labels))):
                    
                    # Attention matrices
                    mask = torch.tensor([data_label == label for data_label in data_labels])

                    average_along_attention_head_dim = torch.mean(attn_matrix[mask,:,:,:], dim=1)

                    sum_along_samples = torch.sum(average_along_attention_head_dim, dim=0).to('cpu')

                    attn_matrices[label] = attn_matrices[label] + sum_along_samples / target_counts[label]

                    # Latent space representations
                    latent_space_sum_along_samples = torch.sum(latent_space, dim=0).to('cpu')

                    latent_space_dictionary[label] = latent_space_dictionary[label] + latent_space_sum_along_samples / target_counts[label]
                
                #print(attn_matrices[list(set(data_labels))[0]][:5,:5])
                #break
        
        for key, value in attn_matrices.items():
            attn_matrices[key] = value.numpy()
        for key, value in latent_space_dictionary.items():
            latent_space_dictionary[key] = value.numpy()

        return attn_matrices, latent_space_dictionary
    
    def GetCellTypeNames(self):
        print(list(self.attention_matrices.keys()))

    def DownloadAttentionMatrixDictionary(self, name: str="AttetionMatrixDictionary"):

        # Save the dictionary to a NumPy .npz file
        np.savez(f'attention_matrices/{name}.npz', **self.attention_matrices)

    def LoadAttentionMatrixDictionary(self, name: str="AttetionMatrixDictionary"):
    
        # Load the dictionary back from the .npz file
        self.attention_matrices = np.load(f'attention_matrices/{name}.npz')

    def DownloadLatentSpaceDictionary(self, name: str="LatentSpaceDictionary"):

        # Save the dictionary to a NumPy .npz file
        np.savez(f'latent_space_dictionary/{name}.npz', **self.latent_space_dictionary)

    def LoadLatentSpaceDictionary(self, name: str="LatentSpaceDictionary"):
    
        # Load the dictionary back from the .npz file
        self.latent_space_dictionary = np.load(f'latent_space_dictionary/{name}.npz')

    
    def HeatMapVisualization(self, cell_type: str, num_of_top_genes: int = 50):
        def select_positions_based_on_sum(attention_matrix, num_positions=50):
            # Calculate row and column sums
            row_sums = np.sum(attention_matrix, axis=1) / np.sum(attention_matrix)
            col_sums = np.sum(attention_matrix, axis=0) / np.sum(attention_matrix)

            # Get indices of the top rows and columns
            top_row_indices = np.argsort(row_sums)[-num_positions:]
            top_col_indices = np.argsort(col_sums)[-num_positions:]

            gene_names_row = self.data_env.hvg_genes[top_row_indices]
            row_sums = row_sums[top_row_indices]
            gene_names_col = self.data_env.hvg_genes[top_col_indices]
            col_sums = col_sums[top_col_indices]

            return gene_names_row, row_sums, gene_names_col, col_sums

        gene_names_row, row_sums, gene_names_col, col_sums = select_positions_based_on_sum(self.attention_matrices[cell_type], num_positions=num_of_top_genes)

        # Create a dataframe for seaborn
        df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

        # Select the top genes
        df = df.loc[gene_names_col, gene_names_col]

        # Visualize the heatmap with seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, cmap='plasma', annot=False, linewidths=.5, fmt=".2f", cbar_kws={'label': 'Attention'})

        # Set axis titles
        plt.xlabel("Gene")
        plt.ylabel("Gene")

        plt.title(f'Attention Matrix Heatmap of Top {num_of_top_genes} Genes for {cell_type}')
        plt.show()

    def BarPlotVisualization(self, cell_type: str="Classical Monocytes", row_or_col: str="col", num_of_top_genes: int = 50, color: str="green"):
        matrix = self.attention_matrices[cell_type]

        # Calculate the sum of each row and column
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)

        # Get indices of the top rows and columns
        top_row_indices = np.argsort(row_sums)[-num_of_top_genes:]
        top_col_indices = np.argsort(col_sums)[-num_of_top_genes:]

        # Calculate the relative sum (percentage) for each row and column
        relative_row_sums = row_sums[top_row_indices] / np.sum(row_sums)
        relative_col_sums = col_sums[top_col_indices] / np.sum(col_sums)

        # Gene names
        gene_names_row = self.data_env.hvg_genes[top_row_indices]
        gene_names_col = self.data_env.hvg_genes[top_col_indices]

        # Plot horizontal bar plots for relative sums
        fig, axes = plt.subplots(1, 1, figsize=(8, int(num_of_top_genes*2/10)))

        if row_or_col == "row":
            axes.barh(gene_names_row, relative_row_sums, color=color)
            axes.set_title('Relative Sum of Row Genes')
            axes.set_xlabel('Relative Sum')
            axes.set_ylabel('Gene')
        elif row_or_col == "col":
            axes.barh(gene_names_col, relative_col_sums, color=color)
            axes.set_title('Relative Sum of Column Genes')
            axes.set_xlabel('Relative Sum')
            axes.set_ylabel('Gene')

        plt.tight_layout()
        plt.show()

    def BarPlotOfIndividualGenes(self, cell_type: str="Classical Monocytes", gene: str="LYZ", num_of_top_genes: int = 50, color: str="blue"):

        df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

        # Select the top genes
        df = df.loc[gene, :] / np.sum(df.loc[gene, :])

        top_indices = np.argsort(df)[-num_of_top_genes:]

        df = df[top_indices]
        gene_names= self.data_env.hvg_genes[top_indices]

        # Plot horizontal bar plots for relative sums
        fig, axes = plt.subplots(1, 1, figsize=(8, int(num_of_top_genes*2/10)))

        axes.barh(gene_names, df, color=color)
        axes.set_title(f'Attention between {gene} and top {num_of_top_genes} genes with highest attention score')
        axes.set_xlabel('Relative attention')
        axes.set_ylabel('Gene')

        plt.tight_layout()
        plt.show()

    def VisualizeCellTypeCorrelations(self, attention_or_latent: str="attention"):
        
        if attention_or_latent == "attention":
            data = self.attention_matrices
        elif attention_or_latent == "latent":
            data = self.latent_space_dictionary
        
        # Make a dictionary containing the vector representations of each cell type where each dimesnion is a gene.
        cell_type_vectors = {}
        for cell_type in list(data.keys()):

            matrix = data[cell_type]

            if attention_or_latent == "attention":
                # Calculate the sum of each column
                col = np.sum(matrix, axis=0)
            elif attention_or_latent == "latent":
                col = matrix

            # Calculate the relative sum (percentage) for each column
            relative_sums = col / np.sum(col)

            # drop Erythroid-like and erythroid precursor cells due to the bigg values
            if (attention_or_latent == "latent") and (cell_type == 'Erythroid-like and erythroid precursor cells'):
                continue

            cell_type_vectors[cell_type] = relative_sums

        # Calculate the eucledian distance between all cell type vectors to create a distance matrix
        cell_types = list(cell_type_vectors.keys())
        vectors = [cell_type_vectors[cell_type] for cell_type in cell_types]

        # Calculate pairwise Euclidean distances
        distance_matrix = squareform(pdist(vectors, metric='euclidean'))

        # Create a heatmap to visualize the distance matrix
        plt.figure(figsize=(10, 8))
        heatmap = plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Euclidean Distance')
        plt.xticks(range(len(cell_types)), cell_types, rotation=45, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(cell_types)), cell_types)
        plt.title('Cell Type Correlations')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    def GeneVsCellTypeVisualization(self, num_of_top_genes: int = 20):
        
        df_all = []
        all_top_indices = []
        for cell_type in list(self.attention_matrices.keys()):

            df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

            col = np.sum(df, axis=0)
            
            relative_sums = col / np.sum(col)

            if cell_type == list(self.attention_matrices.keys())[0]:
                df_all = pd.DataFrame(data=relative_sums, index=self.data_env.hvg_genes, columns=[cell_type])
            else:
                temp = pd.DataFrame(data=relative_sums, index=self.data_env.hvg_genes, columns=[cell_type])
                df_all = pd.concat([df_all, temp], axis=1)

            top_indices = np.argsort(relative_sums)[-num_of_top_genes:]
            all_top_indices.extend(top_indices)

        unique_top_indices = list(set(all_top_indices))
        df = df_all.iloc[unique_top_indices,:]
        gene_names= df.index.tolist()
        cell_types = df.columns.tolist()

        print(f"Number of genes: {len(gene_names)}")

        # Create a heatmap to visualize 
        plt.figure(figsize=(int(2*num_of_top_genes), 26))
        heatmap = plt.imshow(df, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Relative attention')
        plt.xticks(range(len(cell_types)), cell_types, rotation=45, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(gene_names)), gene_names)
        plt.title('Cell Type Correlations to Genes')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()


class prep_test_data(data.Dataset):
    """
    PyTorch Dataset for preparing test data for the machine learning model.

    Parameters:
        adata : AnnData
            An AnnData object containing single-cell RNA sequencing data.
        prep_data_env 
            The data environment used for preprocessing training data.

    Methods:
        __len__()
            Returns the number of data samples.

        __getitem__(idx) 
            Retrieves a specific data sample by index.

        bucketize_expression_levels(expression_levels, num_buckets)
            Bucketize expression levels into categories based on the specified number of buckets and absolute min/max values.

        bucketize_expression_levels_per_gene(expression_levels, num_buckets)
            Bucketize expression levels into categories based on the specified number of buckets and min/max values of each individual gene.
    """

    def __init__(self, adata, prep_data_env):
        self.adata = adata
        self.adata = self.adata[:, prep_data_env.hvg_genes].copy()
        if prep_data_env.scaled:
            self.adata.X = dp.scale_data(data=self.adata.X, feature_means=prep_data_env.feature_means, feature_stdevs=prep_data_env.feature_stdevs)

        self.X = self.adata.X
        self.X = torch.tensor(self.X)

        self.pathways_file_path = prep_data_env.pathways_file_path
        self.use_HVG_buckets = prep_data_env.use_HVG_buckets

        self.labels = self.adata.obs[prep_data_env.target_key]
        self.target = self.labels #prep_data_env.label_encoder.transform(self.labels)

        # Pathway information
        if self.pathways_file_path is not None:
            self.pathway_mask = prep_data_env.pathway_mask
        
        if prep_data_env.use_HVG_buckets:
            self.training_expression_levels = prep_data_env.X_not_tokenized
            self.X_not_tokenized = self.X.clone()
            #self.X = self.bucketize_expression_levels(self.X, prep_data_env.HVG_buckets) 
            self.X = self.bucketize_expression_levels_per_gene(self.X, prep_data_env.HVG_buckets) 

    def bucketize_expression_levels(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and absolut min/max values.

        Parameters
        ----------
        expression_levels : Tensor
            Should be the expression levels (adata.X, or in this case self.X).

        num_buckets : int
            Number of buckets to create.

        Returns
        ----------
        bucketized_levels : LongTensor
            Bucketized expression levels.
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

         # Generate continuous thresholds
        eps = 1e-6
        thresholds = torch.linspace(torch.min(self.training_expression_levels) - eps, torch.max(self.training_expression_levels) + eps, steps=num_buckets + 1)

        bucketized_levels = torch.bucketize(expression_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def bucketize_expression_levels_per_gene(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and min/max values of each individual gene.

        Parameters
        ----------
        expression_levels : Tensor
            Should be the expression levels (adata.X, or in this case self.X).

        num_buckets : int
            Number of buckets to create.

        Returns
        ----------
        bucketized_levels : LongTensor
            Bucketized expression levels.
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

        # Generate continuous thresholds
        eps = 1e-6
        min_values, _ = torch.min(self.training_expression_levels, dim=0)
        max_values, _ = torch.max(self.training_expression_levels, dim=0)
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            min_scalar = min_values[i].item()
            max_scalar = max_values[i].item()
            thresholds = torch.linspace(min_scalar - eps, max_scalar + eps, steps=num_buckets + 1)
            bucketized_levels[:, i] = torch.bucketize(gene_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)

    def __len__(self):
        """
        Get the number of data samples in the dataset.

        Returns
        ----------
        int: The number of data samples.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get a specific data sample by index.

        Parameters
        ----------
        idx (int): Index of the data sample to retrieve.

        Returns
        ----------
        tuple: A tuple containing the data point and pathways.
        """

        data_point = self.X[idx]

        # Get labels
        data_label = self.target[idx]

        if (self.use_HVG_buckets == True) and (self.pathways_file_path is not None):
            data_pathways = self.X_not_tokenized[idx] * self.pathway_mask
        elif (self.use_HVG_buckets == True) and (self.pathways_file_path is None):
            data_pathways = self.X_not_tokenized[idx] 
        elif self.pathways_file_path is not None:
            data_pathways = self.X[idx] * self.pathway_mask
        else:
            data_pathways = torch.tensor([])

        return data_point, data_label, data_pathways

