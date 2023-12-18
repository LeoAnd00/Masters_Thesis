
import scanpy as sc
from functions import train as trainer
from functions import data_preprocessing as dp
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class VisualizeEnv():

    def __init__(self):
        pass

    def MakeAttentionMatrix(self, 
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
        self.attention_matrices = self.predict(data_=self.pred_adata, model_path=model_path, batch_size=batch_size)

        print(list(self.attention_matrices.keys()))
        
    def predict(self, data_, model_path: str, batch_size: int=32, device: str=None):
        """
        Generate latent represntations for data using the trained model.

        Parameters
        ----------
        data_ : AnnData
            An AnnData object containing data for prediction.
        model_path : str
            The path to the directory where the trained model is saved.
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

        model = torch.load(f'{model_path}.pt')
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

        model.eval()
        with torch.no_grad():
            for data_inputs, data_labels, data_pathways in data_loader:

                data_inputs = data_inputs.to(device)
                data_pathways = data_pathways.to(device)

                _, attn_matrix = model(data_inputs, data_pathways, gene2vec_tensor, return_attention=True)

                for label in torch.unique(data_labels):
                    attn_matrices[label] = attn_matrices[label] + torch.sum(torch.stack(attn_matrix[data_labels==label]), dim=0) / target_counts[label]
        
        for key, value in attn_matrices.items():
            attn_matrices[key] = value.numpy()

        return attn_matrices
    
    def GetCellTypeNames(self):
        print(list(self.attention_matrices.keys()))
    
    def HeatMapVisualization(self, cell_type: str, num_of_top_genes: int = 100):
        def select_positions_based_on_sum(attention_matrix, num_positions=100):
            # Calculate row and column sums
            row_sums = np.sum(attention_matrix, axis=1)
            col_sums = np.sum(attention_matrix, axis=0)

            # Get indices of the top rows and columns
            top_row_indices = np.argsort(row_sums)[-num_positions:]
            top_col_indices = np.argsort(col_sums)[-num_positions:]

            gene_names = self.data_env.hvg_genes[top_row_indices]

            # Create a set of unique positions
            selected_positions = set((i, j) for i in top_row_indices for j in top_col_indices)

            return selected_positions, gene_names

        selected_positions, gene_names = select_positions_based_on_sum(self.attention_matrices[cell_type], num_positions=num_of_top_genes)

        # Create a dataframe for seaborn
        df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

        # Select the top genes
        df = df.loc[gene_names, gene_names]

        # Visualize the heatmap with seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, cmap='plasma', annot=False, linewidths=.5, fmt=".2f", cbar_kws={'label': 'Attention'})

        # Set axis titles
        plt.xlabel("Gene")
        plt.ylabel("Gene")

        plt.title(f'Attention Matrix Heatmap of Top {num_of_top_genes} Genes for {cell_type}')
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

