import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import random
import json
from tqdm import tqdm
import time as time
import pandas as pd
import copy
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


class prep_data(data.Dataset):
    """
    A class for preparing and handling data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.

    target_key : str
        The key in the adata.obs dictionary specifying the target labels.

    gene2vec_path: str
        Path to gene2vec representations.

    save_model_path: str
        Path to model. Creates a subfolder in this path with information needed for predictions.

    pathways_file_path : str, optional
        The path to a JSON file containing pathway/gene set information (default is None). If defined as None it's not possible to use pathway information. Needs to be defined if a model designed to use pathway information is used.
    
    num_pathways : int, optional
        The number of top pathways to select based on relative HVG abundance (default is None). Only used if json_file_path is given.
    
    pathway_gene_limit : int, optional
        The minimum number of HVGs in a pathway/gene set for it to be considered (default is 10). Only used if json_file_path is given.
    
    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).
    
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True. This option is suitable for certain transformer models.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    batch_keys : list, optional
        A list of keys for batch labels (default is None).
    
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec representations when training the HVG Transformer model relying on tokenization. use_HVG_buckets must be set to True for the use of gene2vec to work.
    
    model_output_dim : int, optional
        Output dimension from the model to be used.

    for_classification : bool, optional
        Whether to process data if it's for classifier training or latent space training (default is False)

    Methods
    -------
    __len__()
        Returns the number of data points in the dataset.

    __getitem(idx)
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

    bucketize_expression_levels(expression_levels, num_buckets)
        Bucketize expression levels into categories based on the specified number of buckets and absolute min/max values.

    bucketize_expression_levels_per_gene(expression_levels, num_buckets)
        Bucketize expression levels into categories based on the specified number of buckets and min/max values of each individual gene.

    make_gene2vec_embedding()
        Match correct gene2vec embeddings to correct genes and return a PyTorch tensor with gene embeddings.
    """

    def __init__(self, 
                 adata, 
                 target_key: str,
                 gene2vec_path: str,
                 save_model_path: str,
                 pathways_file_path: str=None, 
                 num_pathways: int=None,  
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False,
                 model_output_dim: int=100,
                 for_classification: bool=False):
        
        self.adata = adata
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs
        self.pathways_file_path = pathways_file_path
        self.HVG_buckets = HVG_buckets
        self.use_HVG_buckets = use_HVG_buckets
        self.use_gene2vec_emb = use_gene2vec_emb
        self.expression_levels_min = None
        self.expression_levels_max = None
        self.pathway_names = None
        self.feature_means = None
        self.feature_stdevs = None

        # Import gene2vec embeddings
        if use_gene2vec_emb:
            gene2vec_emb = pd.read_csv(gene2vec_path, sep=' ', header=None)
            # Create a dictionary
            self.gene2vec_dic = {row[0]: row[1:201].to_list() for index, row in gene2vec_emb.iterrows()}
            self.gene2vec_tensor = self.make_gene2vec_embedding()

        # Filter highly variable genes if specified
        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.hvg_genes = self.adata.var_names[self.adata.var["highly_variable"]] # Store the HVG names for making predictions later
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        else:
            self.hvg_genes = self.adata.var_names
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        if for_classification:
            # Encode the target information
            self.label_encoder = LabelEncoder()
            self.target_label_encoded = self.label_encoder.fit_transform(self.labels)
            self.onehot_label_encoder = OneHotEncoder()
            self.target = self.onehot_label_encoder.fit_transform(self.target_label_encoded.reshape(-1, 1))
            self.target = self.target.toarray()
        else:
            # Encode the target information
            self.label_encoder = LabelEncoder()
            self.target = self.label_encoder.fit_transform(self.labels)

        # Calculate the avergae centroid distance between cell type clusters of PCA transformed data
        self.cell_type_centroids_distances_matrix = self.cell_type_centroid_distances(n_components=model_output_dim)

        # Encode the batch effect information for each batch key
        if self.batch_keys is not None:
            self.batch_encoders = {}
            self.encoded_batches = []
            for batch_key in self.batch_keys:
                encoder = LabelEncoder()
                encoded_batch = encoder.fit_transform(self.adata.obs[batch_key])
                self.batch_encoders[batch_key] = encoder
                self.encoded_batches.append(encoded_batch)

            self.encoded_batches = [torch.tensor(batch, dtype=torch.long) for batch in self.encoded_batches]

        # Convert expression level to buckets, suitable for nn.Embbeding() used in certain transformer models
        if use_HVG_buckets:
            self.X_not_tokenized = self.X.clone()
            #self.X = self.bucketize_expression_levels(self.X, HVG_buckets)  
            self.X = self.bucketize_expression_levels_per_gene(self.X, HVG_buckets) 

        # Pathway information
        # Load the JSON data into a Python dictionary
        if pathways_file_path is not None:
            with open(pathways_file_path, 'r') as json_file:
                all_pathways = json.load(json_file)

            # Get all gene symbols
            gene_symbols = list(self.adata.var.index)
            # Initiate a all zeros mask
            pathway_mask = np.zeros((len(list(all_pathways.keys())), len(gene_symbols)))
            # List to be filed with number of hvgs per pathway
            num_hvgs = []
            # List to be filed with pathway lengths
            pathway_length = []
            # List to be filed with pathway names
            pathway_names = []
            # Initiate a all zero mask for dispersions_norm
            dispersions_norm_mask = np.zeros((len(list(all_pathways.keys())), len(gene_symbols)))

            for key_idx, key in enumerate(list(all_pathways.keys())):
                pathway = all_pathways[key]
                pathway_length.append(len(pathway))
                pathway_names.append(key)
                # Make mask entries into 1.0 when a HVG is present in the pathway
                for gene_idx, gene in enumerate(gene_symbols):
                    if gene in pathway:
                        pathway_mask[key_idx,gene_idx] = 1.0
                        dispersions_norm_mask[key_idx,gene_idx] = self.adata.var["dispersions_norm"][gene_idx]
                num_hvgs.append(np.sum(pathway_mask[key_idx,:]))

            pathway_length = np.array(pathway_length)
            # Filter so that there must be more than pathway_gene_limit genes in a pathway/gene set
            pathway_mask = pathway_mask[pathway_length>pathway_gene_limit,:]
            dispersions_norm_mask = dispersions_norm_mask[pathway_length>pathway_gene_limit,:]
            pathway_names = np.array(pathway_names)[pathway_length>pathway_gene_limit]
            num_hvgs = np.array(num_hvgs)[pathway_length>pathway_gene_limit]
            pathway_length = pathway_length[pathway_length>pathway_gene_limit]

            # Realtive percentage of HVGs in each pathway
            relative_hvg_abundance = np.sum(pathway_mask, axis=1)/pathway_length

            self.pathway_names = pathway_names[np.argsort(relative_hvg_abundance)[-num_pathways:]]

            # Filter based on realtive percentage of HVGs
            self.pathway_mask = torch.FloatTensor(pathway_mask[np.argsort(relative_hvg_abundance)[-num_pathways:],:])

        # Save information needed to make prediction later on new data
        file_path = f"{save_model_path}/ModelMetadata/"

        # Create folder if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Save gene2vec embeddings
        if use_gene2vec_emb:
            torch.save(self.gene2vec_tensor, f"{file_path}gene2vec_tensor.pt")
        # Save HVG gene names
        torch.save(self.hvg_genes, f"{file_path}hvg_genes.pt")
        # Save gene set mask
        if pathways_file_path is not None:
            torch.save(self.pathway_mask, f"{file_path}gene_set_mask.pt")
        # Save HVG bucket thresholds
        if use_HVG_buckets:
            torch.save(self.all_thresholds_values, f"{file_path}all_bucketization_threshold_values.pt")
        # Save target encoders
        if for_classification:
            torch.save(self.label_encoder, f"{file_path}label_encoder.pt")
            torch.save(self.onehot_label_encoder, f"{file_path}onehot_label_encoder.pt")
    
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

        # Generate buckets
        eps = 1e-6
        self.expression_levels_min, _ = torch.min(expression_levels, dim=0)
        self.expression_levels_max, _ = torch.max(expression_levels, dim=0)
        # Equally spaced buckets in space
        self.all_thresholds_values = []
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            min_scalar = self.expression_levels_min[i].item()
            max_scalar = self.expression_levels_max[i].item()
            thresholds = torch.linspace(min_scalar - eps, max_scalar + eps, steps=num_buckets + 1)
            self.all_thresholds_values.append(thresholds)
            bucketized_levels[:, i] = torch.bucketize(gene_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def make_gene2vec_embedding(self):
        """
        Match correct gene2vec embeddings to correct genes and return a PyTorch tensor with gene embeddings.

        Returns
        ----------
        gene_embeddings_tensor: Tensor
            Tensor containing gene2vec embeddings.
        """
        # Match correct gene2vec embeddings to correct genes
        gene_embeddings_dic = {}
        missing_gene_symbols = []
        gene_symbol_list = list(self.gene2vec_dic.keys())
        gene_symbols_to_use = []
        
        # Get top self.HVGs number of HVGs that has a defined Gene2vec embedding
        if self.HVG:
            adata_temp = self.adata.copy()
            sc.pp.highly_variable_genes(adata_temp, n_top_genes=len(adata_temp.var.index), flavor="cell_ranger")
            
            # Get the normalized dispersion values
            norm_dispersion_values = adata_temp.var['dispersions_norm'].values

            # Create a DataFrame with gene names and corresponding norm dispersion values
            gene_df = pd.DataFrame({'gene': adata_temp.var.index, 'norm_dispersion': norm_dispersion_values})

            # Sort the DataFrame based on norm dispersion values
            sorted_gene_df = gene_df.sort_values(by='norm_dispersion', ascending=False)

            # Get the sorted gene names
            sorted_gene_names = sorted_gene_df['gene'].values

            for gene_symbol in sorted_gene_names:
                if gene_symbol in gene_symbol_list:
                    gene_embeddings_dic[gene_symbol] = self.gene2vec_dic[gene_symbol]
                    gene_symbols_to_use.append(gene_symbol)
                else:
                    missing_gene_symbols.append(gene_symbol)
                # Break ones the requested number of HVGs has been reached
                if len(gene_symbols_to_use) == self.HVGs:
                    break
        else:
            for gene_symbol in self.adata.var.index:
                if gene_symbol in gene_symbol_list:
                    gene_embeddings_dic[gene_symbol] = self.gene2vec_dic[gene_symbol]
                    gene_symbols_to_use.append(gene_symbol)
                else:
                    missing_gene_symbols.append(gene_symbol)

        # Remove genes without a gene2vec representation
        self.adata = self.adata[:, gene_symbols_to_use].copy()
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        self.hvg_genes = gene_symbols_to_use

        # Convert values to a list
        gene_embeddings_tensor = torch.tensor(list(gene_embeddings_dic.values()))

        '''# When gene symbol doesn't have an embedding we'll simply make a one hot encoded embedding for these
        onehot_template = torch.zeros((1,len(missing_gene_symbols)))[0] # one hot embedding template

        # Add zero vector to the end of all embedded gene symbols
        for idx, gene_symbol in enumerate(gene_embeddings_dic.keys()):
            gene_embeddings_dic[gene_symbol] = torch.concatenate([torch.tensor(gene_embeddings_dic[gene_symbol]),onehot_template])
        # Add the one hot encoding for missing gene symbols
        for idx, gene_symbol in enumerate(missing_gene_symbols):
            onehot_temp = onehot_template
            onehot_temp[idx] = 1.0
            gene_embeddings_dic[gene_symbol] = torch.concatenate([torch.zeros((1,200))[0],onehot_temp])
        
        gene_embeddings_tensor = torch.tensor([])
        for gene_symbol in self.adata.var.index:
            gene_embeddings_tensor = torch.cat(gene_embeddings_tensor,gene_embeddings_dic[gene_symbol])

        print(gene_embeddings_tensor.shape)
        sadkjahdjs
        # Convert the list to a PyTorch tensor
        #gene_embeddings_tensor = torch.transpose(torch.tensor(values_list))'''

        return gene_embeddings_tensor
    
    def cell_type_centroid_distances(self, n_components: int=100):
        """
        Calculate the average centroid distances between different cell types across batch effects using PCA.

        Parameters
        -------
        n_components : int, optional 
            Number of principal components to retain after PCA (default is 100).

        Returns
        -------
            average_distance_df: DataFrame containing the normalized average centroid distances between different cell types.
        """

        # Step 1: Perform PCA on AnnData.X
        pca = PCA(n_components=n_components)
        adata = self.adata.copy()  # Make a copy of the original AnnData object
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
                average_distance_matrix[i, j] = average_distance

        # Convert average_distance_matrix into a DataFrame
        average_distance_df = pd.DataFrame(average_distance_matrix, index=self.label_encoder.fit_transform(adata.obs['cell_type'].unique()), columns=self.label_encoder.fit_transform(adata.obs['cell_type'].unique()))

        # Replace NaN values with 0
        average_distance_df = average_distance_df.fillna(0)
        # Normalize to get relative distance
        average_distance_df = average_distance_df/average_distance_df.max().max()

        return average_distance_df

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

        Parameters
        ----------
        idx : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple
            A tuple containing data point, data label, batch information (if available), and selected pathways.
        """

        # Get HVG expression levels
        data = self.X[idx] 

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        if self.use_HVG_buckets == True:
            data_not_tokenized = self.X_not_tokenized[idx] 
        else:
            data_not_tokenized = torch.tensor([])

        return data, data_label, batches, data_not_tokenized
    

class prep_data_validation(data.Dataset):
    """
    A class for preparing and handling data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.

    train_env: prep_data()
        prep_data() environment used for training.

    target_key : str
        The key in the adata.obs dictionary specifying the target labels.

    gene2vec_path: str
        Path to gene2vec representations.

    save_model_path: str
        Path to model. Assumes the saved information needed to make predictions will be located in a subfolder in this directory.

    for_classification : bool, optional
        Whether to process data so it's suitable for classification (True) or only produce latent space (False) (defualt is False)

    pathways_file_path : str, optional
        The path to a JSON file containing pathway/gene set information (default is None). If defined as None it's not possible to use pathway information. Needs to be defined if a model designed to use pathway information is used.
    
    pathway_gene_limit : int, optional
        The minimum number of genes in a pathway/gene set for it to be considered (default is 10). Only used if json_file_path is given.
    
    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).

    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True. This option is suitable for certain transformer models.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    batch_keys : list, optional
        A list of keys for batch labels (default is None).
    
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec representations when training the HVG Transformer model relying on tokenization. use_HVG_buckets must be set to True for the use of gene2vec to work.
        
    Methods
    -------
    __len__()
        Returns the number of data points in the dataset.

    __getitem(idx)
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

    bucketize_expression_levels(expression_levels, num_buckets)
        Bucketize expression levels into categories based on the specified number of buckets and absolute min/max values.

    bucketize_expression_levels_per_gene(expression_levels, num_buckets)
        Bucketize expression levels into categories based on the specified number of buckets and min/max values of each individual gene.

    make_gene2vec_embedding()
        Match correct gene2vec embeddings to correct genes and return a PyTorch tensor with gene embeddings.
    """

    def __init__(self, 
                 adata, 
                 train_env,
                 target_key: str,
                 gene2vec_path: str,
                 save_model_path: str,
                 for_classification: bool=False,
                 pathways_file_path: str=None, 
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False):
        
        self.adata = adata
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs
        self.pathways_file_path = pathways_file_path
        self.HVG_buckets = HVG_buckets
        self.use_HVG_buckets = use_HVG_buckets
        self.use_gene2vec_emb = use_gene2vec_emb

        # Filter highly variable genes if specified
        if HVG:
            self.adata = self.adata[:, train_env.hvg_genes].copy()
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        # Import gene2vec embeddings
        if use_gene2vec_emb:
            #gene2vec_emb = pd.read_csv(gene2vec_path, sep=' ', header=None)
            # Create a dictionary
            #self.gene2vec_dic = {row[0]: row[1:201].to_list() for index, row in gene2vec_emb.iterrows()}
            self.gene2vec_tensor = train_env.gene2vec_tensor

        # Encode the target information
        if for_classification:
            temp = train_env.label_encoder.transform(self.labels)
            self.target = train_env.onehot_label_encoder.transform(temp.reshape(-1, 1))
            self.target = self.target.toarray()
        else:
            self.target = train_env.label_encoder.transform(self.labels)

        # Encode the batch effect information for each batch key
        if self.batch_keys is not None:
            self.encoded_batches = []
            for batch_key in self.batch_keys:
                encoder = train_env.batch_encoders[batch_key]
                encoded_batch = encoder.transform(self.adata.obs[batch_key])
                self.encoded_batches.append(encoded_batch)

            self.encoded_batches = [torch.tensor(batch, dtype=torch.long) for batch in self.encoded_batches]

        # Convert expression level to buckets, suitable for nn.Embbeding() used in certain transformer models
        if use_HVG_buckets:
            self.X_not_tokenized = self.X.clone()
            self.X = self.bucketize_expression_levels_per_gene(self.X, train_env.all_thresholds_values)  

            if torch.max(self.X) == self.HVG_buckets:
                # Mask where the specified value is located
                mask = self.X == self.HVG_buckets

                # Replace the specified value with the new value
                self.X[mask] = self.HVG_buckets - 1

        # Pathway information
        # Load the JSON data into a Python dictionary
        if pathways_file_path is not None:
            self.pathway_mask = train_env.pathway_mask
    
    def bucketize_expression_levels_per_gene(self, 
                                            expression_levels, 
                                            all_thresholds_values: float=None):
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

        # Generate buckets
        eps = 1e-6
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            bucketized_levels[:, i] = torch.bucketize(gene_levels, all_thresholds_values[i])

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def make_gene2vec_embedding(self):
        """
        Match correct gene2vec embeddings to correct genes and return a PyTorch tensor with gene embeddings.

        Returns
        ----------
        gene_embeddings_tensor: Tensor
            Tensor containing gene2vec embeddings.
        """
        # Match correct gene2vec embeddings to correct genes
        gene_embeddings_dic = {}
        missing_gene_symbols = []
        gene_symbol_list = list(self.gene2vec_dic.keys())
        gene_symbols_to_use = []
        for gene_symbol in self.adata.var.index:
            if gene_symbol in gene_symbol_list:
                gene_embeddings_dic[gene_symbol] = self.gene2vec_dic[gene_symbol]
                gene_symbols_to_use.append(gene_symbol)
            else:
                missing_gene_symbols.append(gene_symbol)

        # Remove genes without a gene2vec representation
        self.adata = self.adata[:, gene_symbols_to_use].copy()
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        self.hvg_genes = gene_symbols_to_use

        # Convert values to a list
        gene_embeddings_tensor = torch.tensor(list(gene_embeddings_dic.values()))

        return gene_embeddings_tensor

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

        Parameters
        ----------
        idx : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple
            A tuple containing data point, data label, batch information (if available), and selected pathways.
        """

        # Get HVG expression levels
        data_point = self.X[idx] 

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        #if self.pathways_file_path is not None:
        #    data_pathways = self.X[idx] * self.pathway_mask
        #else:
        #    data_pathways = torch.tensor([])
        
        if self.use_HVG_buckets == True:
            data_pathways = self.X_not_tokenized[idx] 
        else:
            data_pathways = torch.tensor([])

        return data_point, data_label, batches, data_pathways


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
        

class EarlyStopping():
    """
    Early Stopping Callback for Training

    This class is a callback for early stopping during training based on validation loss. It monitors the validation loss and stops training if the loss does not improve for a certain number of consecutive epochs.

    Parameters
    -------
        tolerance (int, optional): Number of evaluations to wait for an improvement in validation loss before stopping. Default is 10.
    """
    
    def __init__(self, tolerance: int=10):

        self.tolerance = tolerance
        self.min_val = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss >= self.min_val:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.min_val = val_loss
            self.counter = 0
    

class CustomSNNLoss(nn.Module):
    """
    A Custom Soft Nearest Neighbor Loss used for training the machine learning models.

    This PyTorch loss function computes the Soft Nearest Neighbor (SNN) loss in such a way that the clustering of samples of the same cell type is reinforced while at the same time promoting the creation of noise in terms of batch effect properties in each cell type cluster.

    Parameters
    ----------
    use_target_weights : bool, optional
        If True, calculate target weights based on label frequency (default is True).
    
    use_batch_weights : bool, optional
        If True, calculate class weights for specified batch effects based on label frequency (default is True).   
    
    targets : Tensor, optional
        A tensor containing the class labels for the input vectors. Required if use_target_weights is True.
    
    batches : Tensor, optional
        A list of tensors containing the batch effect labels. Required if use_batch_weights is True.
    
    batch_keys : list, optional
        A list containing batch keys to account for batch effects (default is None).
    
    temperature : float, optional
        Initial scaling factor applied to the cosine similarity of the target contribution to the loss (default is 0.25).
   
    min_temperature : float, optional
        The minimum temperature value allowed during optimization (default is 0.1).
    
    max_temperature : float, optional
        The maximum temperature value allowed during optimization (default is 1.0).
    
    device : str, optional
        Which device to be used (default is "cuda").

    Methods
    -------
    calculate_class_weights(targets)
        Calculate class weights based on label frequency.
    
    forward(input, targets, batches)
        Compute the SNN loss for the input vectors and targets.
    """

    def __init__(self, 
                 cell_type_centroids_distances_matrix,
                 use_target_weights: bool=True, 
                 use_batch_weights: bool=True, 
                 targets: torch.tensor=None, 
                 batches: list=None,
                 batch_keys: list=None, 
                 temperature: float=0.25, 
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 device: str="cuda"):
        super(CustomSNNLoss, self).__init__()
        
        # Define temperature variables to be optimized durring training
        self.temperature_target = nn.Parameter(torch.tensor(temperature), requires_grad=True) 
        if batch_keys is not None:
            self.temperatures_batches = []
            for _ in range(len(batch_keys)):
                temperature = 0.5 # Set the temperature term for the batch effect contribution to be 0.5
                self.temperatures_batches.append(temperature)

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.device = device
        self.use_target_weights = use_target_weights
        self.use_batch_weights = use_batch_weights
        self.batch_keys = batch_keys
        self.cell_type_centroids_distances_matrix = cell_type_centroids_distances_matrix

        # Calculate weights for the loss based on label frequency
        if self.use_target_weights:
            if targets is not None:
                self.weight_target = self.calculate_class_weights(targets)
            else:
                raise ValueError("'use_target_weights' is True, but 'targets' is not provided.")
        if self.use_batch_weights: 
            if batch_keys is not None:
                self.weight_batch = []
                for i in range(len(batch_keys)):
                    self.weight_batch.append(self.calculate_class_weights(batches[i]))
            else:
                raise ValueError("'use_weights' is True, but 'batch_keys' is not provided.")

    def calculate_class_weights(self, targets):
        """
        Calculate class weights based on label frequency.

        Parameters
        ----------
        targets : Tensor
            A tensor containing the class labels.
        """

        class_counts = torch.bincount(targets)  # Count the occurrences of each class
        class_weights = 1.0 / class_counts.float()  # Calculate inverse class frequencies
        class_weights /= class_weights.sum()  # Normalize to sum to 1

        class_weight_dict = {class_label: weight for class_label, weight in enumerate(class_weights)}

        return class_weight_dict
    
    def cell_type_centroid_distances(self, X, cell_type_vector):
        """
        Calculate the Mean Squared Error (MSE) loss between target centroids and current centroids based on cell type information.

        Parameters:
        X : torch.tensor
            Input data matrix with each row representing a data point and each column representing a feature.
        
        cell_type_vector : torch.tensor
            A vector containing the cell type annotations for each data point in X.

        Returns:
            loss: The MSE loss between target centroids and current centroids.
        """

        # Step 1: Calculate centroids for each cell type cluster 
        centroids = {}
        for cell_type in cell_type_vector.unique():
            mask = (cell_type_vector == cell_type)
            centroid = torch.mean(X[mask], axis=0)
            centroids[cell_type.item()] = centroid

        # Step 2: Calculate the average centroid distance between all cell types
        average_distance_matrix_input = torch.zeros((len(cell_type_vector.unique()), len(cell_type_vector.unique())))
        for i, cell_type_i in enumerate(cell_type_vector.unique()):
            for j, cell_type_j in enumerate(cell_type_vector.unique()):
                centroid_i = centroids[cell_type_i.item()]
                centroid_j = centroids[cell_type_j.item()]
                average_distance = torch.norm(centroid_j - centroid_i, p=2)
                average_distance_matrix_input[i, j] = average_distance

        # Replace values with 0 if they were 0 in the PCA centorid matrix
        cell_type_centroids_distances_matrix_filter = self.cell_type_centroids_distances_matrix.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]
        mask = (cell_type_centroids_distances_matrix_filter != 0.0)
        mask = torch.tensor(mask.values, dtype=torch.float32)
        average_distance_matrix_input = torch.mul(mask, average_distance_matrix_input)
        average_distance_matrix_input = average_distance_matrix_input / torch.max(average_distance_matrix_input)

        cell_type_centroids_distances_matrix_filter = torch.tensor(cell_type_centroids_distances_matrix_filter.values, dtype=torch.float32)

        # Only use non-zero elemnts for loss calculation
        non_zero_mask = cell_type_centroids_distances_matrix_filter != 0
        average_distance_matrix_input = average_distance_matrix_input[non_zero_mask]
        cell_type_centroids_distances_matrix_filter = cell_type_centroids_distances_matrix_filter[non_zero_mask]

        # Step 3: Calculate the MSE between target centroids and current centroids
        # Set to zero if loss can't be calculated, like if there's only one cell type per batch effect element for all elements
        loss = 0
        try:
            loss = F.mse_loss(average_distance_matrix_input, cell_type_centroids_distances_matrix_filter)
        except:
            loss = 0
            pass

        return loss

    def forward(self, input, targets, batches=None):
        """
        Compute the SNN loss for the input vectors and targets.

        Parameters
        ----------
        input : Tensor
            Input vectors (predicted latent space).
        targets : Tensor
            Class labels for the input vectors.
        batches : list, optional
            List of batch keys to account for batch effects.

        Returns
        -------
        loss : Tensor
            The calculated SNN loss.
        """

        ### Target loss

        # Restrict the temperature term
        if self.temperature_target.item() <= self.min_temperature:
            self.temperature_target.data = torch.tensor(self.min_temperature)
        elif self.temperature_target.item() >= self.max_temperature:
            self.temperature_target.data = torch.tensor(self.max_temperature)

        # Calculate the cosine similarity matrix, and also apply exp()
        cosine_similarity_matrix = torch.exp(F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperature_target)

        # Define a loss dictionary containing the loss of each label
        loss_dict = {str(target): torch.tensor([]).to(self.device) for target in targets.unique()}
        for idx, (sim_vec, target) in enumerate(zip(cosine_similarity_matrix,targets)):
            positiv_samples = sim_vec[(targets == target)]
            negativ_samples = sim_vec[(targets != target)]
            # Must be more or equal to 2 samples per sample type for the loss to work
            if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                positiv_sum = torch.sum(positiv_samples) - sim_vec[idx]
                negativ_sum = torch.sum(negativ_samples)
                loss = -torch.log(positiv_sum / (positiv_sum + negativ_sum))
                # New loss:
                #loss = -torch.log(1/positiv_sum)
                loss_dict[str(target)] = torch.cat((loss_dict[str(target)], loss.unsqueeze(0)))

        del cosine_similarity_matrix

        # Calculate the weighted average loss of each cell type
        weighted_losses = []
        for target in targets.unique():
            losses_for_target = loss_dict[str(target)]
            # Make sure there's values in losses_for_target of given target
            if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                if self.use_target_weights:
                    weighted_loss = torch.mean(losses_for_target) * self.weight_target[int(target)]
                else:
                    weighted_loss = torch.mean(losses_for_target)

                weighted_losses.append(weighted_loss)

        # Calculate the sum loss accross cell types
        loss_target = torch.sum(torch.stack(weighted_losses))

        ### Minimize difference between PCA cell type centorid of data and centroids of cell types in latent space
        loss_centorid = self.cell_type_centroid_distances(input, targets)

        ### Batch effect loss

        if batches is not None:

            loss_batches = []
            for outer_idx, batch in enumerate(batches):

                # Calculate the cosine similarity matrix, and also apply exp()
                cosine_similarity_matrix = torch.exp(F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperatures_batches[outer_idx])

                # Define a loss dictionary containing the loss of each label
                loss_dict = {str(target_batch): torch.tensor([]).to(self.device) for target_batch in batch.unique()}
                for idx, (sim_vec, target_batch, target) in enumerate(zip(cosine_similarity_matrix,batch,targets)):
                    positiv_samples = sim_vec[(targets == target) & (batch == target_batch)]
                    negativ_samples = sim_vec[(targets == target) & (batch != target_batch)]
                    # Must be more or equal to 2 samples per sample type for the loss to work
                    if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                        positiv_sum = torch.sum(positiv_samples) - sim_vec[idx]
                        negativ_sum = torch.sum(negativ_samples)
                        #loss = -torch.log(negativ_sum / (positiv_sum + negativ_sum))
                        loss = (-torch.log(positiv_sum / (positiv_sum + negativ_sum)))**-1
                        loss_dict[str(target_batch)] = torch.cat((loss_dict[str(target_batch)], loss.unsqueeze(0)))

                # Calculate the weighted average loss of each batch effect
                losses = []
                for batch_target in batch.unique():
                    losses_for_target = loss_dict[str(batch_target)]
                    # Make sure there's values in losses_for_target of given batch effect
                    if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                        if self.use_batch_weights:
                            temp_loss = torch.mean(losses_for_target) * self.weight_batch[outer_idx][int(batch_target)]
                        else:
                            temp_loss = torch.mean(losses_for_target)
                        losses.append(temp_loss)

                # Only use loss if it was possible to caluclate it from previous steps
                if losses != []:
                    loss_ = torch.sum(torch.stack(losses))
                    loss_batches.append(loss_)

                del cosine_similarity_matrix

            if loss_batches != []:
                loss_batch = torch.mean(torch.stack(loss_batches, dim=0))
            else:
                loss_batch = torch.tensor([0.0]).to(self.device)

            # Apply weights to the two loss contributions
            loss = 0.9*loss_target + 0.1*loss_batch + 1.0*loss_centorid 

            return loss
        else:
            return loss_target + loss_centorid
    

class train_module():
    """
    A class for training the machine learning model using single-cell RNA sequencing data as input and/or pathway/gene set information.

    Parameters
    ----------
    data_path : str or AnnData
        Path to the data file or an AnnData object containing single-cell RNA sequencing data. If a path is provided,
        the data will be loaded from the specified file. If an AnnData object is provided, it will be used directly.

    pathways_file_path : str
        Path to the JSON file containing metadata and pathway information.

    num_pathways : int
        The number of pathways in the dataset.
    
    save_model_path : str
        The path to save the trained model.

    gene2vec_path: str
        Path to gene2vec representations.
    
    pathway_gene_limit : int, optional
        The minimum number of genes in a pathway for the pathway to be considered (default is 10). Only used if json_file_path is given.
    
    HVG : bool, optional
        Whether to identify highly variable genes (HVGs) in the data (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).
    
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    target_key : str, optional
        The metadata key specifying the target variable (default is "cell_type").
    
    batch_keys : list, optional
        List of batch keys to account for batch effects (default is None).
    
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec representations when training the HVG Transformer model relying on tokenization. use_HVG_buckets must be set to True for the use of gene2vec to work.
    
    validation_pct : float, optional
        What percentage of data to use for validation (defualt is 0.2, meaning 20%).
    """

    def __init__(self, 
                 data_path, 
                 save_model_path: str,
                 num_pathways: int=500,
                 pathways_file_path: str=None,
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 target_key: str="cell_type", 
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False,
                 gene2vec_path: str=None,
                 validation_pct: float=0.2):
        
        if type(data_path) == str:
            self.adata = sc.read(data_path, cache=True)
        else:
            self.adata = data_path

        self.HVG = HVG
        self.HVGs = HVGs
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.num_pathways = num_pathways

        # Specify the number of folds (splits)
        if (validation_pct > 0.0) and (validation_pct < 1.0):
            n_splits = int(100/(validation_pct*100))  
        elif validation_pct < 0.0:
            raise ValueError('Invalid choice of validation_pct. Needs to be 0.0 <= validation_pct < 1.0')

        if validation_pct == 0.0:
            self.adata_train = self.adata.copy()
            self.adata_validation = self.adata.copy()
        else:
            # Initialize Stratified K-Fold
            stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Iterate through the folds
            self.adata_train = self.adata.copy()
            self.adata_validation = self.adata.copy()
            for train_index, val_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.target_key]):
                # Filter validation indices based on labels present in the training data
                unique_train_labels = np.unique(self.adata.obs[self.target_key][train_index])
                filtered_val_index = [idx for idx in val_index if self.adata.obs[self.target_key][idx] in unique_train_labels]

                self.adata_train = self.adata_train[train_index, :].copy()
                self.adata_validation = self.adata_validation[filtered_val_index, :].copy()
                break

        self.data_env = prep_data(adata=self.adata_train, 
                                  pathways_file_path=pathways_file_path, 
                                  num_pathways=num_pathways, 
                                  pathway_gene_limit=pathway_gene_limit,
                                  HVG=HVG,  
                                  HVGs=HVGs, 
                                  HVG_buckets=HVG_buckets,
                                  use_HVG_buckets=use_HVG_buckets,
                                  target_key=target_key, 
                                  batch_keys=batch_keys,
                                  use_gene2vec_emb=use_gene2vec_emb,
                                  gene2vec_path=gene2vec_path,
                                  save_model_path=save_model_path)
        
        self.data_env_validation = prep_data_validation(adata=self.adata_validation, 
                                                        train_env = self.data_env,
                                                        pathways_file_path=pathways_file_path, 
                                                        pathway_gene_limit=pathway_gene_limit,
                                                        HVG=HVG,  
                                                        HVGs=HVGs, 
                                                        HVG_buckets=HVG_buckets,
                                                        use_HVG_buckets=use_HVG_buckets,
                                                        target_key=target_key, 
                                                        batch_keys=batch_keys,
                                                        use_gene2vec_emb=use_gene2vec_emb,
                                                        gene2vec_path=gene2vec_path,
                                                        save_model_path=save_model_path)
        
        self.data_env_for_classification = prep_data(adata=self.adata_train, 
                                                    pathways_file_path=pathways_file_path, 
                                                    num_pathways=num_pathways, 
                                                    pathway_gene_limit=pathway_gene_limit,
                                                    HVG=HVG,  
                                                    HVGs=HVGs, 
                                                    HVG_buckets=HVG_buckets,
                                                    use_HVG_buckets=use_HVG_buckets,
                                                    target_key=target_key, 
                                                    batch_keys=batch_keys,
                                                    use_gene2vec_emb=use_gene2vec_emb,
                                                    gene2vec_path=gene2vec_path,
                                                    save_model_path=save_model_path,
                                                    for_classification=True)
        
        self.data_env_validation_for_classification = prep_data_validation(adata=self.adata_validation, 
                                                        train_env = self.data_env_for_classification,
                                                        pathways_file_path=pathways_file_path, 
                                                        pathway_gene_limit=pathway_gene_limit,
                                                        HVG=HVG,  
                                                        HVGs=HVGs, 
                                                        HVG_buckets=HVG_buckets,
                                                        use_HVG_buckets=use_HVG_buckets,
                                                        target_key=target_key, 
                                                        batch_keys=batch_keys,
                                                        use_gene2vec_emb=use_gene2vec_emb,
                                                        gene2vec_path=gene2vec_path,
                                                        save_model_path=save_model_path,
                                                        for_classification=True)
        

        self.save_model_path = save_model_path

    
    def train_model(self,
                    model, 
                    model_name,
                    optimizer, 
                    lr_scheduler, 
                    loss_module, 
                    device, 
                    out_path, 
                    train_loader, 
                    val_loader, 
                    num_epochs, 
                    eval_freq,
                    earlystopping_threshold,
                    use_classifier,
                    only_print_best: bool=False,
                    accum_grad: int=1,
                    model_classifier: nn.Module=None):
        """
        Don't use this function by itself! It's aimed to be used in the train() function.
        """

        print()
        print(f"Start Training")
        print()

        # Add model to device
        model.to(device)
        if use_classifier:
            model_classifier.to(device)

        # Initiate EarlyStopping
        early_stopping = EarlyStopping(earlystopping_threshold)

        # Define gene2vec_tensor if gene2ve is used
        if self.data_env.use_gene2vec_emb:
            gene2vec_tensor = self.data_env.gene2vec_tensor
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    gene2vec_tensor = torch.cat((gene2vec_tensor, self.data_env.gene2vec_tensor), dim=0)
            gene2vec_tensor = gene2vec_tensor.to(device)

        # Training loop
        best_val_loss = np.inf  
        best_epoch = 0
        train_start = time.time()
        for epoch in tqdm(range(num_epochs)):

            # Training
            if use_classifier:
                model.eval()
                model_classifier.train()
            else:
                model.train()
            train_loss = []
            all_preds_train = []
            all_labels_train = []
            batch_idx = -1
            for data_inputs, data_labels, data_batches, data_not_tokenized in train_loader:
                batch_idx += 1

                data_labels = data_labels.to(device)
                data_inputs_step = data_inputs.to(device)
                data_not_tokenized_step = data_not_tokenized.to(device)

                if model_name == "Model3":
                    if self.data_env.use_gene2vec_emb:
                        preds = model(data_inputs_step, data_not_tokenized_step, gene2vec_tensor)
                    else:
                        preds = model(data_inputs_step, data_not_tokenized_step)
                elif model_name == "Model2":
                    if self.data_env.use_gene2vec_emb:
                        preds = model(data_inputs_step, gene2vec_tensor)
                    else:
                        preds = model(data_inputs_step)
                elif model_name == "Model1":
                    preds = model(data_inputs_step)

                if use_classifier:
                    preds_latent = preds.cpu().detach().to(device)
                    preds = model_classifier(preds_latent)
                
                # Whether to use classifier loss or latent space creation loss
                if use_classifier:
                    loss = loss_module(preds, data_labels)/accum_grad
                else:
                    if self.batch_keys is not None:
                        data_batches = [batch.to(device) for batch in data_batches]
                        loss = loss_module(preds, data_labels, data_batches)/accum_grad
                    else:
                        loss = loss_module(preds, data_labels)/accum_grad

                loss.backward()

                train_loss.append(loss.item())

                # Perform updates to model weights
                if ((batch_idx + 1) % accum_grad == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                #optimizer.step()
                #optimizer.zero_grad()

                all_preds_train.extend(preds.cpu().detach().numpy())
                all_labels_train.extend(data_labels.cpu().detach().numpy())

            # Validation
            if (epoch % eval_freq == 0) or (epoch == (num_epochs-1)):
                model.eval()
                if use_classifier:
                    model_classifier.eval()
                val_loss = []
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for data_inputs, data_labels, data_batches, data_not_tokenized in val_loader:

                        data_inputs_step = data_inputs.to(device)
                        data_labels_step = data_labels.to(device)
                        data_not_tokenized_step = data_not_tokenized.to(device)

                        if model_name == "Model3":
                            if self.data_env.use_gene2vec_emb:
                                preds = model(data_inputs_step, data_not_tokenized_step, gene2vec_tensor)
                            else:
                                preds = model(data_inputs_step, data_not_tokenized_step)
                        elif model_name == "Model2":
                            if self.data_env.use_gene2vec_emb:
                                preds = model(data_inputs_step, gene2vec_tensor)
                            else:
                                preds = model(data_inputs_step)
                        elif model_name == "Model1":
                            preds = model(data_inputs_step)

                        if use_classifier:
                            preds_latent = preds.cpu().detach().to(device)
                            preds = model_classifier(preds_latent)

                        # Check and fix the number of dimensions
                        if preds.dim() == 1:
                            preds = preds.unsqueeze(0)  # Add a dimension along axis 0

                        # Whether to use classifier loss or latent space creation loss
                        if use_classifier:
                            loss = loss_module(preds, data_labels_step)/accum_grad
                        else:
                            if self.batch_keys is not None:
                                data_batches = [batch.to(device) for batch in data_batches]
                                loss = loss_module(preds, data_labels_step, data_batches) /accum_grad
                            else:
                                loss = loss_module(preds, data_labels_step)/accum_grad

                        val_loss.append(loss.item())
                        all_preds.extend(preds.cpu().detach().numpy())
                        all_labels.extend(data_labels_step.cpu().detach().numpy())

                # Metrics
                avg_train_loss = sum(train_loss) / len(train_loss)
                avg_val_loss = sum(val_loss) / len(val_loss)

                # Check early stopping
                early_stopping(avg_val_loss)

                # Print epoch information
                if use_classifier:

                    binary_preds_train = []
                    # Loop through the predictions
                    for pred in all_preds_train:
                        # Apply thresholding
                        binary_pred = np.argmax(pred)

                        binary_preds_train.append(binary_pred)

                    # Convert the list of arrays to a numpy array
                    binary_preds_train = np.array(binary_preds_train)

                    binary_labels_train = []
                    for pred in all_labels_train:
                        binary_pred = np.argmax(pred)

                        binary_labels_train.append(binary_pred)

                    binary_labels_train = np.array(binary_labels_train)

                    binary_preds_valid = []
                    for label in all_preds:
                        binary_pred = np.argmax(label)

                        binary_preds_valid.append(binary_pred)

                    binary_preds_valid = np.array(binary_preds_valid)

                    binary_labels_valid = []
                    for label in all_labels:
                        binary_pred = np.argmax(label)

                        binary_labels_valid.append(binary_pred)

                    binary_labels_valid = np.array(binary_labels_valid)

                    # Calculate accuracy
                    accuracy_train = accuracy_score(binary_labels_train, binary_preds_train)
                    accuracy = accuracy_score(binary_labels_valid, binary_preds_valid)

                    if only_print_best == False:
                        print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Training Accuracy: {accuracy_train} | Validation loss: {avg_val_loss:.4f} | Validation Accuracy: {accuracy}")
                else:
                    if only_print_best == False:
                        print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

                # Apply early stopping
                if early_stopping.early_stop:
                    print(f"Stopped training using EarlyStopping at epoch {epoch+1}")
                    break

                # Save model if performance has improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1

                    if use_classifier:
                        # Move the model to CPU before saving
                        model_classifier.to('cpu')
                        
                        # Save the entire model to a file
                        torch.save(model_classifier.module.state_dict() if hasattr(model_classifier, 'module') else model_classifier.state_dict(), f'{out_path}model_classifier.pt')
                        
                        # Move the model back to the original device
                        model_classifier.to(device)
                    else:
                        # Move the model to CPU before saving
                        model.to('cpu')
                        
                        # Save the entire model to a file
                        #torch.save(model, f'{out_path}model.pt')
                        #torch.save(model.state_dict(), f'{out_path}model.pt')
                        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), f'{out_path}model.pt')
                        
                        # Move the model back to the original device
                        model.to(device)

            # Update learning rate
            lr_scheduler.step()

        print()
        print(f"**Finished training**")
        print()
        print(f"Best validation loss (reached after {best_epoch} epochs): {best_val_loss}")
        print()
        train_end = time.time()
        print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

        return best_val_loss
    
    def train(self, 
                 model: nn.Module,
                 model_name: str,
                 device: str=None,
                 seed: int=42,
                 batch_size: int=236,
                 use_target_weights: bool=True,
                 use_batch_weights: bool=True,
                 init_temperature: float=0.25,
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 init_lr: float=0.001,
                 lr_scheduler_warmup: int=4,
                 lr_scheduler_maxiters: int=100,
                 eval_freq: int=1,
                 epochs: int=100,
                 earlystopping_threshold: int=10,
                 accum_grad: int=1):
        """
        Perform training of the machine learning model.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        
        device : str or None, optional
            The device to run the training on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
        
        seed : int, optional
            Random seed for ensuring reproducibility (default is 42).
        
        batch_size : int, optional
            Batch size for data loading during training (default is 256).

        use_target_weights : bool, optional
            If True, calculate target weights based on label frequency (default is True).
        
        use_batch_weights : bool, optional
            If True, calculate class weights for specified batch effects based on label frequency (default is True).   
        
        init_temperature : float, optional
            Initial temperature for the loss function (default is 0.25).
        
        min_temperature : float, optional
            The minimum temperature value allowed during optimization (default is 0.1).
        
        max_temperature : float, optional
            The maximum temperature value allowed during optimization (default is 1.0).
        
        init_lr : float, optional
            Initial learning rate for the optimizer (default is 0.001).
        
        lr_scheduler_warmup : int, optional
            Number of warm-up iterations for the cosine learning rate scheduler (default is 4).
        
        lr_scheduler_maxiters : int, optional
            Maximum number of iterations for the cosine learning rate scheduler (default is 25).
        
        eval_freq : int, optional
            Rate at which the model is evaluated on validation data (default is 2).
        
        epochs : int, optional
            Number of training epochs (default is 20).
        
        earlystopping_threshold : int, optional
            Early stopping threshold (default is 10).

        train_classifier : bool, optional
            Whether to train the model as a classifier (True) or just to generate latent space (False) (default is False)

        Returns
        -------
        all_preds : list
            List of predictions.
        """

        model_step_1 = copy.deepcopy(model)

        out_path = self.save_model_path

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Ensure reproducibility
        def rep_seed(seed):
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        rep_seed(seed)

        total_train_start = time.time()

        train_loader = data.DataLoader(self.data_env, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = data.DataLoader(self.data_env_validation, batch_size=batch_size, shuffle=True, drop_last=True)
        #val_loader = data.DataLoader(self.data_env_validation, batch_size=batch_size, shuffle=True, drop_last=True)

        total_params = sum(p.numel() for p in model_step_1.parameters())
        print(f"Number of parameters: {total_params}")

        # Define custom SNN loss
        loss_module = CustomSNNLoss(cell_type_centroids_distances_matrix=self.data_env.cell_type_centroids_distances_matrix,
                                    use_target_weights=use_target_weights, 
                                    use_batch_weights=use_batch_weights, 
                                    targets=torch.tensor(self.data_env.target), 
                                    batches=self.data_env.encoded_batches, 
                                    batch_keys=self.batch_keys, 
                                    temperature=init_temperature, 
                                    min_temperature=min_temperature, 
                                    max_temperature=max_temperature)
        
        # Define Adam optimer
        optimizer = optim.Adam([{'params': model_step_1.parameters(), 'lr': init_lr}, {'params': loss_module.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        
        # Define scheduler for the learning rate
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)

        # To run on multiple GPUs:
        if torch.cuda.device_count() > 1:
            model_step_1= nn.DataParallel(model_step_1)

        # Train
        _ = self.train_model(model=model_step_1, 
                        model_name=model_name,
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        loss_module=loss_module, 
                        device=device, 
                        out_path=out_path,
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        num_epochs=epochs, 
                        eval_freq=eval_freq,
                        earlystopping_threshold=earlystopping_threshold,
                        accum_grad=accum_grad,
                        use_classifier=False)

        del model_step_1, loss_module, optimizer, lr_scheduler

        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")
        print()

    def train_classifier(self, 
                        model: nn.Module,
                        model_name: str,
                        model_classifier: nn.Module=None,
                        device: str=None,
                        seed: int=42,
                        init_lr: float=0.001,
                        batch_size: int=256,
                        lr_scheduler_warmup: int=4,
                        lr_scheduler_maxiters: int=100,
                        eval_freq: int=5,
                        epochs: int=100,
                        earlystopping_threshold: int=10,
                        accum_grad: int=1,
                        only_print_best: bool=False):

        print("Start training classifier")
        print()

        if model_classifier == None:
            raise ValueError('Need to define model_classifier if train_classifier=True.')
        
        out_path = self.save_model_path

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Ensure reproducibility
        def rep_seed(seed):
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        rep_seed(seed)

        total_train_start = time.time()

        model_step_2 = copy.deepcopy(model)

        # Load model state
        model_step_2.load_state_dict(torch.load(f'{out_path}model.pt'))

        # Define data
        train_loader = data.DataLoader(self.data_env_for_classification, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = data.DataLoader(self.data_env_validation_for_classification, batch_size=batch_size, shuffle=True, drop_last=True)

        # Define loss
        loss_module = nn.CrossEntropyLoss() 
        # Define Adam optimer
        optimizer = optim.Adam([{'params': model_classifier.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)

        # Train
        val_loss = self.train_model(model=model_step_2, 
                        model_name=model_name,
                        model_classifier=model_classifier,
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        loss_module=loss_module, 
                        device=device, 
                        out_path=out_path,
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        num_epochs=epochs, 
                        eval_freq=eval_freq,
                        earlystopping_threshold=earlystopping_threshold,
                        accum_grad=accum_grad,
                        use_classifier=True,
                        only_print_best=only_print_best)
        
        del model_step_2, loss_module, optimizer, lr_scheduler
        
        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")

        return val_loss
    