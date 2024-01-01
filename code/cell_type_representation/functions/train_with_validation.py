import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from functions import data_preprocessing as dp
import random
import json
from tqdm import tqdm
import time as time
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean


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

    pathways_file_path : str, optional
        The path to a JSON file containing pathway/gene set information (default is None). If defined as None it's not possible to use pathway information. Needs to be defined if a model designed to use pathway information is used.
    
    num_pathways : int, optional
        The number of top pathways to select based on relative HVG abundance (default is None). Only used if json_file_path is given.
    
    pathway_gene_limit : int, optional
        The minimum number of genes in a pathway/gene set for it to be considered (default is 10). Only used if json_file_path is given.
    
    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 4000).
    
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True. This option is suitable for certain transformer models.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    Scaled : bool, optional
        Whether to scale the data so that the mean of each feature becomes zero and std becomes the approximate std of each individual feature (default is False).
    
    batch_keys : list, optional
        A list of keys for batch labels (default is None).
    
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec representations when training the HVG Transformer model relying on tokenization. use_HVG_buckets must be set to True for the use of gene2vec to work.
    
    model_output_dim : int, optional
        Output dimension from the model to be used.

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
                 pathways_file_path: str=None, 
                 num_pathways: int=None,  
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 Scaled: bool=False, 
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False,
                 model_output_dim: int=100):
        
        self.adata = adata
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs
        self.scaled = Scaled
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

        # Import gene2vec embeddings
        #if use_gene2vec_emb:
        #    gene2vec_emb = pd.read_csv(gene2vec_path, sep=' ', header=None)
        #    # Create a dictionary
        #    self.gene2vec_dic = {row[0]: row[1:201].to_list() for index, row in gene2vec_emb.iterrows()}
        #    self.gene2vec_tensor = self.make_gene2vec_embedding()

        if Scaled:
            self.adata.X, self.feature_means, self.feature_stdevs = dp.scale_data(self.adata.X, return_mean_and_std=True)
            self.X = self.adata.X

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

            self.pathway_names = pathway_names[np.argsort(relative_hvg_abundance)[-num_pathways:],:]

            # Filter based on realtive percentage of HVGs
            self.pathway_mask = torch.FloatTensor(pathway_mask[np.argsort(relative_hvg_abundance)[-num_pathways:],:])

    def bucketize_expression_levels(self, expression_levels, num_buckets: int):
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

        self.expression_levels_min = torch.min(expression_levels)
        self.expression_levels_max = torch.max(expression_levels)

        # Generate continuous thresholds
        eps = 1e-6
        thresholds = torch.linspace(self.expression_levels_min - eps, self.expression_levels_max + eps, steps=num_buckets + 1)

        # Generate buckets
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

        # Generate buckets
        eps = 1e-6
        min_values, _ = torch.min(expression_levels, dim=0)
        max_values, _ = torch.max(expression_levels, dim=0)
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            min_scalar = min_values[i].item()
            max_scalar = max_values[i].item()
            thresholds = torch.linspace(min_scalar - eps, max_scalar + eps, steps=num_buckets + 1)
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
        data_point = self.X[idx] 

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        if (self.use_HVG_buckets == True) and (self.pathways_file_path is not None):
            data_pathways = self.X_not_tokenized[idx] * self.pathway_mask
        elif (self.use_HVG_buckets == True) and (self.pathways_file_path is None):
            data_pathways = self.X_not_tokenized[idx] 
        elif self.pathways_file_path is not None:
            data_pathways = self.X[idx] * self.pathway_mask
        else:
            data_pathways = torch.tensor([])

        return data_point, data_label, batches, data_pathways
    

class prep_data_validation(data.Dataset):
    """
    A class for preparing and handling data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.

    label_encoder : Labelencoder()
        Should be the Labelencoder() fitted during training for targets.

    batch_encoders : Labelencoder()
        Should be the Labelencoder() fitted during training for batch effects.

    target_key : str
        The key in the adata.obs dictionary specifying the target labels.

    gene2vec_path: str
        Path to gene2vec representations.

    feature_means: float
        If Scaled is True during training, the mean values need to be given here (default is None).

    feature_stdevs: float
        If Scaled is True during training, the std values need to be given here (default is None).

    expression_levels_min: float
        Min expression value in training data (default is None). Need to be specified if bucketized HVGs were used, meaning use_HVG_buckets is True.

    expression_levels_max: float
        Max expression value in training data (default is None). Need to be specified if bucketized HVGs were used, meaning use_HVG_buckets is True.

    pathway_names_to_use: str
        List of pathway names used for training (default is None). Need to be specified if pathways_file_path is defined.

    pathways_file_path : str, optional
        The path to a JSON file containing pathway/gene set information (default is None). If defined as None it's not possible to use pathway information. Needs to be defined if a model designed to use pathway information is used.
    
    pathway_gene_limit : int, optional
        The minimum number of genes in a pathway/gene set for it to be considered (default is 10). Only used if json_file_path is given.
    
    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 4000).

    HVG_list : list, optional
        List containing all HVGs used for training (default is None). Needs to be sepcified if HVG is True.
    
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True. This option is suitable for certain transformer models.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    Scaled : bool, optional
        Whether to scale the data so that the mean of each feature becomes zero and std becomes the approximate std of each individual feature (default is False).
    
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
                 label_encoder,
                 batch_encoders,
                 target_key: str,
                 gene2vec_path: str,
                 feature_means: float=None,
                 feature_stdevs: float=None,
                 expression_levels_min: float=None,
                 expression_levels_max: float=None,
                 pathway_names_to_use: list=None,
                 pathways_file_path: str=None, 
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 HVG_list: list=None,
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 Scaled: bool=False, 
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False):
        
        self.adata = adata
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs
        self.scaled = Scaled
        self.pathways_file_path = pathways_file_path
        self.HVG_buckets = HVG_buckets
        self.use_HVG_buckets = use_HVG_buckets
        self.use_gene2vec_emb = use_gene2vec_emb

        # Filter highly variable genes if specified
        if HVG:
            self.adata = self.adata[:, HVG_list].copy()
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        # Import gene2vec embeddings
        if use_gene2vec_emb:
            gene2vec_emb = pd.read_csv(gene2vec_path, sep=' ', header=None)
            # Create a dictionary
            self.gene2vec_dic = {row[0]: row[1:201].to_list() for index, row in gene2vec_emb.iterrows()}
            self.gene2vec_tensor = self.make_gene2vec_embedding()

        if Scaled:
            self.adata.X = dp.scale_data(data=self.adata.X, feature_means=feature_means, feature_stdevs=feature_stdevs)
            self.X = self.adata.X

        # Encode the target information
        self.target = label_encoder.transform(self.labels)

        # Encode the batch effect information for each batch key
        if self.batch_keys is not None:
            self.encoded_batches = []
            for batch_key in self.batch_keys:
                encoder = batch_encoders[batch_key]
                encoded_batch = encoder.transform(self.adata.obs[batch_key])
                self.encoded_batches.append(encoded_batch)

            self.encoded_batches = [torch.tensor(batch, dtype=torch.long) for batch in self.encoded_batches]

        # Convert expression level to buckets, suitable for nn.Embbeding() used in certain transformer models
        if use_HVG_buckets:
            self.X = self.bucketize_expression_levels(self.X, HVG_buckets, expression_levels_min, expression_levels_max)  

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

            # Filter based on realtive percentage of HVGs
            self.pathway_mask = torch.FloatTensor(pathway_mask[np.isin(pathway_names, pathway_names_to_use),:])

    def bucketize_expression_levels(self, 
                                    expression_levels,
                                    num_buckets: int,
                                    expression_levels_min: float=None,
                                    expression_levels_max: float=None):
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

        if (expression_levels_min==None) and (expression_levels_max==None):
            self.expression_levels_min = torch.min(expression_levels)
            self.expression_levels_max = torch.max(expression_levels)
        else:
            self.expression_levels_min = expression_levels_min
            self.expression_levels_max = expression_levels_max

        # Generate continuous thresholds
        eps = 1e-6
        thresholds = torch.linspace(self.expression_levels_min - eps, self.expression_levels_max + eps, steps=num_buckets + 1)

        # Generate buckets
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

        # Generate buckets
        eps = 1e-6
        min_values, _ = torch.min(expression_levels, dim=0)
        max_values, _ = torch.max(expression_levels, dim=0)
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            min_scalar = min_values[i].item()
            max_scalar = max_values[i].item()
            thresholds = torch.linspace(min_scalar - eps, max_scalar + eps, steps=num_buckets + 1)
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

        if self.pathways_file_path is not None:
            data_pathways = self.X[idx] * self.pathway_mask
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
                 targets=None, 
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
        The number of highly variable genes to select (default is 4000).
    
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000). Only used if use_HVG_buckets is set to True.
    
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False). This option is required to be set to True if using a HVG transformer model relying on tokenization.
    
    Scaled : bool, optional
        Whether to scale the data so that the mean of each feature becomes zero and std becomes the approximate std of each individual feature (default is False).
    
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
                 pathways_file_path: str,
                 num_pathways: int,
                 save_model_path: str,
                 gene2vec_path: str,
                 pathway_gene_limit: int=10,
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 HVG_buckets: int=1000,
                 use_HVG_buckets: bool=False,
                 Scaled: bool=False, 
                 target_key: str="cell_type", 
                 batch_keys: list=None,
                 use_gene2vec_emb: bool=False,
                 validation_pct: float=0.2):
        
        if type(data_path) == str:
            self.adata = sc.read(data_path, cache=True)
        else:
            self.adata = data_path

        self.HVG = HVG
        self.HVGs = HVGs
        self.Scaled = Scaled
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.num_pathways = num_pathways

        # Specify the number of folds (splits)
        n_splits = int(100/(validation_pct*100))  

        # Initialize Stratified K-Fold
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Iterate through the folds
        self.adata_train = self.adata.copy()
        self.adata_validation = self.adata.copy()
        for train_index, val_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.target_key]):
            self.adata_train = self.adata_train[train_index, :].copy()
            self.adata_validation = self.adata_validation[val_index, :].copy()
            break

        self.data_env = prep_data(adata=self.adata_train, 
                                  pathways_file_path=pathways_file_path, 
                                  num_pathways=num_pathways, 
                                  pathway_gene_limit=pathway_gene_limit,
                                  HVG=HVG,  
                                  HVGs=HVGs, 
                                  HVG_buckets=HVG_buckets,
                                  use_HVG_buckets=use_HVG_buckets,
                                  Scaled=Scaled,
                                  target_key=target_key, 
                                  batch_keys=batch_keys,
                                  use_gene2vec_emb=use_gene2vec_emb,
                                  gene2vec_path=gene2vec_path)
        
        self.data_env_validation = prep_data_validation(adata=self.adata_validation, 
                                                        label_encoder=self.data_env.label_encoder,
                                                        batch_encoders=self.data_env.batch_encoders,
                                                        pathways_file_path=pathways_file_path, 
                                                        pathway_gene_limit=pathway_gene_limit,
                                                        pathway_names_to_use=self.data_env.pathway_names,
                                                        HVG=HVG,  
                                                        HVGs=HVGs, 
                                                        HVG_list=self.data_env.hvg_genes,
                                                        HVG_buckets=HVG_buckets,
                                                        use_HVG_buckets=use_HVG_buckets,
                                                        Scaled=Scaled,
                                                        feature_means=self.data_env.feature_means,
                                                        feature_stdevs=self.data_env.feature_stdevs,
                                                        target_key=target_key, 
                                                        batch_keys=batch_keys,
                                                        use_gene2vec_emb=use_gene2vec_emb,
                                                        gene2vec_path=gene2vec_path,
                                                        expression_levels_max=self.data_env.expression_levels_max,
                                                        expression_levels_min=self.data_env.expression_levels_min)
        

        self.save_model_path = save_model_path

    
    def train_model(self,
                    model, 
                    optimizer, 
                    lr_scheduler, 
                    loss_module, 
                    device, 
                    out_path, 
                    train_loader, 
                    val_loader, 
                    num_epochs, 
                    eval_freq,
                    earlystopping_threshold):
        """
        Don't use this function by itself! It's aimed to be used in the train() function.
        """

        print()
        print(f"Start Training")
        print()

        # Add model to device
        model.to(device)

        # Initiate EarlyStopping
        early_stopping = EarlyStopping(earlystopping_threshold)

        # Define gene2vec_tensor if gene2ve is used
        if self.data_env.use_gene2vec_emb:
            gene2vec_tensor = self.data_env.gene2vec_tensor
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    gene2vec_tensor = torch.cat((gene2vec_tensor, self.data_env.gene2vec_tensor), dim=0)
            gene2vec_tensor = gene2vec_tensor.to(device)
            #gene2vec_tensor = self.data_env.gene2vec_tensor.to(device)

        # Training loop
        best_val_loss = np.inf  
        train_start = time.time()
        for epoch in tqdm(range(num_epochs)):

            # Training
            model.train()
            train_loss = []
            for data_inputs, data_labels, data_batches, data_pathways in train_loader:

                data_labels = data_labels.to(device)

                # Calculate the number of iterations needed
                num_iterations = (data_inputs.shape[0] + self.batch_size_step_size - 1) // self.batch_size_step_size

                # Store preds without remembering gradient. Used for calculating loss for sub-parts of the batch.
                all_train_preds = torch.tensor([]).to(device)
                if num_iterations > 1:
                    with torch.no_grad():
                        for i in range(num_iterations):
                            start_index = i * self.batch_size_step_size
                            end_index = (i + 1) * self.batch_size_step_size if i < num_iterations - 1 else data_inputs.shape[0]

                            data_inputs_step = data_inputs[start_index:end_index,:].to(device)
                            data_pathways_step = data_pathways[start_index:end_index,:].to(device)

                            if self.data_env.use_gene2vec_emb:
                                preds = model(data_inputs_step, data_pathways_step, gene2vec_tensor)
                            else:
                                preds = model(data_inputs_step, data_pathways_step)

                            # Check and fix the number of dimensions
                            if preds.dim() == 1:
                                preds = preds.unsqueeze(0)  # Add a dimension along axis 0

                            all_train_preds = torch.cat((all_train_preds, preds), dim=0)

                train_loss_temp = []
                for i in range(num_iterations):
                    start_index = i * self.batch_size_step_size
                    end_index = (i + 1) * self.batch_size_step_size if i < num_iterations - 1 else data_inputs.shape[0]

                    data_inputs_step = data_inputs[start_index:end_index,:].to(device)
                    data_pathways_step = data_pathways[start_index:end_index,:].to(device)

                    if self.data_env.use_gene2vec_emb:
                        preds = model(data_inputs_step, data_pathways_step, gene2vec_tensor)
                    else:
                        preds = model(data_inputs_step, data_pathways_step)
                
                    #print(f"Works {i}: ",torch.cuda.memory_allocated())
                    #print("Works: ",torch.cuda.memory_cached())

                    if num_iterations > 1:
                        all_train_preds_temp = all_train_preds.clone()
                        all_train_preds_temp[start_index:end_index,:] = preds
                    else:
                        all_train_preds_temp = preds

                    # Check and fix the number of dimensions
                    if all_train_preds_temp.dim() == 1:
                        all_train_preds_temp = all_train_preds_temp.unsqueeze(0)  # Add a dimension along axis 0

                    if self.batch_keys is not None:
                        data_batches = [batch.to(device) for batch in data_batches]
                        loss = loss_module(all_train_preds_temp, data_labels, data_batches) / num_iterations
                    else:
                        loss = loss_module(all_train_preds_temp, data_labels) / num_iterations

                    loss.backward()

                    train_loss_temp.append(loss.item())

                train_loss.append(np.sum(train_loss_temp))

                optimizer.step()
                optimizer.zero_grad()

            # Validation
            if (epoch % eval_freq == 0) or (epoch == (num_epochs-1)):
                model.eval()
                val_loss = []
                all_preds = []
                with torch.no_grad():
                    for data_inputs, data_labels, data_batches, data_pathways in val_loader:

                        data_inputs_step = data_inputs.to(device)
                        data_labels_step = data_labels.to(device)
                        data_pathways_step = data_pathways.to(device)

                        if self.data_env.use_gene2vec_emb:
                            preds = model(data_inputs_step, data_pathways_step, gene2vec_tensor)
                        else:
                            preds = model(data_inputs_step, data_pathways_step)

                        # Check and fix the number of dimensions
                        if preds.dim() == 1:
                            preds = preds.unsqueeze(0)  # Add a dimension along axis 0

                        if self.batch_keys is not None:
                            data_batches = [batch.to(device) for batch in data_batches]
                            loss = loss_module(preds, data_labels_step, data_batches) 
                        else:
                            loss = loss_module(preds, data_labels_step)

                        val_loss.append(loss.item())
                        all_preds.extend(preds.cpu().detach().numpy())

                # Metrics
                avg_train_loss = sum(train_loss) / len(train_loss)
                avg_val_loss = sum(val_loss) / len(val_loss)

                # Check early stopping
                early_stopping(avg_val_loss)

                # Print epoch information
                print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

                # Apply early stopping
                if early_stopping.early_stop:
                    print(f"Stopped training using EarlyStopping at epoch {epoch+1}")
                    break

                # Save model if performance has improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_preds = all_preds
                    
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
        train_end = time.time()
        print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

        return best_val_loss, best_preds
    
    def train(self, 
                 model: nn.Module,
                 device: str=None,
                 seed: int=42,
                 batch_size: int=256,
                 batch_size_step_size: int=256,
                 use_target_weights: bool=True,
                 use_batch_weights: bool=True,
                 init_temperature: float=0.25,
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 init_lr: float=0.001,
                 lr_scheduler_warmup: int=4,
                 lr_scheduler_maxiters: int=25,
                 eval_freq: int=2,
                 epochs: int=20,
                 earlystopping_threshold: int=10):
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

        batch_size_step_size: int, optional
            Step size to take to reach batch_size samples where the gradient is accumulated after each step and parameters of the model are updated after batch_size samples have been processed (default is 256).
        
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

        Returns
        -------
        all_preds : list
            List of predictions.
        """

        if batch_size_step_size > batch_size:
            raise ValueError("batch_size_step_size must be smaller or equal to batch_size.")
        
        self.batch_size_step_size = batch_size_step_size
        self.batch_size = batch_size


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

        all_preds = []
        total_train_start = time.time()

        train_loader = data.DataLoader(self.data_env, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = data.DataLoader(self.data_env_validation, batch_size=batch_size, shuffle=True)

        total_params = sum(p.numel() for p in model.parameters())
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
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': init_lr}, {'params': loss_module.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        
        # Define scheduler for the learning rate
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)
        out_path = self.save_model_path

        # To run on multiple GPUs:
        if torch.cuda.device_count() > 1:
            model= nn.DataParallel(model)

        # Train
        loss, preds = self.train_model(model=model, 
                                    optimizer=optimizer, 
                                    lr_scheduler=lr_scheduler, 
                                    loss_module=loss_module, 
                                    device=device, 
                                    out_path=out_path,
                                    train_loader=train_loader, 
                                    val_loader=val_loader,
                                    num_epochs=epochs, 
                                    eval_freq=eval_freq,
                                    earlystopping_threshold=earlystopping_threshold)
        
        all_preds.extend(preds)

        del model, loss_module, optimizer, lr_scheduler

        print()
        print(f"Loss score: {np.mean(loss):.4f}")
        print()

        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")

        return all_preds
    
    
    def predict(self, data_, model_path: str, model=None, batch_size: int=32, device: str=None, return_attention: bool=False):
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
        preds : np.array
            Array of predicted latent embeddings.
        """

        data_ = prep_test_data(data_, self.data_env)

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model == None:
            model = torch.load(f'{model_path}model.pt')
        else:
            model.load_state_dict(torch.load(f'{model_path}model.pt'))
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

        preds = []
        model.eval()
        with torch.no_grad():
            for data_inputs, data_pathways in data_loader:

                data_inputs = data_inputs.to(device)
                data_pathways = data_pathways.to(device)

                if self.data_env.use_gene2vec_emb:
                    if return_attention:
                        _, pred = model(data_inputs, data_pathways, gene2vec_tensor, return_attention)
                    else:
                        pred = model(data_inputs, data_pathways, gene2vec_tensor, return_attention)
                else:
                    pred = model(data_inputs, data_pathways)

                pred = pred.cpu().detach().numpy()

                # Ensure all tensors have at least two dimensions
                if pred.ndim == 1:
                    pred = np.expand_dims(pred, axis=0)  # Add a dimension along axis 0

                preds.extend(pred)

        return np.array(preds)

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

        if (self.use_HVG_buckets == True) and (self.pathways_file_path is not None):
            data_pathways = self.X_not_tokenized[idx] * self.pathway_mask
        elif (self.use_HVG_buckets == True) and (self.pathways_file_path is None):
            data_pathways = self.X_not_tokenized[idx] 
        elif self.pathways_file_path is not None:
            data_pathways = self.X[idx] * self.pathway_mask
        else:
            data_pathways = torch.tensor([])

        return data_point, data_pathways

