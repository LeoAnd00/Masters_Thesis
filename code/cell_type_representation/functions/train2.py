import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import scanpy as sc
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from functions import data_preprocessing as dp
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
from tqdm import tqdm
import time as time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from functions import data_preprocessing as dp


class prep_data(data.Dataset):
    """
    A class for preparing and handling data.

    ...

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.
    target_key : str
        The key in the adata.obs dictionary specifying the target labels.
    json_file_path : str, optional
        The path to a JSON file containing pathway information (default is None). If defined as None it's not possible to use pathway information.
    num_pathways : int, optional
        The number of top pathways to select based on relative HVG abundance (default is None). Only used if json_file_path is given.
    pathway_hvg_limit : int, optional
        The minimum number of HVGs in a pathway to consider (default is 10).
    pathways_buckets : int, optional
        The number of buckets for binning pathway information (default is 100).
    use_pathway_buckets : bool, optional
        Whether to use pathway information representations as relativ percentage of active genes of a pathway and to use buckets (True). Or To simply apply the binary pathway mask on the expression levels (False) (default is False).
    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    HVGs : int, optional
        The number of highly variable genes to select (default is 4000).
    HVG_buckets : int, optional
        The number of buckets for binning HVG expression levels (default is 1000).
    use_HVG_buckets : bool, optional
        Whether to use buckets for HVG expression levels (True). Or to not use buckets (False) (defualt is False).
    Scaled : bool, optional
        Whether to scale the data (default is False).
    batch_keys : list, optional
        A list of keys for batch labels (default is None).

    Attributes
    ----------
    X : torch.Tensor
        The input data tensor.
    labels : pd.Series
        The target labels.
    pathway_mask : torch.Tensor
        The binary mask representing selected pathways.
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec embbedings or not.

    Methods
    -------
    __len__()
        Returns the number of data points in the dataset.
    __getitem(idx)
        Returns a data point, its label, batch information, and selected pathways for a given index.
    """

    def __init__(self, 
                 adata, 
                 target_key: str,
                 json_file_path: str=None, 
                 num_pathways: int=None,  
                 pathway_hvg_limit: int=10,
                 pathways_buckets: int=100,
                 use_pathway_buckets: bool=True,
                 HVG: bool=True, 
                 HVGs: int=4000, 
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
        self.json_file_path = json_file_path
        self.pathways_buckets = pathways_buckets
        self.HVG_buckets = HVG_buckets
        self.use_pathway_buckets = use_pathway_buckets
        self.use_HVG_buckets = use_HVG_buckets
        self.use_gene2vec_emb = use_gene2vec_emb

        # Some processing depending on what's specified in the call
        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.hvg_genes = self.adata.var_names[self.adata.var["highly_variable"]] # Store the HVG names for making predictions later
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        else:
            self.hvg_genes = self.adata.var_names
        if Scaled:
            self.adata.X, self.feature_means, self.feature_stdevs = dp.scale_data(self.adata.X, return_mean_and_std=True)
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        # Encode the target information
        self.label_encoder = LabelEncoder()
        self.target = self.label_encoder.fit_transform(self.labels)

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

        # Convert expression level to buckets, suitable for nn.Embbeding()
        if use_HVG_buckets:
            self.X = self.bucketize_expression_levels_per_gene(self.X, HVG_buckets)  

        # Pathway information
        # Load the JSON data into a Python dictionary
        if json_file_path is not None:
            with open(json_file_path, 'r') as json_file:
                all_pathways = json.load(json_file)

            # Get all gene symbols
            gene_symbols = list(self.adata.var.index)
            # Initiate a all zeros mask
            pathway_mask = np.zeros((len(list(all_pathways.keys())), len(gene_symbols)))
            # List to be filed with pathway lengths
            num_hvgs = []
            # List to be filed with number of hvgs per pathway
            pathway_length = []
            # List to be filed with pathway names
            pathway_names = []
            for key_idx, key in enumerate(list(all_pathways.keys())):
                pathway = all_pathways[key]
                pathway_length.append(len(pathway))
                pathway_names.append(key)
                # Make mask entries into 1.0 when a HVG is present in the pathway
                for gene_idx, gene in enumerate(gene_symbols):
                    if gene in pathway:
                        pathway_mask[key_idx,gene_idx] = 1.0
                num_hvgs.append(np.sum(pathway_mask[key_idx,:]))

            pathway_length = np.array(pathway_length)
            # Filter so that there must be more than pathway_hvg_limit HVGs
            pathway_mask = pathway_mask[pathway_length>pathway_hvg_limit,:]
            pathway_names = np.array(pathway_names)[pathway_length>pathway_hvg_limit]
            num_hvgs = np.array(num_hvgs)[pathway_length>pathway_hvg_limit]
            pathway_length = pathway_length[pathway_length>pathway_hvg_limit]

            # Realtive percentage of HVGs in each pathway
            relative_hvg_abundance = np.sum(pathway_mask, axis=1)/pathway_length
            #print("Index: ", np.argsort(relative_hvg_abundance)[-num_pathways:])
            #print("Number of HVGs: ", pathway_length[np.argsort(relative_hvg_abundance)[-num_pathways:]])
            #print("Relative HVG abundance: ", relative_hvg_abundance[np.argsort(relative_hvg_abundance)[-num_pathways:]])
            #print("Pathway names: ", pathway_names[np.argsort(relative_hvg_abundance)[-num_pathways:]])
            #ldsfjhsdj

            # Filter the pathway_mask to contain the top pathways with highest relative HVG abundance
            #random_order = np.random.permutation(len(relative_hvg_abundance))[:num_pathways]
            #self.pathway_mask = torch.FloatTensor(pathway_mask[random_order,:])
            self.pathway_mask = torch.FloatTensor(pathway_mask[np.argsort(relative_hvg_abundance)[-num_pathways:],:])

            if use_pathway_buckets:
                self.num_hvgs = torch.tensor(num_hvgs[np.argsort(relative_hvg_abundance)[-num_pathways:]])

                self.pathway_information = torch.tensor([])
                for idx in range(len(self.X)):
                    # Apply pathway mask to HVGs
                    data_pathways = torch.where(self.X[idx] * self.pathway_mask != 0, torch.tensor(1.0), torch.tensor(0.0))
                    data_pathways = torch.sum(data_pathways, dim=1).squeeze() / self.num_hvgs
                    #data_pathways = torch.where(data_pathways != 0, torch.tensor(1.0), torch.tensor(0.0))
                    self.pathway_information = torch.cat((self.pathway_information, data_pathways.unsqueeze(0)))

                # Bucketize levels, making it suitable for nn.Embedding() 
                self.pathway_information = self.bucketize_expression_levels(self.pathway_information, pathways_buckets)

        # Import gene2vec embeddings
        if use_gene2vec_emb:
            gene2vec_emb = pd.read_csv('../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt', sep=' ', header=None)
            # Create a dictionary
            self.gene2vec_dic = {row[0]: row[1:201].to_list() for index, row in gene2vec_emb.iterrows()}
            self.gene2vec_tensor = self.make_gene2vec_embedding()

    def bucketize_expression_levels(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and absolut min/max values.

        Parameters:
            expression_levels: Tensor, shape [samples, num_genes]
            num_buckets: Number of buckets to create

        Returns:
            bucketized_levels: LongTensor, shape [samples, num_genes]
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

        # Generate continuous thresholds
        eps = 1e-6
        thresholds = torch.linspace(torch.min(expression_levels) - eps, torch.max(expression_levels) + eps, steps=num_buckets + 1)

        bucketized_levels = torch.bucketize(expression_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def bucketize_expression_levels_per_gene(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and min/max values of each individual gene.

        Parameters:
            expression_levels: Tensor, shape [samples, num_genes]
            num_buckets: Number of buckets to create

        Returns:
            bucketized_levels: LongTensor, shape [samples, num_genes]
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

        # Generate continuous thresholds
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
        # Match correct gene2vec embeddings to correct genes
        gene_embeddings_dic = {}
        missing_gene_symbols = []
        gene_symbol_list = list(self.gene2vec_dic.keys())
        for gene_symbol in self.adata.var.index:
            if gene_symbol in gene_symbol_list:
                gene_embeddings_dic[gene_symbol] = self.gene2vec_dic[gene_symbol]
            else:
                #print(f"Gene symbol {gene_symbol} doesn't exists in embedded format")
                missing_gene_symbols.append(gene_symbol)

        #print("Number of missing gene symbols: ", len(missing_gene_symbols))

        # When gene symbol doesn't have an embedding we'll simply make a one hot encoded embedding for these
        onehot_template = torch.zeros((1,len(missing_gene_symbols)))[0] # one hot embedding template

        # Add zero vector to the end of all embedded gene symbols
        for idx, gene_symbol in enumerate(gene_embeddings_dic.keys()):
            gene_embeddings_dic[gene_symbol] = torch.concatenate([gene_embeddings_dic[gene_symbol],onehot_template])
        # Add the one hot encoding for missing gene symbols
        for idx, gene_symbol in enumerate(missing_gene_symbols):
            onehot_temp = onehot_template.copy()
            onehot_temp[idx] = 1.0
            gene_embeddings_dic[gene_symbol] = torch.concatenate([torch.zeros((1,200))[0],onehot_temp])
        
        # Convert values to a list
        values_list = list(gene_embeddings_dic.values())

        # Convert the list to a PyTorch tensor
        gene_embeddings_tensor = torch.transpose(torch.tensor(values_list))

        #print(f"Final length of embedding: {len(gene_embeddings_dic[list(gene_embeddings_dic.keys())[0]])}")
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
        Returns a data point, its label, batch information, and selected pathways for a given index.

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
        data_point = self.X[idx] #torch.where(self.X[idx] != 0, torch.tensor(1.0), torch.tensor(0.0)) #self.X[idx]

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        if self.json_file_path is not None:
            if self.use_pathway_buckets:
                # Extract bucketed percentage of active genes of total possible HVGs in each pathway
                data_pathways = self.pathway_information[idx]
            else:
                # Apply pathway mask to HVGs
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

    Parameters:
        tolerance (int, optional): Number of epochs to wait for an improvement in validation loss before stopping. Default is 10.
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
    

class SNNLoss(nn.Module):
    """
    Soft Nearest Neighbor Loss

    This PyTorch loss function computes the Soft Nearest Neighbor (SNN) loss for a given set of input vectors and their corresponding targets. The SNN loss encourages the similarity between vectors of the same class while discouraging the similarity between vectors of different classes.

    Parameters
    ----------
    use_weights : bool, optional
        If True, calculate class weights based on label frequency (default is True).
    targets : Tensor, optional
        A tensor containing the class labels for the input vectors. Required if use_weights is True.
    batch_keys : list, optional
        A list containing batch keys to account for batch effects (default is None).
    temperature : float, optional
        Initial scaling factor applied to the cosine similarity (default is 0.25).
    min_temperature : float, optional
        The minimum temperature value allowed during optimization (default is 0.1).
    max_temperature : float, optional
        The maximum temperature value allowed during optimization (default is 1.0).
    device : str, optional
        Which device to be used (default is "cuda").

    Attributes
    ----------
    temperature_target : nn.Parameter
        A parameter for the temperature value to be optimized during training.
    temperatures_batches : list
        List of temperature values for different batches.
    min_temperature : float
        The minimum temperature value allowed during optimization.
    max_temperature : float
        The maximum temperature value allowed during optimization.
    use_weights : bool
        Whether to calculate class weights based on label frequency.
    batch_keys : list
        List of batch keys to account for batch effects.
    weight : dict
        Class weights based on label frequency.

    Methods
    -------
    calculate_class_weights(targets)
        Calculate class weights based on label frequency.
    forward(input, targets, batches)
        Compute the SNN loss for the input vectors and targets.
    """

    def __init__(self, 
                 use_weights: bool=True, 
                 targets=None, 
                 batch_keys: list=None, 
                 temperature: float=0.25, 
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 device: str="cuda"):
        super(SNNLoss, self).__init__()
        
        # Define temperature variables to be optimized durring training
        #self.temperature = temperature
        self.temperature_target = nn.Parameter(torch.tensor(temperature), requires_grad=True) 
        if batch_keys is not None:
            self.temperatures_batches = []
            for _ in range(len(batch_keys)):
                #temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True) 
                temperature = 0.5
                self.temperatures_batches.append(temperature)

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.device = device
        self.use_weights = use_weights
        self.batch_keys = batch_keys

        # Calculate weights for the loss based on label frequency
        if self.use_weights:
            if targets is not None:
                self.weight = self.calculate_class_weights(targets)
            else:
                raise ValueError("'use_weights' is True, but 'targets' is not provided.")

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

    def forward(self, input, targets, batches=None):
        """
        Compute the SNN loss for the input vectors and targets.

        Parameters
        ----------
        input : Tensor
            Input vectors.
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

        # Calculate the cosine similarity matrix
        cosine_similarity_matrix = F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperature_target

        # Define a loss dictionary containing the loss of each label
        loss_dict = {str(target): torch.tensor([]).to(self.device) for target in targets.unique()}
        for idx, (sim_vec, target) in enumerate(zip(cosine_similarity_matrix,targets)):
            positiv_samples = sim_vec[(targets == target)]
            negativ_samples = sim_vec[(targets != target)]
            # Must be more or equal to 2 samples per sample type for the loss to work
            if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                positiv_sum = torch.sum(torch.exp(positiv_samples)) - torch.exp(sim_vec[idx])
                negativ_sum = torch.sum(torch.exp(negativ_samples))
                loss = -torch.log(positiv_sum / (positiv_sum + negativ_sum))
                loss_dict[str(target)] = torch.cat((loss_dict[str(target)], loss.unsqueeze(0)))
            else:
                continue

        del cosine_similarity_matrix

        # Calculate the weighted average loss
        weighted_losses = []
        for target in targets.unique():
            losses_for_target = loss_dict[str(target)]
            # Make sure there's values in losses_for_target of given target
            if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                if self.use_weights:
                    weighted_loss = torch.mean(losses_for_target) * self.weight[int(target)]
                else:
                    weighted_loss = torch.mean(losses_for_target)

                weighted_losses.append(weighted_loss)
            else:
                continue

        loss_target = torch.mean(torch.stack(weighted_losses))

        ### Batch loss

        if batches is not None:

            loss_batches = []
            for outer_idx, batch in enumerate(batches):

                # Calculate the cosine similarity matrix
                cosine_similarity_matrix = F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperatures_batches[outer_idx]

                # Define a loss dictionary containing the loss of each label
                loss_dict = {str(target_batch): torch.tensor([]).to(self.device) for target_batch in batch.unique()}
                for idx, (sim_vec, target_batch, target) in enumerate(zip(cosine_similarity_matrix,batch,targets)):
                    positiv_samples = sim_vec[(targets == target) & (batch == target_batch)]
                    negativ_samples = sim_vec[(targets == target) & (batch != target_batch)]
                    # Must be more or equal to 2 samples per sample type for the loss to work
                    if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                        positiv_sum = torch.sum(torch.exp(positiv_samples)) - torch.exp(sim_vec[idx])
                        negativ_sum = torch.sum(torch.exp(negativ_samples))
                        #loss = -torch.log(negativ_sum / (positiv_sum + negativ_sum))
                        loss = (-torch.log(positiv_sum / (positiv_sum + negativ_sum)))**-1
                        loss_dict[str(target_batch)] = torch.cat((loss_dict[str(target_batch)], loss.unsqueeze(0)))
                    else:
                        continue

                losses = []
                for batch_target in batch.unique():
                    losses_for_target = loss_dict[str(batch_target)]
                    # Make sure there's values in losses_for_target of given batch effect
                    if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                        temp_loss = torch.mean(losses_for_target)
                        losses.append(temp_loss)
                    else:
                        continue

                if losses != []:
                    loss_ = torch.mean(torch.stack(losses))
                    loss_batches.append(loss_)

                del cosine_similarity_matrix

            if loss_batches != []:
                loss_batch = torch.mean(torch.stack(loss_batches, dim=0))
            else:
                loss_batch = torch.tensor([0.0]).to(self.device)

            loss = 0.95*loss_target + 0.05*loss_batch

            return loss
        else:
            return loss_target
    

class train_module():
    """
    A class for training the machine learning model using single-cell RNA sequencing data as input.

    Parameters
    ----------
    data_path : str or AnnData
        Path to the data file or an AnnData object containing single-cell RNA sequencing data. If a path is provided,
        the data will be loaded from the specified file. If an AnnData object is provided, it will be used directly.
    json_file_path : str
        Path to the JSON file containing metadata and pathway information.
    num_pathways : int
        The number of pathways in the dataset.
    save_model_path : str
        The path to save the trained model.
    HVG : bool, optional
        Whether to identify highly variable genes (HVGs) in the data (default is True).
    HVGs : int, optional
        The number of highly variable genes to select (default is 4000).
    Scaled : bool, optional
        Whether to scale (normalize) the data before training (default is False).
    target_key : str, optional
        The metadata key specifying the target variable (default is "cell_type").
    batch_keys : list, optional
        List of batch keys to account for batch effects (default is None).

    Methods
    -------
    train(device=None, seed=42, batch_size=256, attn_embed_dim=24*4, depth=2, num_heads=4, output_dim=100,
          attn_drop_out=0.0, proj_drop_out=0.2, drop_ratio=0.2, attn_bias=False, act_layer=nn.ReLU,
          norm_layer=nn.BatchNorm1d, loss_with_weights=True, init_temperature=0.25, min_temperature=0.1,
          max_temperature=1.0, init_lr=0.001, lr_scheduler_warmup=4, lr_scheduler_maxiters=25, eval_freq=2,
          epochs=20, earlystopping_threshold=10, pathway_emb_dim=50)
        Perform cross-validation training on the machine learning model.
    predict(data_, out_path, batch_size=32, device=None)
        Predict the target variable for new data using the trained model.

    """

    def __init__(self, 
                 data_path, 
                 json_file_path: str,
                 num_pathways: int,
                 save_model_path: str,
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 Scaled: bool=False, 
                 target_key: str="cell_type", 
                 batch_keys: list=None):
        
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

        self.data_env = prep_data(adata=self.adata, json_file_path=json_file_path, num_pathways=num_pathways, HVG=HVG, Scaled=Scaled, HVGs=HVGs, target_key=target_key, batch_keys=batch_keys)

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

        # Training loop
        best_val_loss = np.inf  
        train_start = time.time()
        for epoch in tqdm(range(num_epochs)):

            # Training
            model.train()
            train_loss = []
            #acc_grad_count = len(train_loader)
            for data_inputs, data_labels, data_batches, data_pathways in train_loader:

                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)
                data_pathways = data_pathways.to(device)

                optimizer.zero_grad()
                if self.data_env.use_gene2vec_emb:
                    preds = model(data_inputs, data_pathways, self.data_env.gene2vec_tensor)
                else:
                    preds = model(data_inputs, data_pathways)

                if self.batch_keys is not None:
                    data_batches = [batch.to(device) for batch in data_batches]
                    loss = loss_module(preds, data_labels, data_batches) #/ acc_grad_count
                else:
                    loss = loss_module(preds, data_labels)#/ acc_grad_count

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
            #optimizer.step()
            #optimizer.zero_grad()

            # Validation
            if (epoch % eval_freq == 0) or (epoch == (num_epochs-1)):
                model.eval()
                val_loss = []
                all_preds = []
                with torch.no_grad():
                    for data_inputs, data_labels, data_batches, data_pathways in val_loader:

                        data_inputs = data_inputs.to(device)
                        data_labels = data_labels.to(device)
                        data_pathways = data_pathways.to(device)

                        if self.data_env.use_gene2vec_emb:
                            preds = model(data_inputs, data_pathways, self.data_env.gene2vec_tensor)
                        else:
                            preds = model(data_inputs, data_pathways)

                        if self.batch_keys is not None:
                            data_batches = [batch.to(device) for batch in data_batches]
                            loss = loss_module(preds, data_labels, data_batches)
                        else:
                            loss = loss_module(preds, data_labels)

                        val_loss.append(loss.item())
                        all_preds.extend(preds.cpu().detach().numpy())

                # Metrics
                avg_train_loss = sum(train_loss) / len(train_loss)
                avg_val_loss = sum(val_loss) / len(val_loss)

                # Check early stopping
                early_stopping(avg_val_loss)

                # Print epoch information
                print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

                # Update learning rate
                lr_scheduler.step()

                # Apply early stopping
                if early_stopping.early_stop:
                    print(f"Stopped training using EarlyStopping at epoch {epoch+1}")
                    break

                # Save model if performance have improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_preds = all_preds
                    torch.save(model, f'{out_path}model.pt')

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
                 loss_with_weights: bool=True,
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
        device : str or None, optional
            The device to run the training on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
        seed : int, optional
            Random seed for ensuring reproducibility (default is 42).
        batch_size : int, optional
            Batch size for data loading during training (default is 256).
        loss_with_weights : bool, optional
            Whether to use weights in the loss function (default is True).
        init_temperature : float, optional
            Initial temperature for the loss function (default is 0.25).
        min_temperature : float, optional
            The minimum temperature value allowed during optimization (default is 0.1).
        max_temperature : float, optional
            The maximum temperature value allowed during optimization (default is 1.0).
        init_lr : float, optional
            Initial learning rate for the optimizer (default is 0.001).
        lr_scheduler_warmup : int, optional
            Number of warm-up iterations for the learning rate scheduler (default is 4).
        lr_scheduler_maxiters : int, optional
            Maximum number of iterations for the learning rate scheduler (default is 25).
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
        val_loader = data.DataLoader(self.data_env, batch_size=batch_size, shuffle=True)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        loss_module = SNNLoss(use_weights=loss_with_weights, targets=torch.tensor(self.data_env.target), batch_keys=self.batch_keys, temperature=init_temperature, min_temperature=min_temperature, max_temperature=max_temperature)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': init_lr}, {'params': loss_module.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)
        out_path = self.save_model_path

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
    
    
    def predict(self, data_, out_path: str, batch_size: int=32, device: str=None):
        """
        Generate latent represntations for data using the trained model.

        Parameters
        ----------
        data_ : AnnData
            An AnnData object containing data for prediction.
        out_path : str
            The path to the directory where the trained model is saved.
        batch_size : int, optional
            Batch size for data loading during prediction (default is 32).
        device : str or None, optional
            The device to run the prediction on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.

        Returns
        -------
        preds : ndarray
            Array of predicted latent embeddings.
        """

        data_ = prep_test_data(data_, self.data_env)

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = torch.load(f'{out_path}model.pt')
        model.to(device)

        data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

        preds = []
        model.eval()
        with torch.no_grad():
            for data_inputs, data_pathways in data_loader:

                data_inputs = data_inputs.to(device)
                data_pathways = data_pathways.to(device)

                if self.data_env.use_gene2vec_emb:
                    pred = model(data_inputs, data_pathways, self.data_env.gene2vec_tensor)
                else:
                    pred = model(data_inputs, data_pathways)

                preds.extend(pred.cpu().detach().numpy())

        return np.array(preds)

class prep_test_data(data.Dataset):
    """
    PyTorch Dataset for preparing test data for the machine learning model.

    Parameters:
        adata (AnnData): An AnnData object containing single-cell RNA sequencing data.
        prep_data_env: The data environment used for preprocessing training data.

    Attributes:
        X (Tensor): The test data features as a PyTorch tensor.
        pathway_mask (Tensor): Pathway information used in the data.

    Methods:
        __len__(): Returns the number of data samples.
        __getitem__(idx): Retrieves a specific data sample by index.

    """

    def __init__(self, adata, prep_data_env):
        self.adata = adata
        self.adata = self.adata[:, prep_data_env.hvg_genes].copy()
        if prep_data_env.scaled:
            self.adata.X = dp.scale_data(data=self.adata.X, feature_means=prep_data_env.feature_means, feature_stdevs=prep_data_env.feature_stdevs)

        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        self.json_file_path = prep_data_env.json_file_path
        self.use_pathway_buckets = prep_data_env.use_pathway_buckets

        # Pathway information
        if self.json_file_path is not None:
            self.pathway_mask = prep_data_env.pathway_mask
        if (self.json_file_path is not None) and (prep_data_env.use_pathway_buckets == True):

            self.pathway_information = torch.tensor([])
            for idx in range(len(self.X)):
                # Apply pathway mask to HVGs
                data_pathways = torch.where(self.X[idx] * prep_data_env.pathway_mask != 0, torch.tensor(1.0), torch.tensor(0.0))
                data_pathways = torch.sum(data_pathways, dim=1).squeeze() / prep_data_env.num_hvgs
                self.pathway_information = torch.cat((self.pathway_information, data_pathways.unsqueeze(0)))

            # Bucketize levels
            self.pathway_information = self.bucketize_expression_levels(self.pathway_information, prep_data_env.pathways_buckets)

        if prep_data_env.use_HVG_buckets:
            self.X = self.bucketize_expression_levels_per_gene(self.X, prep_data_env.HVG_buckets)  

    def bucketize_expression_levels(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and absolut min/max values.

        Parameters:
            expression_levels: Tensor, shape [samples, num_genes]
            num_buckets: Number of buckets to create

        Returns:
            bucketized_levels: LongTensor, shape [samples, num_genes]
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

         # Generate continuous thresholds
        eps = 1e-6
        thresholds = torch.linspace(torch.min(expression_levels) - eps, torch.max(expression_levels) + eps, steps=num_buckets + 1)

        bucketized_levels = torch.bucketize(expression_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def bucketize_expression_levels_per_gene(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and min/max values of each individual gene.

        Parameters:
            expression_levels: Tensor, shape [samples, num_genes]
            num_buckets: Number of buckets to create

        Returns:
            bucketized_levels: LongTensor, shape [samples, num_genes]
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

        # Generate continuous thresholds
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

    def __len__(self):
        """
        Get the number of data samples in the dataset.

        Returns:
        int: The number of data samples.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get a specific data sample by index.

        Parameters:
        idx (int): Index of the data sample to retrieve.

        Returns:
        tuple: A tuple containing the data point and pathways.
        """

        data_point = self.X[idx]

        if self.json_file_path is not None:
            if self.use_pathway_buckets:
                data_pathways = self.pathway_information[idx]
            else:
                data_pathways = self.X[idx] * self.pathway_mask
        else:
            data_pathways = torch.tensor([])

        return data_point, data_pathways

