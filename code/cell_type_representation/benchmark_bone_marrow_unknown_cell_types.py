# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import warnings
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import torch
import torch.nn as nn
from functions import train as trainer
from models import model_tokenized_hvg_transformer_with_pathways as model_tokenized_hvg_transformer_with_pathways
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def in_house_model_tokenized_HVG_transformer_with_pathways(adata_in_house, adata_in_house_predict, pathway_path: str, label_key: str, gene2vec_path: str, image_path: str, save_path: str, umap_plot: bool=True, train: bool=True, save_figure: bool=False):
    """
    Perform cell type prediction using a tokenized HVG transformer with pathways.
    It starts by training the model on training data and then predicts on data to be predicted.

    Parameters:
    - adata_in_house (AnnData): The training dataset containing expression data and associated metadata.
    - adata_in_house_predict (AnnData): The dataset to predict cell types for, containing expression data.
    - pathway_path (str): The file path to the pathway information.
    - label_key (str): Key in the metadata of `adata_in_house` containing cell type labels.
    - gene2vec_path (str): The file path to the gene2vec embeddings.
    - image_path (str): The directory path to save UMAP plots if `save_figure` is True.
    - save_path (str): The directory path to save the trained model and predictions.
    - umap_plot (bool, optional): Whether to generate and display UMAP plots. Default is True.
    - train (bool, optional): Whether to train the model. Default is True.
    - save_figure (bool, optional): Whether to save UMAP plots as SVG files. Default is False.

    Returns:
    None
    """

    HVG_buckets_ = 1000
    HVGs_num = 2000
    seed = 42

    train_env = trainer.train_module(data_path=adata_in_house,
                                    pathways_file_path=pathway_path,
                                    num_pathways=300,
                                    pathway_gene_limit=10,
                                    save_model_path=save_path,
                                    HVG=True,
                                    HVGs=HVGs_num,
                                    HVG_buckets=HVG_buckets_,
                                    use_HVG_buckets=True,
                                    Scaled=False,
                                    target_key=label_key,
                                    batch_keys=["batch"],
                                    use_gene2vec_emb=True,
                                    gene2vec_path=gene2vec_path)
    
    #Model
    model = model_tokenized_hvg_transformer_with_pathways.CellType2VecModel(input_dim=min([HVGs_num,int(train_env.data_env.X.shape[1])]),
                                                                output_dim=100,
                                                                num_pathways=300,
                                                                drop_out=0.2,
                                                                act_layer=nn.ReLU,
                                                                norm_layer=nn.LayerNorm,#nn.BatchNorm1d,
                                                                attn_embed_dim=24*4,
                                                                num_heads=4,
                                                                mlp_ratio=4,
                                                                attn_bias=False,
                                                                attn_drop_out=0.,
                                                                depth=3,
                                                                pathway_embedding_dim=50,
                                                                nn_tokens=HVG_buckets_,
                                                                nn_embedding_dim=train_env.data_env.gene2vec_tensor.shape[1],
                                                                use_gene2vec_emb=True)
                
    # Train
    if train:
        _ = train_env.train(model=model,
                            device=None,
                            seed=seed,
                            batch_size=256,
                            batch_size_step_size=256,
                            use_target_weights=True,
                            use_batch_weights=True,
                            init_temperature=0.25,
                            min_temperature=0.1,
                            max_temperature=2.0,
                            init_lr=0.001,
                            lr_scheduler_warmup=4,
                            lr_scheduler_maxiters=110,#25,
                            eval_freq=1,
                            epochs=100,#20,
                            earlystopping_threshold=40)#5)
    
    predictions = train_env.predict(data_=adata_in_house_predict, model=model, model_path=save_path)
    adata_in_house_predict.obsm["In_house"] = predictions

    del predictions
    sc.pp.neighbors(adata_in_house_predict, use_rep="In_house")

    random_order = np.random.permutation(adata_in_house_predict.n_obs)
    adata_in_house_predict = adata_in_house_predict[random_order, :]

    if umap_plot:
        sc.tl.umap(adata_in_house_predict)
        sc.pl.umap(adata_in_house_predict, color=label_key, ncols=1, title="Cell type")
        sc.pl.umap(adata_in_house_predict, color="batch", ncols=1, title="Batch effect")
    if save_figure:
        sc.tl.umap(adata_in_house_predict)
        sc.pl.umap(adata_in_house_predict, color=label_key, ncols=1, title="Cell type", show=False, save=f"{image_path}InHouse_Tokenized_HVG_Transformer_Encoder_with_Pathways_Model_cell_type.svg")
        sc.pl.umap(adata_in_house_predict, color="batch", ncols=1, title="Batch effect", show=False, save=f"{image_path}InHouse_Tokenized_HVG_Transformer_Encoder_with_Pathways_Model_batch_effect.svg")

    del adata_in_house, adata_in_house_predict

def main(data_path: str, model_path: str, image_path: str, pathway_path: str, gene2vec_path: str, batch_key: str, label_key: str):
    """
    Execute the pipeline for training and evaluating a cell type prediction model. 
    Finds batch effects containing cell types that only exists for that batch effect. 
    These samples are used for prediction to show whether the model identifies these cell types
    even though it wasn't used during training.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - image_path (str): Directory path to save UMAP plots if 'save_figure' is True.
    - pathway_path (str): File path to the pathway information.
    - gene2vec_path (str): File path to the gene2vec embeddings.
    - batch_key (str): Key in the metadata of 'adata' containing batch information.
    - label_key (str): Key in the metadata of 'adata' containing cell type labels.

    Returns:
    None

    Note:
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): python benchmark_bone_marrow_unknown_cell_types.py '../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'trained_models/Assess_unknown_cells/' '_Assess_unknown_cells' '../../data/processed/pathway_information/all_pathways.json' '../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' 'patientID' 'cell_type'
    """
    
    adata = sc.read(data_path, cache=True)

    adata.obs["batch"] = adata.obs[batch_key]

    # Step 1: Calculate the count of each cell type for each batch
    cell_type_counts = adata.obs.groupby(["batch", label_key]).size().reset_index(name="count")

    # Step 2: Count the number of batches that contain each cell type
    unique_cell_type_batch_dict = {}
    unique_cell_type_batch_counts_dict = {}
    count_threshold = 50
    for cell_type in cell_type_counts.cell_type.unique():
        batch_effects = []
        all_counts = []
        for batch in cell_type_counts.batch.unique():
            counts = cell_type_counts.loc[(cell_type_counts[label_key] == cell_type) & (cell_type_counts['batch'] == batch), 'count']
            counts = int(counts)
            if counts > count_threshold:
                batch_effects.append(batch)
                all_counts.append(counts)
        if len(batch_effects) == 1:
            unique_cell_type_batch_dict[cell_type] = batch_effects[0]
            unique_cell_type_batch_counts_dict[cell_type] = all_counts[0]

    # Step 3: Create a mask to identify rows for training and testing
    train_mask = ~adata.obs[batch_key].isin(unique_cell_type_batch_dict.values())
    test_mask = adata.obs[batch_key].isin(unique_cell_type_batch_dict.values())

    # Step 4: Split the data into training and testing sets
    train_data = adata[train_mask]
    test_data = adata[test_mask]

    # Train
    in_house_model_tokenized_HVG_transformer_with_pathways(adata_in_house=train_data, adata_in_house_predict=test_data, pathway_path=pathway_path, label_key=label_key, gene2vec_path=gene2vec_path, image_path=image_path, save_path=model_path, umap_plot=False, train=True, save_figure=True)
        

if __name__ == "__main__":
    """
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): python benchmark_bone_marrow_unknown_cell_types.py '../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'trained_models/Assess_unknown_cells/' '_Assess_unknown_cells' '../../data/processed/pathway_information/all_pathways.json' '../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' 'patientID' 'cell_type'
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('image_path', type=str, help='Path to save the UMAP image.')
    parser.add_argument('pathway_path', type=str, help='Path to the pathway information json file.')
    parser.add_argument('gene2vec_path', type=str, help='Path to gene2vec representations.')
    parser.add_argument('batch_key', type=str, help='Key name representing the batch effect.')
    parser.add_argument('label_key', type=str, help='Key name representing the label (cell type).')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.image_path, args.pathway_path, args.gene2vec_path, args.batch_key, args.label_key)