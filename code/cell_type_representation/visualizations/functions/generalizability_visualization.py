
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import scanpy as sc
import torch.nn as nn
import scib
from functions import train as trainer
from benchmarks.benchmark_generalizability import benchmark as benchmark
from benchmarks.benchmark_generalizability_with_validation import benchmark as benchmark_with_validation
from benchmarks.benchmark_generalizability2 import benchmark as benchmark2
from benchmarks.benchmark_generalizability_with_validation2 import benchmark as benchmark_with_validation2

def generalizability_error_bar_plot(csv_path: str, image_path: str, x_axis_title: str='Nr. of Patients for Training', scale: float=0.1):
    """
    Generates an error bar plot to display generalizability based on performance metrics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing performance metrics.
    image_path : str
        Directory path to save the error bar plot.

    Returns
    -------
    None
    """
    
    # Make a suitable plot to display generalizability
    metrics = pd.read_csv(f'{csv_path}.csv', index_col=0)

    #metrics['Model Type'] = [re.sub(r'\d+$', '', model_string) for model_string in metrics.index]
    metrics['Model Type'] = metrics.index
    # Replace model names
    metrics.loc[metrics['Model Type'] == "In-house HVG Encoder Model ", 'Model Type'] = "Model 1"
    metrics.loc[metrics['Model Type'] == "In-house Tokenized HVG Transformer Encoder Model", 'Model Type'] = "Model 2"
    metrics.loc[metrics['Model Type'] == "In-house Tokenized HVG Transformer Encoder with HVG Encoder", 'Model Type'] = "Model 3"
    metrics.loc[metrics['Model Type'] == "In-house Tokenized HVG Transformer Encoder with Pathways Model	", 'Model Type'] = "Model 4"

    # Group by train_num and model type, calculate mean and std
    grouped_df = metrics.groupby(['train_num', 'Model Type'])['Overall'].agg(['mean', 'std']).reset_index()

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(grouped_df['Model Type'].unique()))

    # Create a dictionary to map model types to unique colors
    color_dict = dict(zip(grouped_df['Model Type'].unique(), [cmap(i) for i in range(len(grouped_df['Model Type'].unique()))]))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot all model types in the same plot
    for model_type, color in color_dict.items():
        model_df = grouped_df[grouped_df['Model Type'].str.contains(model_type)]

        # Add jitter to x-coordinates for each individual point
        jittered_x = model_df['train_num'] + np.random.normal(scale=scale, size=len(model_df))
        jittered_x.reset_index(drop=True, inplace=True) 

        # Plot each data point separately
        for i in range(len(model_df)):
            plt.errorbar(
                jittered_x[i],
                model_df['mean'].iloc[i],
                yerr=model_df['std'].iloc[i],
                fmt='o',  # Use 'o' for markers only, without lines
                linestyle='',
                label=model_type if i == 0 else "",  # Label only the first point for each model type
                color=color,
                markersize=8,
                capsize=5,
                capthick=2,
                alpha=1.0,
                linewidth=2,  # Set linewidth to 0 for markers only
            )

    # Set xticks to only include the desired values
    plt.xticks(model_df['train_num'].unique())

    plt.xlabel(x_axis_title)
    plt.ylabel('Overall Metric')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title('')

    # Turn off grid lines
    plt.grid(False)

    # Adjust layout to ensure the x-axis label is not cut off
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(f'{image_path}.svg', format='svg')

    plt.show()


def UMAPLatentSpace(train_num: int=16, seed: int=42, model_path: str='../trained_models/Assess_generalisability/'):

    benchmark_env = benchmark(data_path='../../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad', 
                                    pathway_path='../../../data/processed/pathway_information/c5_pathways.json',
                                    gene2vec_path='../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt',
                                    image_path=f'_train_num_{train_num}_seed_{seed}_',
                                    batch_key="patientID", 
                                    HVG=True, 
                                    HVGs=2000, 
                                    num_patients_for_testing=4,
                                    num_patients_for_training=train_num,
                                    Scaled=False, 
                                    seed=42,
                                    select_patients_seed=seed)
    
    benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Generalizability/train_num_{train_num}_seed_{seed}_', train=False, umap_plot=True, save_figure=True)
    
    del benchmark_env

def UMAPLatentSpace_with_validation(train_num: int=16, seed: int=42, model_path: str = '../trained_models/Assess_generalisability/'):

    benchmark_env = benchmark_with_validation(data_path='../../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad', 
                                            pathway_path='../../../data/processed/pathway_information/c5_pathways.json',
                                            gene2vec_path='../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt',
                                            image_path=f'_train_num_{train_num}_seed_{seed}_',
                                            batch_key="patientID", 
                                            HVG=True, 
                                            HVGs=2000, 
                                            num_patients_for_testing=4,
                                            num_patients_for_training=train_num,
                                            Scaled=False, 
                                            seed=42,
                                            select_patients_seed=seed)
    
    benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Generalizability_with_validation/train_num_{train_num}_seed_{seed}_', train=False, umap_plot=True, save_figure=True)
    
    del benchmark_env

def UMAPLatentSpace2(train_num: int=16, seed: int=42, model_path: str='../trained_models/Assess_generalisability/'):

    benchmark_env = benchmark2(data_path='../../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad', 
                                    pathway_path='../../../data/processed/pathway_information/c5_pathways.json',
                                    gene2vec_path='../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt',
                                    image_path=f'_train_num_{train_num}_seed_{seed}_',
                                    batch_key="patientID", 
                                    HVG=True, 
                                    HVGs=2000, 
                                    num_patients_for_testing=4,
                                    num_patients_for_training=train_num,
                                    Scaled=False, 
                                    seed=42,
                                    select_patients_seed=seed)
    
    benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Generalizability2/train_num_{train_num}_seed_{seed}_', train=False, umap_plot=True, save_figure=True)
    
    del benchmark_env

def UMAPLatentSpace_with_validation2(train_num: int=16, seed: int=42, model_path: str = '../trained_models/Assess_generalisability/'):

    benchmark_env = benchmark_with_validation2(data_path='../../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad', 
                                            pathway_path='../../../data/processed/pathway_information/c5_pathways.json',
                                            gene2vec_path='../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt',
                                            image_path=f'_train_num_{train_num}_seed_{seed}_',
                                            batch_key="patientID", 
                                            HVG=True, 
                                            HVGs=2000, 
                                            num_patients_for_testing=4,
                                            num_patients_for_training=train_num,
                                            Scaled=False, 
                                            seed=42,
                                            select_patients_seed=seed)
    
    benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Generalizability_with_validation2/train_num_{train_num}_seed_{seed}_', train=False, umap_plot=True, save_figure=True)
    
    del benchmark_env
    