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
from benchmarks.benchmark_generalizability_with_validation import benchmark as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(data_path: str, model: str, model_path: str, result_csv_path: str, pathway_path: str, gene2vec_path: str, image_path: str):
    """
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): python benchmark_bone_marrow_generalizability_with_validation.py '../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'Encoder' 'trained_models/Assess_generalisability/' 'benchmarks/results/Generalizability_with_validation/Benchmark_results' '../../data/processed/pathway_information/all_pathways.json' '../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_generalizability_with_validation/'
    """
    
    # Calculate for model at different number of patient for training and different random seeds
    num_patients_for_training_list = [4,8,12,16]
    r_seeds = [42,43,44,45,46,47]
    counter = 0
    for train_num in num_patients_for_training_list:
        
        for seed in r_seeds:
            counter += 1

            benchmark_env = benchmark(data_path=data_path, 
                                    pathway_path=pathway_path,
                                    gene2vec_path=gene2vec_path,
                                    image_path=f'{image_path}train_num_{train_num}_seed_{seed}_',
                                    batch_key="patientID", 
                                    HVG=True, 
                                    HVGs=4000, 
                                    num_patients_for_testing=4,
                                    num_patients_for_training=train_num,
                                    Scaled=False, 
                                    seed=42,
                                    select_patients_seed=seed)
            
            # Calculate for unintegrated and PCA
            if train_num == num_patients_for_training_list[0]:
                print("Start evaluating unintegrated data")
                print()
                benchmark_env.unintegrated(save_figure=False, umap_plot=False)

                print("Start evaluating PCA transformed data")
                print()
                benchmark_env.pca(save_figure=False, umap_plot=False)
            
            # Calculate for model
            if model == 'Encoder':
                
                print(f"Start training model with {train_num} patients and seed {seed}")
                print()

                benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/train_num_{train_num}_seed_{seed}_', train=True, umap_plot=False, save_figure=False)

                benchmark_env.make_benchamrk_results_dataframe(counter=f"with 20% validation set", min_max_normalize=False)

                benchmark_env.metrics["train_num"] = [train_num]*benchmark_env.metrics.shape[0]
                benchmark_env.metrics["seed"] = [seed]*benchmark_env.metrics.shape[0]

                if train_num == num_patients_for_training_list[0]:
                    num_replicates = len(num_patients_for_training_list) - 1
                    replicated_unintegrated = pd.concat([benchmark_env.metrics[benchmark_env.metrics.index == 'Unintegrated']] * num_replicates, ignore_index=False, axis="rows")
                    replicated_pca = pd.concat([benchmark_env.metrics[benchmark_env.metrics.index == 'PCA']] * num_replicates, ignore_index=False, axis="rows")
                    benchmark_env.metrics = pd.concat([benchmark_env.metrics, replicated_unintegrated, replicated_pca], ignore_index=False, axis="rows")

                    benchmark_env.metrics['train_num'][benchmark_env.metrics.index == 'Unintegrated'] = num_patients_for_training_list
                    benchmark_env.metrics['train_num'][benchmark_env.metrics.index == 'PCA'] = num_patients_for_training_list

                if counter > 1:
                    benchmark_env.read_csv(name=result_csv_path)

                benchmark_env.save_results_as_csv(name=result_csv_path)

                del benchmark_env

                # Empty the cache
                torch.cuda.empty_cache()

    # Make a suitable plot to display generalizability
    metrics = pd.read_csv(f'{result_csv_path}.csv', index_col=0)

    metrics['Model Type'] = metrics.index#[re.sub(r'\d+$', '', model_string) for model_string in metrics.index]

    # Group by train_num and model type, calculate mean and std
    grouped_df = metrics.groupby(['train_num', 'Model Type'])['Overall'].agg(['mean', 'std']).reset_index()

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(grouped_df['Model Type'].unique()))

    # Create a dictionary to map model types to unique colors
    color_dict = dict(zip(grouped_df['Model Type'].unique(), [cmap(i) for i in range(len(grouped_df['Model Type'].unique()))]))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(20, 9))

    # Plot all model types in the same plot
    for model_type, color in color_dict.items():
        model_df = grouped_df[grouped_df['Model Type'].str.contains(model_type)]
        plt.errorbar(
            model_df['train_num'],
            model_df['mean'],
            yerr=model_df['std'],
            fmt='-o',
            label=model_type,
            color=color,
            markersize=8,
            marker='s',
            capsize=5,  # Adjust the length of the horizontal lines at the top and bottom of the error bars
            capthick=2,      # Adjust the thickness of the horizontal lines
            alpha=0.5,        # Adjust the transparency of the plot
            linewidth=2      # Adjust the thickness of the line connecting the points
        )
    plt.xlabel('Nr. of Patients for Training')
    plt.ylabel('Overall Metric')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title('Generalizability Assessment')

    # Turn off grid lines
    plt.grid(False)

    # Adjust layout to ensure the x-axis label is not cut off
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(f'figures/umap{image_path}Bone_marrow_generalizability.svg', format='svg')

    #plt.show()

    print("Finished generalizability benchmark!")
        
if __name__ == "__main__":
    """
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): python benchmark_bone_marrow_generalizability_with_validation.py '../../data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'Encoder' 'trained_models/Assess_generalisability/' 'benchmarks/results/Generalizability_with_validation/Benchmark_results' '../../data/processed/pathway_information/all_pathways.json' '../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_generalizability_with_validation/'
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model', type=str, help='Which model to use (Options: Encoder)')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('result_csv_path', type=str, help='Path to save the benchmark results as a CSV file.')
    parser.add_argument('pathway_path', type=str, help='Path to the pathway information json file.')
    parser.add_argument('gene2vec_path', type=str, help='Path to gene2vec representations.')
    parser.add_argument('image_path', type=str, help='Path where images will be saved.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model, args.model_path, args.result_csv_path, args.pathway_path, args.gene2vec_path, args.image_path)