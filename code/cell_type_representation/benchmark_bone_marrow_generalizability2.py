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
import random
from benchmarks.benchmark_generalizability2 import benchmark as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd Masters_Thesis/code/cell_type_representation/
# Then:
# sbatch jobscript_generalizability_bone_marrow2.sh


def main(data_path: str, model_path: str, result_csv_path: str, pathway_path: str, gene2vec_path: str):
    """
    Execute the generalizability benchmark pipeline. Selects 20% of data for testing and uses the
    remaining 80% for training, but at different amounts (20%, 40%, 60%, 80%).

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - result_csv_path (str): File path to save the benchmark results as a CSV file.
    - pathway_path (str): File path to the pathway information.
    - gene2vec_path (str): File path to the gene2vec embeddings.

    Returns:
    None

    Note:
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): 
    """
    
    # Calculate for model at different number of patient for training and different random seeds
    num_patients_for_training_list = [4,8,12,16]#[4,8,12,16]
    num_patients_for_training_full_list = [4,8,12,16]
    list_of_data_pct = [0.2, 0.4, 0.6, 0.8]
    r_seeds = [42]#[42,43,44,45,46]
    counter = 0
    for idx, train_num in enumerate(num_patients_for_training_list):
        
        for seed in r_seeds:
            counter += 1

            while True:  # Keep trying new seeds until no error occurs
                try:
                    print("seed: ", seed)

                    benchmark_env = benchmark(data_path=data_path, 
                                            pathway_path=pathway_path,
                                            gene2vec_path=gene2vec_path,
                                            image_path=f'train_num_{train_num}_seed_{seed}_',
                                            batch_key="patientID", 
                                            HVG=True, 
                                            HVGs=2000, 
                                            num_patients_for_testing=4,
                                            num_patients_for_training=train_num,
                                            Scaled=False, 
                                            seed=42,
                                            select_patients_seed=seed)
                    
                    # Calculate for unintegrated and PCA
                    if train_num == num_patients_for_training_full_list[0]:
                        print("Start evaluating unintegrated data")
                        print()
                        benchmark_env.unintegrated(save_figure=False, umap_plot=False)

                        print("Start evaluating PCA transformed data")
                        print()
                        benchmark_env.pca(save_figure=False, umap_plot=False)

                    # Calculate for model
                    print(f"Start training model with {train_num} patients and seed {seed}")
                    print()

                    benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Generalizability2/train_num_{train_num}_seed_{seed}_', train=True, umap_plot=False, save_figure=False)
                    #benchmark_env.in_house_model_tokenized_HVG_transformer(save_path=f'{model_path}Tokenized_HVG_Transformer/Generalizability2/train_num_{train_num}_seed_{seed}_', train=True, umap_plot=False, save_figure=False)

                    benchmark_env.make_benchamrk_results_dataframe(counter="", min_max_normalize=False)

                    benchmark_env.metrics["train_num"] = [list_of_data_pct[idx]]*benchmark_env.metrics.shape[0]
                    benchmark_env.metrics["seed"] = [seed]*benchmark_env.metrics.shape[0]

                    if train_num == num_patients_for_training_full_list[0]:
                        num_replicates = len(num_patients_for_training_full_list) - 1
                        replicated_unintegrated = pd.concat([benchmark_env.metrics[benchmark_env.metrics.index == 'Unintegrated']] * num_replicates, ignore_index=False, axis="rows")
                        replicated_pca = pd.concat([benchmark_env.metrics[benchmark_env.metrics.index == 'PCA']] * num_replicates, ignore_index=False, axis="rows")
                        benchmark_env.metrics = pd.concat([benchmark_env.metrics, replicated_unintegrated, replicated_pca], ignore_index=False, axis="rows")

                        benchmark_env.metrics['train_num'][benchmark_env.metrics.index == 'Unintegrated'] = list_of_data_pct
                        benchmark_env.metrics['train_num'][benchmark_env.metrics.index == 'PCA'] = list_of_data_pct

                    #if counter > 1:
                    #    benchmark_env.read_csv(name=result_csv_path)
                    benchmark_env.read_csv(name=result_csv_path)

                    benchmark_env.save_results_as_csv(name=result_csv_path)

                    del benchmark_env

                    # Empty the cache
                    torch.cuda.empty_cache()

                    break
                except Exception as e:
                    # Handle the exception (you can print or log the error if needed)
                    print(f"Error occurred: {e}")

                    # Generate a new random seed not in random_seeds list
                    while True:
                        new_seed = random.randint(1, 10000)
                        if new_seed not in r_seeds:
                            break

                    print(f"Trying a new random seed: {new_seed}")
                    seed = new_seed

    print("Finished generalizability benchmark!")
        
if __name__ == "__main__":
    """
    Start with: cd .\code\cell_type_representation\
    How to run example (on bone marrow data set): 
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('result_csv_path', type=str, help='Path to save the benchmark results as a CSV file.')
    parser.add_argument('pathway_path', type=str, help='Path to the pathway information json file.')
    parser.add_argument('gene2vec_path', type=str, help='Path to gene2vec representations.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.pathway_path, args.gene2vec_path)