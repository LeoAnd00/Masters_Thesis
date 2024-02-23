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
from benchmark_classifiers import classifier_train as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd Masters_Thesis/code/cell_type_representation/
# Then:
# sbatch jobscript_annotation_MacParland.sh
# sbatch jobscript_annotation_Segerstolped.sh
# sbatch jobscript_annotation_Baron.sh


def main(data_path: str, model_path: str, result_csv_path: str, image_path: str):
    """
    Execute the annotation generalizability benchmark pipeline. Selects 20% of data for testing and uses the
    remaining 80% for training. Performs 5-fold cross testing.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - result_csv_path (str): File path to save the benchmark results as a CSV file.
    - pathway_path (str): File path to the pathway information.
    - gene2vec_path (str): File path to the gene2vec embeddings.
    - image_path (str): Path where images will be saved.

    Returns:
    None 
    """
    
    # Calculate for model at different number of patient for training and different random seeds
    folds = [1]
    num_folds = 5
    seed = 42
    counter = 0  
    for fold in folds:
        counter += 1

        while True:  # Keep trying new seeds until no error occurs
            try:
                print("fold: ", fold)
                print("seed: ", seed)

                benchmark_env = benchmark(data_path=data_path,
                                          image_path=image_path,
                                          HVGs=2000,
                                          fold=1,
                                          seed=seed)

                print("**Start benchmarking TOSICA method**")
                benchmark_env.tosica(umap_plot=False,save_figure=False)

                # Calculate for model
                print(f"Start training model, fold {fold} and seed {seed}")
                print()
                if fold == 1:
                    benchmark_env.Model1_benchmark(save_path=f'{model_path}Model1/', train=True, umap_plot=False, save_figure=False)
                    #benchmark_env.Model3_benchmark(save_path=f'{model_path}Model3/', train=True, umap_plot=False, save_figure=True)
                    #benchmark_env.Model2_benchmark(save_path=f'{model_path}Model2/', train=True, umap_plot=False, save_figure=False)
                else:
                    benchmark_env.Model1_benchmark(save_path=f'{model_path}Model1/', train=True, umap_plot=False, save_figure=False)
                    #benchmark_env.Model3_benchmark(save_path=f'{model_path}Model3/', train=True, umap_plot=False, save_figure=False)
                
                benchmark_env.make_benchamrk_results_dataframe()

                benchmark_env.metrics["train_pct"] = [list_of_data_pct[idx]]*benchmark_env.metrics.shape[0]
                benchmark_env.metrics["seed"] = [seed]*benchmark_env.metrics.shape[0]
                benchmark_env.metrics["fold"] = [fold]*benchmark_env.metrics.shape[0]

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
                new_seed = random.randint(1, 10000)

                print(f"Trying a new random seed: {new_seed}")
                seed = new_seed

                break

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
    parser.add_argument('image_path', type=str, help='Path where images will be saved.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.image_path)