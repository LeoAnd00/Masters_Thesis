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
from benchmark_classifiers2 import classifier_train as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd Masters_Thesis/code/novel_cell_type_detection/
# Then:
# sbatch jobscript_annotation_macparland2.sh


# sbatch jobscript_annotation_segerstolpe.sh
# sbatch jobscript_annotation_baron.sh
# sbatch jobscript_annotation_zheng68k.sh


def main(novel_cell: str,
         novel_cell_name: str,
         fold: int,
         threshold: float,
         data_path: str, 
         model_path: str, 
         result_csv_path: str, 
         image_path: str, 
         dataset_name: str):
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

    seed = 42

    benchmark_env = benchmark(data_path=data_path,
                            exclude_cell_types = novel_cell,
                            dataset_name=dataset_name,
                            image_path=f"{image_path}/{novel_cell_name}/Fold_{fold}/Threshold_{threshold}/",
                            HVGs=2000,
                            fold=fold,
                            seed=seed)

    #print("**Start benchmarking TOSICA method**")
    #benchmark_env.tosica(excluded_cell = exclude_cell_types_list_names[novel_cell_counter], threshold=threshold)

    # Calculate for model
    print(f"Start evaluating model, fold {fold} and seed {seed}")

    benchmark_env.Model1_classifier(threshold=threshold, save_path=f'{model_path}{novel_cell_name}/Model1/', excluded_cell = novel_cell_name, train=False, umap_plot=True, save_figure=True)

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
    parser.add_argument('dataset_name', type=str, help='Name of dataset.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.image_path, args.dataset_name)