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


def main(data_path: str, model_path: str, result_csv_path: str, image_path: str, dataset_name: str):
    """
    Execute the code for investigating novel cell type detection of model1. 
    Here you select a cell type to be left out during training. Then we calculate metrics
    on the models novel predictions, and on non-novel predictions separately.
    Selects 20% of data for testing and uses the remaining 80% for training. Performs 5-fold cross testing.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - result_csv_path (str): File path to save the benchmark results as a CSV file.
    - image_path (str): Path where images will be saved.
    - dataset_name (str): Name of dataset.

    Returns:
    None 
    """
    
    # Calculate for model at different number of patient for training and different random seeds
    folds = [1,2,3,4,5]
    num_folds = 5
    counter = 0  
    exclude_cell_types_list = [['Mature_B_Cells'], 
                                ['Plasma_Cells'], 
                                ['alpha-beta_T_Cells'], 
                                ['gamma-delta_T_Cells_1'],
                                ['Mature_B_Cells', 
                                'Plasma_Cells', 
                                'alpha-beta_T_Cells', 
                                'gamma-delta_T_Cells_1']]

    exclude_cell_types_list_names = ['Mature_B_Cells', 
                                'Plasma_Cells', 
                                'alpha-beta_T_Cells', 
                                'gamma-delta_T_Cells_1',
                                'All_Above']

    threshold_list = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.1]

    novel_cell_counter = -1
    for novel_cell in exclude_cell_types_list:
        novel_cell_counter += 1

        for fold in folds:
            counter2 = 0
            for threshold in threshold_list:
                counter += 1
                counter2 += 1

                seed = 42

                print("fold: ", fold)
                print("seed: ", seed)

                benchmark_env = benchmark(data_path=data_path,
                                        exclude_cell_types = novel_cell,
                                        dataset_name=dataset_name,
                                        image_path=f"{image_path}/{novel_cell}/{fold}/{threshold}/",
                                        HVGs=2000,
                                        fold=fold,
                                        seed=seed)

                # Calculate for model
                print(f"Start training model, fold {fold} and seed {seed}")
                print()
                if counter2 == 1:
                    benchmark_env.Model1_classifier(threshold=threshold, save_path=f'{model_path}{exclude_cell_types_list_names[novel_cell_counter]}/Model1/', excluded_cell = exclude_cell_types_list_names[novel_cell_counter], train=True, umap_plot=False, save_figure=False)
                else:
                    benchmark_env.Model1_classifier(threshold=threshold, save_path=f'{model_path}{exclude_cell_types_list_names[novel_cell_counter]}/Model1/', excluded_cell = exclude_cell_types_list_names[novel_cell_counter], train=False, umap_plot=False, save_figure=False)

                benchmark_env.make_benchamrk_results_dataframe()

                #if counter > 1:
                #    benchmark_env.read_csv(name=result_csv_path)
                benchmark_env.read_csv(name=result_csv_path)

                benchmark_env.save_results_as_csv(name=result_csv_path)

                del benchmark_env

                # Empty the cache
                torch.cuda.empty_cache()

    print("Finished generalizability benchmark!")
        
if __name__ == "__main__":
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