# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import re
import torch
import random
from novel_cell_type_classifier import classifier_train as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class novel_cell_type_detection():

    def __init__(self):
        pass

    def main(self, 
            data_path: str, 
            model_path: str, 
            result_csv_path: str, 
            image_path: str, 
            dataset_names: str):
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

        fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(8*len(dataset_names), 6))
        if len(dataset_names) == 1:
            axes = [axes]

        model_path_core = model_path
        data_path_core = data_path
        
        colors = sns.color_palette('deep', 2)
        self.confidence_dict = {}
        for idx, dataset_name in enumerate(dataset_names):

            self.confidence_dict[dataset_name] = {}

            print(f"Start working with dataset: {dataset_name}")

            model_path = f"{model_path_core}{dataset_name}/"
            data_path = f"{data_path_core}/{dataset_name}.h5ad"

            novel_cells, novel_cell_names = self.get_cell_type_names(dataset_name)

            min_non_novel_confidence = []
            min_novel_confidence = []

            for novel_cell, novel_cell_name in zip(novel_cells, novel_cell_names):

                print(f"Start working with removing cell type: {novel_cell_name}")

                for fold in range(5):
                    fold += 1

                    seed = 42

                    benchmark_env = benchmark(data_path=data_path,
                                            exclude_cell_types = novel_cell,
                                            dataset_name=dataset_name,
                                            image_path=f"{image_path}/",
                                            HVGs=2000,
                                            HVG=False,
                                            fold=fold,
                                            seed=seed)
                    
                    # Calculate for model
                    min_non_novel_confidence_temp, min_novel_confidence_temp = benchmark_env.threshold_investigation(save_path=f'{model_path}{novel_cell_name}/Model1/', train=False)
                    
                    min_non_novel_confidence.append(min_non_novel_confidence_temp)
                    min_novel_confidence.append(min_novel_confidence_temp)

                    del benchmark_env


            self.confidence_dict[dataset_name]["min_non_novel_confidence"] = min_non_novel_confidence
            self.confidence_dict[dataset_name]["min_novel_confidence"] = min_novel_confidence

            # Concatenate max_confidence and min_confidence lists
            all_confidence = min_non_novel_confidence + min_novel_confidence

            # Create a list of labels (0 for max_confidence, 1 for min_confidence)
            labels = ['Min Confidence of Non-Novel Cell Type'] * len(min_non_novel_confidence) + ['Min Confidence of Novel Cell Type'] * len(min_novel_confidence)
            
            # Jitter plot
            sns.stripplot(x=labels, y=all_confidence, jitter=True, palette=colors, alpha=0.7, ax=axes[idx])
            axes[idx].set_title(dataset_name, fontsize=14)
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Confidence', fontsize=14)
            axes[idx].grid(axis='y', linewidth=1.5)

            axes[idx].tick_params(axis='x', which='major', labelsize=10) 
            axes[idx].tick_params(axis='y', which='major', labelsize=10) 

        #plt.tight_layout()
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')
        plt.show()

    def get_cell_type_names(self, datset_name: str):
        if datset_name == "MacParland":
            exclude_cell_types_list = [['Mature_B_Cells'], 
                                ['Plasma_Cells'], 
                                ['alpha-beta_T_Cells'], 
                                ['gamma-delta_T_Cells_1'],
                                ['gamma-delta_T_Cells_2'],
                                ['Central_venous_LSECs'], 
                                ['Cholangiocytes'], 
                                ['Non-inflammatory_Macrophage'], 
                                ['Inflammatory_Macrophage'],
                                ['Erythroid_Cells'], 
                                ['Hepatic_Stellate_Cells'],
                                ['NK-like_Cells'],
                                ['Periportal_LSECs'],
                                ['Portal_endothelial_Cells'],
                                ['Hepatocyte_1'],
                                ['Hepatocyte_2'],
                                ['Hepatocyte_3'],
                                ['Hepatocyte_4'],
                                ['Hepatocyte_5'],
                                ['Hepatocyte_6']]
        elif datset_name == "Segerstolpe":
            exclude_cell_types_list = [['not applicable'], 
                                    ['delta'], 
                                    ['alpha'], 
                                    ['gamma'], 
                                    ['ductal'], 
                                    ['acinar'], 
                                    ['beta'], 
                                    ['unclassified endocrine'], 
                                    ['co-expression'], 
                                    ['MHC class II'], 
                                    ['PSC'], 
                                    ['endothelial'], 
                                    ['epsilon'], 
                                    ['mast'], 
                                    ['unclassified']]
        elif datset_name == "Baron":
            exclude_cell_types_list = [['acinar'], 
                                    ['beta'], 
                                    ['delta'], 
                                    ['activated_stellate'], 
                                    ['ductal'], 
                                    ['alpha'], 
                                    ['epsilon'], 
                                    ['gamma'], 
                                    ['endothelial'], 
                                    ['quiescent_stellate'], 
                                    ['macrophage'], 
                                    ['schwann'], 
                                    ['mast'], 
                                    ['t_cell']]
        elif datset_name == "Zheng68k":
            exclude_cell_types_list = [["CD8+ Cytotoxic T"], 
                                    ["CD8+/CD45RA+ Naive Cytotoxic"], 
                                    ["CD4+/CD45RO+ Memory"], 
                                    ["CD19+ B"], 
                                    ["CD4+/CD25 T Reg"], 
                                    ["CD56+ NK"], 
                                    ["CD4+ T Helper2"], 
                                    ["CD4+/CD45RA+/CD25- Naive T"], 
                                    ["CD34+"], 
                                    ["Dendritic"], 
                                    ["CD14+ Monocyte"]]
        else:
            raise ValueError("Not a valid dataset name! Must be MacParland, Segerstolpe or Baron")

        exclude_cell_types_list_names = [item[0] for item in exclude_cell_types_list]

        return exclude_cell_types_list, exclude_cell_types_list_names
    
    def calc_precision_and_coverage(self, threshold: float):
        all_true_positives = 0
        all_false_positives = 0
        all_all_postives = 0
        print(f"Calculations are done with a confidence threshold of {threshold}")
        print("If a data point exists with a confidence below this point, we assume there exists a novel cell type in the data")
        print("")
        for dataset_name in self.confidence_dict:
            data = self.confidence_dict[dataset_name]

            min_non_novel_confidence_temp = np.array(data["min_non_novel_confidence"])
            min_novel_confidence_temp = np.array(data["min_novel_confidence"])

            false_positives = min_non_novel_confidence_temp[min_non_novel_confidence_temp <= threshold]
            true_positives = min_novel_confidence_temp[min_novel_confidence_temp <= threshold]
            all_postives = min_novel_confidence_temp

            precision = 0
            try:
                precision = len(true_positives) / (len(true_positives) + len(false_positives))
            except:
                precision = "Can't calculate"

            coverage = len(true_positives) / len(all_postives)

            print("Dataset: ", dataset_name)
            print("_______________________")
            print("Precision: ", precision)
            print("Coverage: ", coverage)
            print("_______________________")
            print("")

            all_true_positives += len(true_positives)
            all_false_positives += len(false_positives)
            all_all_postives += len(all_postives)

        all_precision = 0
        try:
            all_precision = all_true_positives / (all_true_positives + all_false_positives)
        except:
            all_precision = "Can't calculate"

        all_coverage = all_true_positives / all_all_postives

        print("Precision and coverage when combining all datasets")
        print("_______________________")
        print("Precision: ", all_precision)
        print("Coverage: ", all_coverage)
        print("_______________________")
        print("")
        