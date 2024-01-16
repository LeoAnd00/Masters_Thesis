# Load packages
import warnings
from benchmarks.benchmark import benchmark as benchmark
import argparse
import random

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)

### Commands to run on Alvis cluster
# Start by runing: cd Masters_Thesis/code/cell_type_representation/
# Then:
# Bone marrow: sbatch jobscript_bone_marrow.sh
# PBMC: sbatch jobscript_pbmc.sh
# Pancreas: sbatch jobscript_pancreas.sh
# Kidney: sbatch jobscript_kidney.sh
# All_merged: sbatch jobscript_all_merged.sh

### Run on CMD:
# How to run example (on bone marrow data set): python run_benchmark.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/immune_cells/merged/Oetjen_merged.h5ad' 'trained_models/Bone_marrow/' 'benchmarks/results/Bone_marrow/benchmark' '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/pathway_information/all_pathways.json' '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''

def read_save(benchmark_env, result_csv_path, read=False):

    if read:
        benchmark_env.read_csv(name=result_csv_path)

    benchmark_env.make_benchamrk_results_dataframe(min_max_normalize=False)

    benchmark_env.save_results_as_csv(name=result_csv_path)

def main(data_path: str, model_path: str, result_csv_path: str, pathway_path: str, gene2vec_path: str, image_path: str):
    """
    Main function to initiate and run the benchmark.

    How to run example (on bone marrow data set): python run_benchmark.py '../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad' '../trained_models/Bone_marrow/' '../benchmarks/results/Bone_marrow/Test' '../../../data/processed/pathway_information/c5_pathways.json' '../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''

    Parameters
    --------
    - data_path (str): 
        Path to the data file.

    - model_path (str): 
        Path to save the trained models.

    - result_csv_path (str): 
        Path to save the benchmark results as a CSV file.

    - pathway_path (str): 
        Path to the pathway information json file.

    - gene2vec_path (str): 
        Path to gene2vec representations.

    - image_path (str): 
        Path where images will be saved.
    """

    print("**Initiate Benchmark**")

    HVGs = 2000
    num_seeds = 1
    random_seeds = list(range(46, 46 + num_seeds))

    for idx, seed in enumerate(random_seeds):
        while True:  # Keep trying new seeds until no error occurs
            try:
                print("seed: ", seed)

                # Load data 
                benchmark_env = benchmark(data_path=data_path, pathway_path=pathway_path, gene2vec_path=gene2vec_path, image_path=f'{image_path}{HVGs}_HVGs_seed_{random_seeds[idx]}', batch_key="patientID", HVG=True, HVGs=HVGs, Scaled=False, seed=seed)

                # These methods doesn't change with random seed, hence only need to train once
                #if seed == random_seeds[0]:
                """if seed == 42:

                    print("**Start benchmarking unintegrated data**")
                    benchmark_env.unintegrated(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                    print("**Start benchmarking PCA method**")
                    benchmark_env.pca(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                    print("**Start benchmarking Scanorama method**")
                    benchmark_env.scanorama(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                    print("**Start benchmarking Harmony method**")
                    benchmark_env.harmony(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                    print("**Start benchmarking ComBat method**")
                    benchmark_env.combat(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                    print("**Start benchmarking DESC method**")
                    benchmark_env.desc(umap_plot=False,save_figure=True)
                    read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                print("**Start benchmarking scVI method**")
                vae = benchmark_env.scvi(umap_plot=False,save_figure=True)
                read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                print("**Start benchmarking scANVI method**")
                benchmark_env.scanvi(vae=vae,umap_plot=False,save_figure=True)
                read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                print("**Start benchmarking scGen method**")
                benchmark_env.scgen(umap_plot=False,save_figure=True)
                read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)"""

                #print("**Start benchmarking FastMNN method**")
                #benchmark_env.fastmnn(umap_plot=False,save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking TOSICA method**")
                #benchmark_env.tosica(umap_plot=False,save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Encoder method**")
                #benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #benchmark_env.in_house_model_encoder(save_path=f'{model_path}Testing/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=False)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)
                #read_save(benchmark_env, f'{result_csv_path}_Testing_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Pathways method**")
                #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Pathways/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Testing/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Encoder with Pathways method**")
                #benchmark_env.in_house_model_encoder_pathways(save_path=f'{model_path}Encoder_with_Pathways/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Transformer on HVGs method**")
                #benchmark_env.in_house_model_transformer_encoder(save_path=f'{model_path}Transformer_Encoder/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Transformer on HVGs and Pathways method**")
                #benchmark_env.in_house_model_transformer_encoder_pathways(save_path=f'{model_path}Transformer_Encoder_with_Pathways/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Transformer on Tokenized HVGs**")
                #benchmark_env.in_house_model_tokenized_HVG_transformer(save_path=f'{model_path}Tokenized_HVG_Transformer/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=False, umap_plot=False, save_figure=True)
                #benchmark_env.in_house_model_tokenized_HVG_transformer(save_path=f'{model_path}Testing/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=False)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)
                #read_save(benchmark_env, f'{result_csv_path}_Testing_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house Transformer on Tokenized HVGs and HVG Encoder**")
                #benchmark_env.in_house_model_tokenized_hvg_transformer_and_hvg_encoder(save_path=f'{model_path}Tokenized_HVG_Transformer_with_HVG_Encoder/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)

                #print("**Start benchmarking In-house Transformer on Tokenized HVGs and pathways**")
                #benchmark_env.in_house_model_tokenized_HVG_transformer_with_pathways(save_path=f'{model_path}Tokenized_HVG_Transformer_with_Pathways/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)
                #benchmark_env.in_house_model_tokenized_HVG_transformer_with_pathways(save_path=f'{model_path}Testing/Tokenized_HVG_Transformer_with_Pathways_{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=False)
                #read_save(benchmark_env, f'{result_csv_path}Tokenized_HVG_Transformer_with_Pathways_Testing_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                #print("**Start benchmarking In-house ITSCR model**")
                #benchmark_env.in_house_model_itscr(save_path=f'{model_path}ITSCR/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                #read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)
                #benchmark_env.in_house_model_itscr(save_path=f'{model_path}Testing/ITSCR_{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=False)
                #read_save(benchmark_env, f'{result_csv_path}_ITSCRTest_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=False)

                print("**Start benchmarking In-house ITSCR model only using HVGs**")
                benchmark_env.in_house_model_itscr_only_HVGs(save_path=f'{model_path}ITSCR_only_HVGs/{HVGs}_HVGs_seed_{random_seeds[idx]}', train=True, umap_plot=False, save_figure=True)
                read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{random_seeds[idx]}', read=True)
                
                # If no error occurs, break out of the while loop
                break
            except Exception as e:
                # Handle the exception (you can print or log the error if needed)
                print(f"Error occurred: {e}")

                # Generate a new random seed not in random_seeds list
                while True:
                    new_seed = random.randint(1, 10000)
                    if new_seed not in random_seeds:
                        break

                print(f"Trying a new random seed: {new_seed}")
                seed = new_seed

                break

    print("**Benchmark Finished**")

if __name__ == "__main__":
    """
    How to run example (on bone marrow data set): python run_benchmark.py '../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad' '../trained_models/Bone_marrow/' '../benchmarks/results/Bone_marrow/Test' '../../../data/processed/pathway_information/c5_pathways.json' '../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('result_csv_path', type=str, help='Path to save the benchmark results as a CSV file.')
    parser.add_argument('pathway_path', type=str, help='Path to the pathway information json file.')
    parser.add_argument('gene2vec_path', type=str, help='Path to gene2vec representations.')
    parser.add_argument('image_path', type=str, help='Path where images will be saved.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.pathway_path, args.gene2vec_path, args.image_path)