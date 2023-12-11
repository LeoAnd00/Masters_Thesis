# Load packages
import warnings
from benchmarks.benchmark import benchmark as benchmark
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    How to run example (on bone marrow data set): python run_benchmark.py '../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad' '../trained_models/Bone_marrow/' '../benchmarks/results/Bone_marrow/Test' '../../../data/processed/pathway_information/all_pathways.json' '../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''

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

    num_seeds = 5
    random_seeds = list(range(42, 42 + num_seeds))

    for seed in random_seeds:

        # Load data 
        benchmark_env = benchmark(data_path=data_path, pathway_path=pathway_path, gene2vec_path=gene2vec_path, image_path=image_path, batch_key="patientID", HVG=True, HVGs=4000, Scaled=False, seed=seed)

        #print("**Start benchmarking unintegrated data**")
        #benchmark_env.unintegrated(umap_plot=False,save_figure=True)
        #benchmark_env.unintegrated(umap_plot=False,save_figure=False)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        """print("**Start benchmarking PCA method**")
        benchmark_env.pca(umap_plot=False,save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        print("**Start benchmarking Scanorama method**")
        benchmark_env.scanorama(umap_plot=False,save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        print("**Start benchmarking Harmony method**")
        benchmark_env.harmony(umap_plot=False,save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        print("**Start benchmarking scVI method**")
        vae = benchmark_env.scvi(umap_plot=False,save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        print("**Start benchmarking scANVI method**")
        benchmark_env.scanvi(vae=vae,umap_plot=False,save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)"""

        #print("**Start benchmarking scGen method**")
        #benchmark_env.scgen(umap_plot=False,save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking ComBat method**")
        #benchmark_env.combat(umap_plot=False,save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking DESC method**")
        #benchmark_env.desc(umap_plot=False,save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking FastMNN method**")
        #benchmark_env.fastmnn(umap_plot=False,save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking TOSICA method**")
        #benchmark_env.tosica(umap_plot=False,save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Encoder method**")
        #benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #benchmark_env.in_house_model_encoder(save_path=f'{model_path}Testing/seed_{seed}_', train=True, umap_plot=False, save_figure=False)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Pathways method**")
        #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Pathways/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Testing/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Encoder with Pathways method**")
        #benchmark_env.in_house_model_encoder_pathways(save_path=f'{model_path}Encoder_with_Pathways/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Transformer on HVGs method**")
        #benchmark_env.in_house_model_transformer_encoder(save_path=f'{model_path}Transformer_Encoder/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Transformer on HVGs and Pathways method**")
        #benchmark_env.in_house_model_transformer_encoder_pathways(save_path=f'{model_path}Transformer_Encoder_with_Pathways/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        print("**Start benchmarking In-house Transformer on Tokenized HVGs**")
        benchmark_env.in_house_model_tokenized_HVG_transformer(save_path=f'{model_path}Tokenized_HVG_Transformer/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

        #print("**Start benchmarking In-house Transformer on Tokenized HVGs and pathways**")
        #benchmark_env.in_house_model_tokenized_HVG_transformer_with_pathways(save_path=f'{model_path}Tokenized_HVG_Transformer_with_Pathways/seed_{seed}_', train=True, umap_plot=False, save_figure=True)
        #read_save(benchmark_env, f'{result_csv_path}_seed_{seed}', read=False)

    print("**Benchmark Finished**")

if __name__ == "__main__":
    """
    How to run example (on bone marrow data set): python run_benchmark.py '../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad' '../trained_models/Bone_marrow/' '../benchmarks/results/Bone_marrow/Test' '../../../data/processed/pathway_information/all_pathways.json' '../../../data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''
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