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
# All_merged: sbatch investigate_hvgs_jobscript_all_merged.sh


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

    print("**Initiate HVG Investiagtion**")

    HVGs_list = [100, 200, 500, 1000, 1500, 2000, 3000, 4000]
    random_seeds = [42]#, 43, 44, 45, 46, 47]

    for HVGs in HVGs_list:

        for seed in random_seeds:

            # Load data 
            benchmark_env = benchmark(data_path=data_path, pathway_path=pathway_path, gene2vec_path=gene2vec_path, image_path=image_path, batch_key="patientID", HVG=True, HVGs=HVGs, Scaled=False, seed=seed)

            print(f"**Start {HVGs} HVG Investiagtion In-house Encoder method**")
            print()
            benchmark_env.in_house_model_encoder(save_path=f'{model_path}Encoder/Invetigate_HVGs/{HVGs}_HVGs_seed_{seed}', train=True, umap_plot=False, save_figure=False)
            read_save(benchmark_env, f'{result_csv_path}_{HVGs}_HVGs_seed_{seed}', read=False)

            #print("**Start benchmarking In-house Pathways method**")
            #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Pathways/', train=True, umap_plot=False, save_figure=True)
            #benchmark_env.in_house_model_pathways(save_path=f'{model_path}Testing/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

            #print("**Start benchmarking In-house Encoder with Pathways method**")
            #benchmark_env.in_house_model_encoder_pathways(save_path=f'{model_path}Encoder_with_Pathways/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

            #print("**Start benchmarking In-house Transformer on HVGs method**")
            #benchmark_env.in_house_model_transformer_encoder(save_path=f'{model_path}Transformer_Encoder/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

            #print("**Start benchmarking In-house Transformer on HVGs and Pathways method**")
            #benchmark_env.in_house_model_transformer_encoder_pathways(save_path=f'{model_path}Transformer_Encoder_with_Pathways/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

            #print("**Start benchmarking In-house Transformer on Tokenized HVGs**")
            #benchmark_env.in_house_model_tokenized_HVG_transformer(save_path=f'{model_path}Tokenized_HVG_Transformer/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

            #print("**Start benchmarking In-house Transformer on Tokenized HVGs and pathways**")
            #benchmark_env.in_house_model_tokenized_HVG_transformer_with_pathways(save_path=f'{model_path}Tokenized_HVG_Transformer_with_Pathways/', train=True, umap_plot=False, save_figure=True)
            #read_save(benchmark_env, result_csv_path, read=True)

    print("**HVG Investiagtion Finished**")

if __name__ == "__main__":

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