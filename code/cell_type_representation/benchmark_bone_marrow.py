# Load packages
import warnings
from benchmarks.benchmark import benchmark as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(data_path):
    print("**Initiate Benchmark**")
    # Load data 
    benchmark_env = benchmark(data_path=data_path, batch_key="patientID", HVG=True, HVGs=4000, Scaled=False, seed=42)

    print("**Start benchmarking unintegrated data**")
    benchmark_env.unintegrated(umap_plot=False,save_figure=True)

    print("**Start benchmarking PCA method**")
    benchmark_env.pca(umap_plot=False,save_figure=True)

    print("**Start benchmarking Scanorama method**")
    benchmark_env.scanorama(umap_plot=False,save_figure=True)

    print("**Start benchmarking Harmony method**")
    benchmark_env.harmony(umap_plot=False,save_figure=True)

    print("**Start benchmarking scVI method**")
    vae = benchmark_env.scvi(umap_plot=False,save_figure=True)

    print("**Start benchmarking scANVI method**")
    benchmark_env.scanvi(vae=vae,umap_plot=False,save_figure=True)

    print("**Start benchmarking scGen method**")
    benchmark_env.scgen(umap_plot=False,save_figure=True)

    print("**Start benchmarking ComBat method**")
    benchmark_env.combat(umap_plot=False,save_figure=True)

    print("**Start benchmarking DESC method**")
    benchmark_env.desc(umap_plot=False,save_figure=True)

    print("**Start benchmarking FastMNN method**")
    benchmark_env.fastmnn(umap_plot=False,save_figure=True)

    print("**Start benchmarking In-house Encoder method**")
    benchmark_env.in_house_model_encoder(save_path='trained_models/Bone_marrow/Encoder/', train=False, umap_plot=False, save_figure=True)

    print("**Start benchmarking In-house Pathways method**")
    benchmark_env.in_house_model_pathways(save_path='trained_models/Bone_marrow/Pathways/', train=False, umap_plot=False, save_figure=True)

    print("**Start benchmarking In-house Encoder with Pathways method**")
    benchmark_env.in_house_model_encoder_pathways(save_path='trained_models/Bone_marrow/Encoder_with_Pathways/', train=False, umap_plot=False, save_figure=True)

    print("**Start benchmarking In-house Encoder with Pathways (Without Attention) method**")
    benchmark_env.in_house_model_encoder_pathways_no_attention(save_path='trained_models/Bone_marrow/Pathways_no_Attention/', train=False, umap_plot=False, save_figure=True)

    print("**Start benchmarking In-house Transformer on HVGs method**")
    benchmark_env.in_house_model_transformer_encoder(save_path='trained_models/Bone_marrow/Transformer_Encoder/', train=False, umap_plot=False, save_figure=True)

    print("**Start benchmarking In-house Transformer on HVGs and Pathways method**")
    benchmark_env.in_house_model_transformer_encoder_pathways(save_path='trained_models/Bone_marrow/Transformer_Encoder_with_Pathways/', train=False, umap_plot=False, save_figure=True)

    benchmark_env.make_benchamrk_results_dataframe(min_max_normalize=False)

    benchmark_env.save_results_as_csv(name="benchmarks/results/Bone_marrow/Test")

    print("**Benchmark Finished**")

if __name__ == "__main__":
    data_path = '../../data/processed/immune_cells/merged/Oetjen_merged.h5ad'
    main(data_path)