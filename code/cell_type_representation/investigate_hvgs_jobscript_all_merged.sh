#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=V100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python run_benchmark_investigate_hvgs.py '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/immune_cells/merged/Oetjen_merged.h5ad' 'trained_models/Bone_marrow/' 'benchmarks/results/Bone_marrow/Investigate_HVGs/benchmark' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/c5_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' ''