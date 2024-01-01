#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=V100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_bone_marrow_generalizability.py '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'trained_models/Assess_generalisability/' 'benchmarks/results/Generalizability/Benchmark_results' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/all_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt'