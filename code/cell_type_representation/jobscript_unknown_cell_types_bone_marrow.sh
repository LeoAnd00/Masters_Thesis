#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 16:00:00
#SBATCH --gpus-per-node=A100:4

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_bone_marrow_unknown_cell_types.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/data_to_assess_generalisability/bone_marrow_human/Assess_generalisability_bone_marrow.h5ad' 'trained_models/Assess_unknown_cells/' '_Assess_unknown_cells' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/c5_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' 'patientID' 'cell_type'