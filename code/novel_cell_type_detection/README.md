**novel_cell_type_detection:** Contains code for visualizing novel cell type detection function of the model.
1. Code for training model is in the *Alvis_cluster_code* repo. 
2. Code for extracting likelihood of all predictions of all datasets, cell type exclusions, and fold can also be found in the *Alvis_cluster_code* repo. This will give a *likelihood.json* file.
3. Move *likelihood.json* to */novel_cell_type_detection/results* in the *main* repo.
4. Simply run *novel_cell_type_confidence.ipynb* to make visualizations.