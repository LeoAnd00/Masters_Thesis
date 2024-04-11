# Learning Meaningful Representations of Cells - a Master's Thesis
## Reproducibility code

**Abstract**
<br>
Batch effects are a significant concern in single-cell RNA sequencing (scRNA-Seq) data analysis, where variations in the data can be attributed to factors unrelated to cell types. This can make downstream analysis a challenging task. In this study, a neural network model is designed utilizing contrastive learning and a novel loss function for creating an generalizable embedding space from scRNA-Seq data. When benchmarked against multiple established methods on scRNA-Seq integration, the model outperformed existing methods on creating a generalizable embedding space on multiple datasets. A downstream application that was investigated for the embedding space was cell type annotation. When compared against multiple well established cell type classifiers the model in this study displayed a performance competitive with top performing methods across multiple metrics, such as accuracy, balanced accuracy and F1 score. These findings motivates the meaningfulness contained within the generated embedding space by the model, highlighting its potential applications.

## Structur
- **code:** Contains all code used in this project, including preprocessing, visualization, machine learning models and more.
- **data:** Contains all data used in this project, including raw and preprocessed data.

Code for training the model developed in this study for reproducibility can be found in the *Alvis_cluster_code* repository, which contains code for running code on the Alvis computer cluser at Chalmers University of Technology. 

## Necessary programming languages
- Python version 3.10.5
- R version 4.3.2

## Official pip package
The model developed in this study have been made into a convenient package at X, and can be intalled by running:
```
pip install CELLULAR
```

## Authors
Leo Andrekson
