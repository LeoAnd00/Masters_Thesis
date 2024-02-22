# scTRAC
scTRAC is a machine learning model designed to learn a latent space based on scRNA-Seq data. This embedding space can be used for visualization, generating cell type representations, and as input to a  cell type annotator. Hence scTRAC has the option to also train a cell type classifier. The classifier can be used for annotating cell types in new scRNA-Seq data or even detect novel celltypes, never seen by the model during training.

## Necessary programming languages
- Python version 3.10.5

## How to install (Coming soon)
```
pip install scTRAC
```

## How to use

### Train to make latent space
```
import scanpy as sc
import scTRAC.scTRAC as scTRAC

adata_train = sc.read(data_path_train, cache=True)
adata_test = sc.read(data_path_test, cache=True)

model = scTRAC.scTRAC(target_key=self.label_key, batch_key="batch")
model.train(adata=adata_train)
predictions = model.predict(adata=adata_test)
```

### Train to predict cell types
```
import scanpy as sc
import scTRAC.scTRAC as scTRAC

adata_train = sc.read(data_path_train, cache=True)
adata_test = sc.read(data_path_test, cache=True)

model = scTRAC.scTRAC(target_key=self.label_key, batch_key="batch")
model.train(adata=adata_train, train_classifier=True)
predictions = model.predict(adata=adata_test, use_classifier=True)
```

### Inference to detect novel cell types
Start by training the classifier as in **Train to predict cell types** and then run:
```
import scanpy as sc
import scTRAC.scTRAC as scTRAC

adata_test = sc.read(data_path_test, cache=True)

model = scTRAC.scTRAC(target_key=self.label_key, batch_key="batch")
predictions = model.predict(adata=adata_test, use_classifier=True, detect_unknowns=True, unknown_threshold=0.5)
```

## Authors
Leo Andrekson