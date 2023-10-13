import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from functions import data_preprocessing as dp

class prep_data(data.Dataset):

    def __init__(self, adata, HVG, Scaled, HVGs = 4000):
        self.adata = adata
        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        if Scaled:
            self.adata.X = dp.scale_data(self.adata.X)
        self.X = self.adata.X
        self.labels = self.adata.obs["cell_type"]

        self.X = torch.tensor(self.X)

        self.label_encoder = LabelEncoder()
        self.labels_labelencoded = self.label_encoder.fit_transform(self.labels)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.labels_onehotencoded = torch.tensor(self.onehot_encoder.fit_transform(np.array(self.labels).reshape(-1, 1)))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        data_point = self.X[idx]
        data_label = self.labels_onehotencoded[idx]
        return data_point, data_label
        

class model(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs, drop_out):
        super().__init__()
        self.drop_out = drop_out
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear1_norm = nn.BatchNorm1d(num_hidden)
        self.act_fn1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, int(num_hidden/2))
        self.act_fn2 = nn.Tanh()
        self.linear3 = nn.Linear(int(num_hidden/2), num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_norm(x)
        x = self.act_fn1(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.linear2(x)
        x = self.act_fn2(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1) 
        return x



