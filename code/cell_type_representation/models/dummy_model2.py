import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import scanpy as sc
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from functions import data_preprocessing as dp

class prep_data(data.Dataset):

    def __init__(self, adata, embedding, HVG, Scaled, HVGs = 4000):
        self.adata = adata
        self.embedding = embedding
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

        self.make_embedding()

    def make_embedding(self):
        # Match correct gene2vec embeddings to correct genes
        gene_embeddings_dic = {}
        missing_gene_symbols = []
        gene_symbol_list = list(self.embedding.keys())
        for gene_symbol in self.adata.var.index:
            if gene_symbol in gene_symbol_list:
                gene_embeddings_dic[gene_symbol] = self.embedding[gene_symbol]
            else:
                #print(f"Gene symbol {gene_symbol} doesn't exists in embedded format")
                missing_gene_symbols.append(gene_symbol)

        #print("Number of missing gene symbols: ", len(missing_gene_symbols))

        # When gene symbol doesn't have an embedding we'll simply make a one hot encoded embedding for these
        onehot_template = np.zeros((1,len(missing_gene_symbols)))[0] # one hot embedding template

        # Add zero vector to the end of all embedded gene symbols
        for idx, gene_symbol in enumerate(gene_embeddings_dic.keys()):
            gene_embeddings_dic[gene_symbol] = np.concatenate([gene_embeddings_dic[gene_symbol],onehot_template])
        # Add the one hot encoding for missing gene symbols
        for idx, gene_symbol in enumerate(missing_gene_symbols):
            onehot_temp = onehot_template.copy()
            onehot_temp[idx] = 1.0
            gene_embeddings_dic[gene_symbol] = np.concatenate([np.zeros((1,200))[0],onehot_temp])

        #print(f"Final length of embedding: {len(gene_embeddings_dic[list(gene_embeddings_dic.keys())[0]])}")

        input_embedding = np.array([gene_embeddings_dic[gene_symbol] for gene_symbol in self.adata.var.index])

        self.input_embedding = torch.tensor(np.concatenate([input_embedding, np.zeros((np.shape(input_embedding)[0], 1))], axis=1))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        data_point = self.input_embedding
        data_point[:,-1] = self.X[idx]
        data_point = data_point.to(torch.float32)
        #data_point = self.X[idx]
        data_label = self.labels_onehotencoded[idx]
        return data_point, data_label
    
def scaled_dot_product(q, k, v):
    """
    From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html 
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    """
    From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html 
    """

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        o = self.o_proj(values).squeeze()

        if return_attention:
            return o, attention
        else:
            return o
        

class model(nn.Module):

    def __init__(self, num_inputs, emb_dim, num_hidden, num_outputs, drop_out):
        super().__init__()
        self.drop_out = drop_out

        self.self_attn = MultiheadAttention(emb_dim, 80, 8)
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear1_norm = nn.BatchNorm1d(num_hidden)
        self.act_fn1 = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, int(num_hidden/2))
        self.act_fn2 = nn.Tanh()
        self.linear3 = nn.Linear(int(num_hidden/2), num_outputs)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.linear1(x)
        x = self.linear1_norm(x)
        x = self.act_fn1(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.linear2(x)
        x = self.act_fn2(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1) 
        return x



