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
    Modified from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html 
    """

    def __init__(self, 
                 input_dim: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 attn_drop_out: float=0., 
                 proj_drop_out: float=0., 
                 attn_bias: bool=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.attn_bias = attn_bias

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=attn_bias)
        self.o_proj = nn.Linear(embed_dim, input_dim)
        self.attn_dropout1 = nn.Dropout(attn_drop_out, self.training)
        self.attn_linear1 = nn.Linear(input_dim, input_dim)
        self.attn_dropout2 = nn.Dropout(proj_drop_out, self.training)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.attn_bias:
            self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention_matrix = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        attn_output = self.o_proj(values)#.squeeze()

        #attn_output = self.attn_dropout1(attn_output)
        #attn_output = self.attn_linear1(attn_output)
        #attn_output = self.attn_dropout2(attn_output)

        if return_attention:
            return attn_output, attention_matrix
        else:
            return attn_output
        
class AttentionMlp(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 out_features: int, 
                 drop: float=0., 
                 act_layer=nn.GELU):
        super().__init__()
        self.mlp_linear1 = nn.Linear(in_features, hidden_features)
        self.mlp_act = act_layer()
        self.mlp_linear2 = nn.Linear(hidden_features, out_features)
        self.mlp_drop = nn.Dropout(drop, self.training)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.mlp_linear1.weight)
        self.mlp_linear1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp_linear2.weight)
        self.mlp_linear2.bias.data.fill_(0)

    def forward(self, x):
        x = self.mlp_linear1(x)
        x = self.mlp_act(x)
        x = self.mlp_drop(x)
        x = self.mlp_linear2(x)
        x = self.mlp_drop(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From: https://github.com/JackieHanLab/TOSICA/tree/main 
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    From: https://github.com/JackieHanLab/TOSICA/tree/main 
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class AttentionBlock(nn.Module):
    def __init__(self,
                 attn_input_dim: int, 
                 attn_embed_dim: int,
                 attn_out_dim: int,
                 num_heads: int,
                 mlp_ratio: float=4., 
                 attn_bias: bool=True,
                 mlp_drop: float=0., 
                 attn_drop_out: float=0.,
                 proj_drop_out: float=0.,
                 drop_path_ratio: float=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(AttentionBlock, self).__init__()
        self.attnblock_norm1 = norm_layer(attn_input_dim)
        self.attnblock_attn = MultiheadAttention(attn_input_dim, attn_embed_dim, num_heads, attn_drop_out, proj_drop_out, attn_bias)
        self.attnblock_drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.attnblock_norm2 = norm_layer(attn_input_dim)
        mlp_hidden_dim = int(attn_input_dim * mlp_ratio)
        self.attnblock_mlp = AttentionMlp(in_features=attn_input_dim, 
                                          hidden_features=mlp_hidden_dim, 
                                          out_features=attn_out_dim, 
                                          act_layer=act_layer, 
                                          drop=mlp_drop)
    def forward(self, x):
        attn = self.attnblock_attn(self.attnblock_norm1(x))
        x = x + attn
        x = x + self.attnblock_mlp(self.attnblock_norm2(x))
        #x = self.attnblock_norm2(x + self.attnblock_drop_path(attn))
        #x = x + self.attnblock_drop_path(self.attnblock_mlp(self.attnblock_norm2(x)))
        return x
        

class model(nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 input_dim_emb: int,
                 attn_embed_dim: int,
                 attn_out_dim: int,
                 num_classes: int,
                 num_heads: int=1,
                 mlp_ratio: float=2., 
                 attn_bias: bool=True,
                 drop_ratio: float=0.3, 
                 attn_drop_out: float=0.3,
                 proj_drop_out: float=0.3,
                 drop_path_ratio: float=0.3,
                 depth: int=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        drop_path_ratios = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=input_dim_emb, 
                                   attn_embed_dim=attn_embed_dim,
                                   attn_out_dim=attn_out_dim,
                                   num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, 
                                   attn_bias=attn_bias,
                                   mlp_drop=drop_ratio, 
                                   attn_drop_out=attn_drop_out, 
                                   proj_drop_out=proj_drop_out,
                                   drop_path_ratio=drop_path_ratios[idx],
                                   norm_layer=norm_layer, 
                                   act_layer=act_layer) for idx in range(depth)])

        self.norm_layer1 = norm_layer(input_dim_emb)
        self.classifier_linear1 = nn.Linear(input_dim_emb, int(input_dim_emb/4))
        self.classifier_linear1_act = nn.Tanh()
        self.classifier_linear2 = nn.Linear(int(input_dim_emb/4), int(input_dim_emb/16))
        self.classifier_linear2_act = nn.Tanh()
        self.classifier_linear3 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        #x_input = x
        for layer in self.blocks:
            x = layer(x)
        x = self.norm_layer1(x)
        x = self.classifier_linear1(x)
        x = self.classifier_linear1_act(x)
        x = self.classifier_linear2(x)
        x = self.classifier_linear2_act(x)
        x = torch.mean(x, dim=2)
        x = self.classifier_linear3(x)
        x = F.softmax(x, dim=1) 
        return x



