import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import scanpy as sc
import numpy as np
import math
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
        self.target = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        data_point = self.X[idx]#.reshape(-1,1)
        data_label = self.target[idx]
        return data_point, data_label
    

class CustomScaleModule(torch.nn.Module):
    """
    Inspired by nn.Linear: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear 
    But does more of a one-to-one unique scaling of each input (bias if wanted) into a new space, N times, making a matrix output
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features)))
        if bias:
            self.bias = Parameter(torch.empty(in_features, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        input = input.unsqueeze(2).expand(-1, -1, self.out_features)

        output = input * self.weight
        if self.bias is not None:
            output += self.bias

        return output
    
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
        self.qkv_proj = CustomScaleModule(input_dim, 3*embed_dim, bias=attn_bias)
        self.o_proj = nn.Linear(embed_dim, 1)
        self.attn_dropout1 = nn.Dropout(attn_drop_out)
        #self.attn_linear1 = nn.Linear(input_dim, input_dim)
        #self.attn_dropout2 = nn.Dropout(proj_drop_out)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        #nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.attn_bias:
        #    self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        #batch_size, seq_length, _ = x.size()
        batch_size, seq_length = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention_matrix = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)

        attn_output = self.o_proj(values).squeeze()

        attn_output = self.attn_dropout1(attn_output)
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
        #self.mlp_linear1 = nn.Linear(in_features, out_features)
        self.mlp_act = act_layer()
        self.mlp_linear2 = nn.Linear(hidden_features, out_features)
        self.mlp_drop = nn.Dropout(drop)

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
    
class AttentionBlock(nn.Module):
    def __init__(self,
                 attn_input_dim: int, 
                 attn_embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float=4., 
                 attn_bias: bool=True,
                 mlp_drop: float=0., 
                 attn_drop_out: float=0.,
                 proj_drop_out: float=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(AttentionBlock, self).__init__()
        self.attnblock_norm1 = norm_layer(attn_input_dim)
        self.attnblock_attn = MultiheadAttention(attn_input_dim, attn_embed_dim, num_heads, attn_drop_out, proj_drop_out, attn_bias)
        #self.attnblock_norm2 = norm_layer(attn_input_dim)
        #mlp_hidden_dim = int(attn_input_dim * mlp_ratio)
        #self.attnblock_mlp = AttentionMlp(in_features=attn_input_dim, 
        #                                  hidden_features=mlp_hidden_dim, 
        #                                  out_features=attn_input_dim, 
        #                                  act_layer=act_layer, 
        #                                  drop=mlp_drop)
    def forward(self, x):
        attn = self.attnblock_attn(self.attnblock_norm1(x))
        x = x + attn
        #x = x + self.attnblock_mlp(x)
        #x = x + self.attnblock_mlp(self.attnblock_norm2(x))
        return x
        

class model(nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 attn_embed_dim: int,
                 output_dim: int,
                 num_heads: int=1,
                 mlp_ratio: float=2., 
                 attn_bias: bool=False,
                 drop_ratio: float=0.3, 
                 attn_drop_out: float=0.0,
                 proj_drop_out: float=0.3,
                 depth: int=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=input_dim, 
                                   attn_embed_dim=attn_embed_dim,
                                   num_heads=num_heads, 
                                   mlp_ratio=mlp_ratio, 
                                   attn_bias=attn_bias,
                                   mlp_drop=drop_ratio, 
                                   attn_drop_out=attn_drop_out, 
                                   proj_drop_out=proj_drop_out,
                                   norm_layer=norm_layer, 
                                   act_layer=act_layer) for idx in range(depth)])

        self.norm_layer_in = norm_layer(int(input_dim))
        self.classifier_linear1 = nn.Linear(int(input_dim), int(input_dim/4))
        self.norm_layer1 = norm_layer(int(input_dim/4))
        self.classifier__drop1 = nn.Dropout(proj_drop_out)
        self.classifier_linear1_act = nn.Tanh()
        self.classifier_linear2 = nn.Linear(int(input_dim/4), output_dim)

    def forward(self, x):
        #for layer in self.blocks:
        #    x = layer(x)
        x = self.norm_layer_in(x)
        x = self.classifier_linear1(x)
        x = self.norm_layer1(x)
        x = self.classifier_linear1_act(x)
        x = self.classifier__drop1(x)
        x = self.classifier_linear2(x)
        return x



