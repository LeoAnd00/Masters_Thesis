import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import pandas as pd

    
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
                 output_dim: int,
                 attn_drop_out: float=0., 
                 attn_bias: bool=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.attn_bias = attn_bias

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=attn_bias)
        #self.qkv_proj = CustomScaleModule(input_dim, 3*embed_dim, bias=attn_bias) # Use when having a vector input instead of matrix
        self.o_proj = nn.Linear(embed_dim, output_dim)
        self.attn_dropout1 = nn.Dropout(attn_drop_out)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        #nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.attn_bias:
        #    self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        batch_size, seq_length, _ = x.size()
        #batch_size, seq_length = x.size() # Use when having a vector input instead of matrix
        qkv = self.qkv_proj(x)#.to_sparse()

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
                 act_layer=nn.ReLU):
        super().__init__()
        self.mlp_linear1 = nn.Linear(in_features, hidden_features)
        self.mlp_act = act_layer()
        self.mlp_linear2 = nn.Linear(hidden_features, out_features)
        self.mlp_drop = nn.Dropout(drop)

        self._reset_parameters()

    def _reset_parameters(self):
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
    """
    A PyTorch module for an attention block in a neural network.

    This block combines self-attention and feedforward neural networks.

    Parameters
    ----------
    attn_input_dim : int
        The input dimension of the attention block.
    attn_embed_dim : int
        The embedding dimension for the attention mechanism.
    num_pathways : int
        The number of selected pathways.
    num_heads : int
        The number of attention heads in the multihead attention.
    output_dim : int
        The output dimension of the attention block. (Typically selected to be the same as attn_input_dim)
    mlp_ratio : float, optional
        The ratio to scale the hidden dimension of the feedforward neural network (default is 4.0).
    attn_bias : bool, optional
        Whether to use bias in the attention mechanism (default is False).
    mlp_drop : float, optional
        The dropout probability in the feedforward neural network (default is 0.0).
    attn_drop_out : float, optional
        The dropout probability in the attention mechanism (default is 0.0).
    act_layer : nn.Module, optional
        The activation function layer to use (default is nn.ReLU).
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.LayerNorm).
    last : bool, optional
        Whether this is the last attention block in the network (default is False).

    Attributes
    ----------
    attnblock_norm1 : nn.Module
        The first normalization layer for the attention block.
    attnblock_norm2 : nn.Module
        The second normalization layer for the attention block.
    attnblock_attn : MultiheadAttention
        The multihead attention mechanism.
    attnblock_mlp : AttentionMlp
        The feedforward neural network.
    linear_out : nn.Linear
        Linear layer for final output (used when 'last' is True).
    linear_out2 : nn.Linear
        Second linear layer for final output (used when 'last' is True).

    Methods
    -------
    forward(x)
        Forward pass of the attention block.
    """

    def __init__(self,
                 attn_input_dim: int, 
                 attn_embed_dim: int,
                 num_pathways: int,
                 num_heads: int,
                 output_dim: int,
                 mlp_ratio: float=4., 
                 attn_bias: bool=False,
                 mlp_drop: float=0., 
                 attn_drop_out: float=0.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,
                 last: bool=False):
        super(AttentionBlock, self).__init__()

        self.last = last

        if norm_layer == nn.LayerNorm:
            self.attnblock_norm1 = norm_layer(attn_input_dim)
            self.attnblock_norm2 = norm_layer(attn_input_dim)
        elif norm_layer == nn.BatchNorm1d:
            self.attnblock_norm1 = norm_layer(num_pathways)
            self.attnblock_norm2 = norm_layer(num_pathways)
        else:
            raise ValueError("norm_layer needs to be nn.LayerNorm or nn.BatchNorm1d")

        self.attnblock_attn = MultiheadAttention(attn_input_dim, 
                                                 attn_embed_dim, 
                                                 num_heads, 
                                                 output_dim, 
                                                 attn_drop_out, 
                                                 attn_bias)
        
        mlp_hidden_dim = int(attn_input_dim * mlp_ratio)
        self.attnblock_mlp = AttentionMlp(in_features=attn_input_dim, 
                                          hidden_features=mlp_hidden_dim, 
                                          out_features=attn_input_dim, 
                                          act_layer=act_layer, 
                                          drop=mlp_drop)
        
        self.linear_out = nn.Linear(output_dim,int(output_dim/4))
        self.act_layer_out=nn.ReLU()
        self.linear_out2 = nn.Linear(int(output_dim/4),1)

        self.linear_out_attn = nn.Linear(output_dim,int(output_dim/4))
        self.act_layer_out_attn=nn.ReLU()
        self.linear_out2_attn = nn.Linear(int(output_dim/4),1)

    def forward(self, x, return_attention=False):
        """
        Forward pass of the attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the attention block.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the attention block.
        """

        if return_attention:
            attn, attn_matrix = self.attnblock_attn(self.attnblock_norm1(x), return_attention=return_attention)
        else:
            attn = self.attnblock_attn(self.attnblock_norm1(x))

        if self.last is False:
            x = x + attn
            x = x + self.attnblock_mlp(self.attnblock_norm2(x))
        else:
            # Removes the additional dimension if it's the last attention block
            x = self.linear_out2(self.act_layer_out(self.linear_out(x))).squeeze() + self.linear_out2_attn(self.act_layer_out_attn(self.linear_out_attn(attn))).squeeze()
        
        if return_attention:
            return x, attn_matrix
        else:
            return x


class HVGTransformer(nn.Module):
    """
    A PyTorch module for a Transformer model for HVGs.

    This model processes pathway data using self-attention blocks.

    Parameters
    ----------
    attn_embed_dim : int
        The embedding dimension for the attention mechanism.
    num_HVGs : int
        The number of highly variable genes.
    num_pathways : int
        The number of pathways to consider.
    nn_embedding_dim : int, optional
        The embedding dimension for the HVGs data to be used by nn.Embedding() (default is 200).
    nn_tokens : int, optional
        The number of tokens used for pathways (default is 1000).
    num_heads : int, optional
        The number of attention heads in the self-attention blocks (default is 4).
    mlp_ratio : float, optional
        The ratio to scale the hidden dimension of the feedforward neural network within attention blocks (default is 4.0).
    attn_bias : bool, optional
        Whether to use bias in the attention mechanism (default is False).
    drop_ratio : float, optional
        The dropout ratio used within the attention blocks (default is 0.2).
    attn_drop_out : float, optional
        The dropout ratio used in the attention mechanism (default is 0.0).
    depth : int, optional
        The number of attention blocks in the model (default is 3).
    act_layer : nn.Module, optional
        The activation function layer to use (default is nn.ReLU).
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.LayerNorm).

    Attributes
    ----------
    pathways_input : nn.Linear
        Linear layer for pathway data input.
    blocks : nn.ModuleList
        List of AttentionBlock modules for processing pathway data.

    Methods
    -------
    forward(pathways)
        Forward pass of the Pathway Transformer model.
    """

    def __init__(self, 
                 attn_embed_dim: int,
                 num_HVGs: int,
                 nn_embedding_dim: int=200,
                 nn_tokens: int=1000,
                 num_heads: int=4,
                 mlp_ratio: float=4., 
                 attn_bias: bool=False,
                 drop_ratio: float=0.2, 
                 attn_drop_out: float=0.0,
                 depth: int=3,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.x_embbed_input = nn.Embedding(nn_tokens, nn_embedding_dim)

        if depth >= 2:
            self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=nn_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_HVGs,
                                    num_heads=num_heads, 
                                    output_dim=nn_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=False) for idx in range(int(depth-1))])
            
            self.blocks.append(AttentionBlock(attn_input_dim=nn_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_HVGs,
                                    num_heads=num_heads, 
                                    output_dim=nn_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=True))
        elif depth == 1:
            self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=nn_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_HVGs,
                                    num_heads=num_heads, 
                                    output_dim=nn_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=True)])

    def forward(self, x, gene2vec_emb, return_attention=False):
        """
        Forward pass of the transformer model.

        Parameters
        ----------
        x : torch.Tensor
            Tokenized expression levels.
        gene2vec_emb : torch.Tensor
            Gene2vec representations of all HVGs.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the Pathway Transformer model.
        """

        x = self.x_embbed_input(x)
        x += gene2vec_emb
        for idx, layer in enumerate(self.blocks):
            if return_attention and idx == 0:
                x, attn_matrix = layer(x, return_attention)
            else:
                x = layer(x, False)

        if return_attention:
            return x, attn_matrix
        else:
            return x
    

class OutputEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_HVGs_transformer: int,
                 output_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.0):
        super().__init__()

        input_dim = int(2*input_dim)
        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), int(input_dim/2))
        self.norm_layer1 = norm_layer(int(input_dim/2))
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.norm_layer2 = norm_layer(int(input_dim/4))
        self.dropout1 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.dropout2 = nn.Dropout(drop_out)
        self.linear3 = nn.Linear(int(input_dim/4), int(input_dim/8))
        self.norm_layer3 = norm_layer(int(input_dim/8))
        self.dropout3 = nn.Dropout(drop_out)
        self.linear3_act = act_layer()
        self.output = nn.Linear(int(input_dim/8), output_dim)

        self.linear1_transformer = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.dropout1_transformer = nn.Dropout(drop_out)

    def forward(self, x, x_transformer):
        
        x = torch.cat((x, x_transformer), dim=1)

        x = self.norm_layer_in(x)
        x = self.linear1(x)
        x = self.norm_layer1(x)
        x = self.linear1_act(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        x_transformer = self.linear1_transformer(x_transformer)
        x_transformer = self.dropout1_transformer(x_transformer)

        x += x_transformer
        x = self.norm_layer2(x)
        x = self.linear2_act(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.norm_layer3(x)
        x = self.linear3_act(x)
        x = self.output(x)
        
        return x

class CellType2VecModel(nn.Module):
    """
    A PyTorch module for a CellType2Vec model that only consists of a Output Encoder.

    This model processes input data through a encoder to produce cell type embeddings.

    Parameters
    ----------
    num_HVGs : int
        The input dimension of the encoder model. (Number of HVGs)
    num_HVGs_transformer : int
        The input dimension of the transformer model. (Number of HVGs, doesn't have to be the same as num_HVGs)
    attn_embed_dim : int
        The embedding dimension for the attention mechanism (defualt is 96).
    output_dim : int
        The output dimension of the model, representing cell type embeddings (default is 100).
    drop_out : float, optional
        The dropout ratio used in the output projection layer (default is 0.2).
    nn_embedding_dim : int, optional
        The embedding dimension for the HVGs data to be used by nn.Embedding() (default is 200).
    nn_tokens : int, optional
        The number of tokens used for pathways (default is 1000).
    num_heads : int, optional
        The number of attention heads in the self-attention blocks (default is 4).
    mlp_ratio : float, optional
        The ratio to scale the hidden dimension of the feedforward neural network within attention blocks (default is 4.0).
    attn_bias : bool, optional
        Whether to use bias in the attention mechanism (default is False).
    attn_drop_out : float, optional
        The dropout ratio used in the attention mechanism (default is 0.0).
    depth : int, optional
        The number of attention blocks in the model (default is 3).
    act_layer : nn.Module, optional
        The activation function layer to use (default is nn.ReLU).
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.BatchNorm1d).
    use_gene2vec_emb : bool, optional
        Whether to use gene2vec embbedings or not.

    Attributes
    ----------
    output_encoder : OutputEncoder
        The Output Encoder component for generating cell type embeddings.
    hvg_transformer : HVGTransformer
        Transformer taking HVG expressions.

    Methods
    -------
    forward(x, pathways)
        Forward pass of the CellType2Vec model.
    """

    def __init__(self, 
                 num_HVGs: int,
                 num_HVGs_transformer: int,
                 attn_embed_dim: int=96,
                 output_dim: int=100,
                 drop_out: float=0.2,
                 nn_embedding_dim: int=200,
                 nn_tokens: int=1000,
                 num_heads: int=4,
                 mlp_ratio: float=4.,
                 attn_bias: bool=False,
                 attn_drop_out: float=0.0,
                 depth: int=3,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 use_gene2vec_emb: bool=False):
        super().__init__()

        self.use_gene2vec_emb = use_gene2vec_emb
        self.nn_embedding_dim = nn_embedding_dim

        self.hvg_transformer = HVGTransformer(attn_embed_dim=attn_embed_dim,
                                                    num_HVGs=num_HVGs_transformer,
                                                    nn_embedding_dim=nn_embedding_dim,
                                                    nn_tokens=nn_tokens,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio, 
                                                    attn_bias=attn_bias,
                                                    drop_ratio=drop_out, 
                                                    attn_drop_out=attn_drop_out,
                                                    depth=depth,
                                                    act_layer=nn.ReLU,
                                                    norm_layer=nn.LayerNorm)
        
        self.output_encoder = OutputEncoder(input_dim=num_HVGs, 
                                            num_HVGs_transformer=nn_embedding_dim,
                                            output_dim=output_dim,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            drop_out=drop_out)

    def forward(self, x, x_not_tokenized, gene2vec_emb, return_attention=False):
        """
        Forward pass of the CellType2Vec model.

        Parameters
        ----------
        x : torch.Tensor
            Tokenized expression levels.
        x_not_tokenized : torch.Tensor
            Input tensor containing non-tokenized expression levels.
        gene2vec_emb : torch.Tensor
            Gene2vec representations of all HVGs.
        return_attention : bool, optional
            Whether to return the attention matrix between the input HVGs.

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type embeddings.
        """


        if self.use_gene2vec_emb:
            if return_attention:
                x_transformer, attn_matrix = self.hvg_transformer(x, gene2vec_emb, return_attention)
            else:
                x_transformer = self.hvg_transformer(x, gene2vec_emb, return_attention)
        else:
            if return_attention:
                x_transformer, attn_matrix = self.hvg_transformer(x, torch.zeros((x.size(1), self.nn_embedding_dim)), return_attention)
            else:
                x_transformer = self.hvg_transformer(x, torch.zeros((x.size(1), self.nn_embedding_dim)), return_attention)

        x = self.output_encoder(x_not_tokenized, x_transformer)

        if return_attention:
            return x, attn_matrix
        else:
            return x

