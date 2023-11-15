import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math

#
# Model using pathway information
#


class CustomScaleModule(torch.nn.Module):
    """
    Inspired by the nn.Linear function: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear 
    \nOne-to-one unique scaling of each input (bias if wanted) into a new space, out_features times, making a matrix output
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
        
        self.linear_out = nn.Linear(output_dim,1)
        self.linear_out2 = nn.Linear(output_dim,1)

    def forward(self, x):
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

        attn = self.attnblock_attn(self.attnblock_norm1(x))
        if self.last is False:
            x = x + attn
            x = x + self.attnblock_mlp(self.attnblock_norm2(x))
        else:
            # Essentially removes the additional dimension if it's the last attention block
            x = self.linear_out(x).squeeze() + self.linear_out2(attn).squeeze()
        return x

class PathwayTransformer(nn.Module):
    """
    A PyTorch module for a Pathway Transformer model.

    This model processes pathway data using self-attention blocks.

    Parameters
    ----------
    attn_embed_dim : int
        The embedding dimension for the attention mechanism.
    HVGs : int
        The number of highly variable genes.
    num_pathways : int
        The number of pathways to consider.
    pathway_embedding_dim : int, optional
        The embedding dimension for the pathway data (default is 50).
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
                 HVGs: int,
                 num_pathways: int,
                 pathway_embedding_dim: int=50,
                 num_heads: int=4,
                 mlp_ratio: float=4., 
                 attn_bias: bool=False,
                 drop_ratio: float=0.2, 
                 attn_drop_out: float=0.0,
                 depth: int=3,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.pathways_input = nn.Linear(HVGs, pathway_embedding_dim)

        if depth >= 2:
            self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=pathway_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_pathways,
                                    num_heads=num_heads, 
                                    output_dim=pathway_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=False) for idx in range(int(depth-1))])
            
            self.blocks.append(AttentionBlock(attn_input_dim=pathway_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_pathways,
                                    num_heads=num_heads, 
                                    output_dim=pathway_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=True))
        elif depth == 1:
            self.blocks = nn.ModuleList([AttentionBlock(attn_input_dim=pathway_embedding_dim, 
                                    attn_embed_dim=attn_embed_dim,
                                    num_pathways=num_pathways,
                                    num_heads=num_heads, 
                                    output_dim=pathway_embedding_dim,
                                    mlp_ratio=mlp_ratio, 
                                    attn_bias=attn_bias,
                                    mlp_drop=drop_ratio, 
                                    attn_drop_out=attn_drop_out, 
                                    norm_layer=norm_layer, 
                                    act_layer=act_layer,
                                    last=True)])

    def forward(self, pathways):
        """
        Forward pass of the Pathway Transformer model.

        Parameters
        ----------
        pathways : torch.Tensor
            Input tensor containing pathway data.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the Pathway Transformer model.
        """

        pathways = self.pathways_input(pathways)
        for layer in self.blocks:
            pathways = layer(pathways)

        return pathways
    
class OutputEncoder(nn.Module):
    def __init__(self, 
                 num_pathways: int,
                 output_dim: int,
                 norm_layer=nn.BatchNorm1d):
        super().__init__()

        self.pathways_norm = norm_layer(num_pathways)
        self.output = nn.Linear(num_pathways, output_dim)

    def forward(self, x, pathways):
        x = self.pathways_norm(pathways)
        x = self.output(x)
        return x

class CellType2VecModel(nn.Module):
    """
    A PyTorch module for a CellType2Vec model that combines a Pathway Transformer and Output Encoder.

    This model processes input data and pathway information to produce cell type embeddings.

    Parameters
    ----------
    input_dim : int
        The input dimension of the model. (Number of HVGs)
    output_dim : int
        The output dimension of the model, representing cell type embeddings.
    HVGs : int
        The number of highly variable genes.
    num_pathways : int
        The number of pathways to consider.
    attn_embed_dim : int
        The embedding dimension for the attention mechanism in the Pathway Transformer (defualt is 96).
    num_heads : int, optional
        The number of attention heads in the Pathway Transformer (default is 4).
    mlp_ratio : float, optional
        The ratio to scale the hidden dimension of the feedforward neural network within the Pathway Transformer (default is 4.0).
    attn_bias : bool, optional
        Whether to use bias in the attention mechanism (default is False).
    drop_ratio : float, optional
        The dropout ratio used within the Pathway Transformer (default is 0.2).
    attn_drop_out : float, optional
        The dropout ratio used in the attention mechanism (default is 0.0).
    proj_drop_out : float, optional
        The dropout ratio used in the output projection layer (default is 0.2).
    depth : int, optional
        The depth of the Pathway Transformer, indicating the number of attention blocks (default is 3).
    act_layer : nn.Module, optional
        The activation function layer to use (default is nn.ReLU).
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.BatchNorm1d).
    pathway_embedding_dim : int, optional
        The embedding dimension for pathway data (default is 50).

    Attributes
    ----------
    pathway_transformer : PathwayTransformer
        The Pathway Transformer component for processing pathway data.
    output_encoder : OutputEncoder
        The Output Encoder component for generating cell type embeddings.

    Methods
    -------
    forward(x, pathways)
        Forward pass of the CellType2Vec model.
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 num_pathways: int,
                 attn_embed_dim: int=96,
                 num_heads: int=4,
                 mlp_ratio: float=4.,
                 attn_bias: bool=False,
                 drop_ratio: float=0.2, 
                 attn_drop_out: float=0.0,
                 depth: int=3,
                 norm_layer=nn.BatchNorm1d,
                 pathway_embedding_dim: int=50):
        super().__init__()
        assert depth >= 1, "depth should be bigger than 1"
        assert num_heads >= 1, "num_heads should be bigger than 1"
        assert mlp_ratio >= 1.0, "mlp_ratio should be bigger than 1.0"
        
        self.pathway_transformer = PathwayTransformer(attn_embed_dim=attn_embed_dim,
                                                        HVGs=input_dim, 
                                                        num_pathways=num_pathways,
                                                        pathway_embedding_dim=pathway_embedding_dim,
                                                        num_heads=num_heads,
                                                        mlp_ratio=mlp_ratio, 
                                                        attn_bias=attn_bias,
                                                        drop_ratio=drop_ratio, 
                                                        attn_drop_out=attn_drop_out,
                                                        depth=depth,
                                                        act_layer=nn.ReLU,
                                                        norm_layer=nn.LayerNorm)
        
        self.output_encoder = OutputEncoder(num_pathways=num_pathways,
                                            output_dim=output_dim,
                                            norm_layer=norm_layer)

    def forward(self, x, pathways):
        """
        Forward pass of the CellType2Vec model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the model.
        pathways : torch.Tensor
            Input tensor containing pathway data.

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type embeddings.
        """

        # Pathways transformer
        pathways = self.pathway_transformer(pathways)

        # Output encoder 
        x = self.output_encoder(x, pathways)

        return x

