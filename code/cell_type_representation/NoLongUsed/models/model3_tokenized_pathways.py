import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import pandas as pd
import math
import numpy as np
from einops import rearrange



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
            self.bias = Parameter(torch.empty(in_features))
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

        output = input * self.weight
        output = torch.sum(output, dim=2)
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
        #self.attn_dropout1 = nn.Dropout(attn_drop_out)

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

        #attn_output = self.attn_dropout1(attn_output)

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

def get_weight(att_mat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #v = v[:,1:,0] # Not original one
    #print(v.size())
    return v
    
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

        self.drop_path = DropPath(mlp_drop) if mlp_drop > 0. else nn.Identity()

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

        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.attnblock_mlp(self.attnblock_norm2(x)))

        #if self.last:
        #    x = x[:,:,0]
        #else:
        #    pass

        if return_attention:
            return x, attn_matrix
        else:
            return x




def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Args:
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        mask = mask.t()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
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

        #x = self.x_embbed_input(x)
        #x += gene2vec_emb
        attn_matrix = []
        for idx, layer in enumerate(self.blocks):
            if return_attention:# and idx == 0:
                x, attn_matrix_temp = layer(x, return_attention)
                attn_matrix.append(attn_matrix_temp)
            else:
                x = layer(x, False)

        if return_attention:
            attn_matrix = get_weight(attn_matrix)

        if return_attention:
            return x, attn_matrix
        else:
            return x
    

class OutputEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_pathways: int,
                 output_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.0):
        super().__init__()

        #input_dim = input_dim + num_pathways #+= num_pathways
        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), int(input_dim/2))
        self.norm_layer1 = norm_layer(int(input_dim/2))
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(input_dim/2), int(input_dim/4)) # num_pathways
        self.norm_layer2 = norm_layer(int(input_dim/4))
        self.dropout2 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(int(input_dim/4), output_dim)

    def forward(self, x, pathways):
        #x = torch.cat((x, pathways), dim=1)
        x = self.norm_layer_in(x)
        x = self.linear1(x)
        x = self.norm_layer1(x)
        x = self.linear1_act(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        #x += pathways
        x = self.norm_layer2(x)
        x = self.linear2_act(x)
        x = self.output(x)
        return x

class CellType2VecModel(nn.Module):
    """
    A PyTorch module for a CellType2Vec model that only consists of a Output Encoder.

    This model processes input data through a encoder to produce cell type embeddings.

    Parameters
    ----------
    input_dim : int
        The input dimension of the model. (Number of HVGs)
    attn_embed_dim : int
        The embedding dimension for the attention mechanism (defualt is 96).
    output_dim : int
        The output dimension of the model, representing cell type embeddings (default is 100).
    num_pathways: int
        The number of pathways (default is 300).
    pathway_embedding_dim : int, optional
        The embedding dimension for the pathway data (default is 50).
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
                 mask,
                 input_dim: int,
                 attn_embed_dim: int=96,
                 output_dim: int=100,
                 num_pathways: int=300,
                 pathway_embedding_dim: int=100,
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

        self.x_embbed_input = nn.Embedding(nn_tokens, nn_embedding_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, nn_embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.feature_embed = FeatureEmbed(input_dim, mask = mask, embed_dim=nn_embedding_dim, fe_bias=True)

        input_dim += 1 # To account for the class token

        input_dim += num_pathways

        self.hvg_transformer = HVGTransformer(attn_embed_dim=attn_embed_dim,
                                                    num_HVGs=input_dim,
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
        
        self.output_encoder = OutputEncoder(input_dim=input_dim, 
                                            num_pathways=num_pathways,
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
            Input tensor containing expression levels.
        gene2vec_emb : torch.Tensor
            Gene2vec representations of all HVGs.
        return_attention : bool, optional
            Whether to return the attention matrix between the input HVGs.

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type embeddings.
        """

        x = self.x_embbed_input(x)
        if self.use_gene2vec_emb:
            x += gene2vec_emb

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) 

        pathways = self.feature_embed(x_not_tokenized)
        x = torch.cat((x, pathways), dim=1) 

        if return_attention:
            x, attn_matrix = self.hvg_transformer(x, gene2vec_emb, return_attention)
        else:
            x = self.hvg_transformer(x, gene2vec_emb, return_attention)

        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add a dimension along axis 0

        x = x[:,:,0]
        
        #pathways = self.pathway_transformer(pathways)
        x = self.output_encoder(x, pathways)

        if return_attention:
            return x, attn_matrix
        else:
            return x

