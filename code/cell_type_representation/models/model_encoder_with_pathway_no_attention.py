import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math

#
# Model using pathway information
#


class PathwayEncoder(nn.Module):
    """
    A PyTorch module for a Pathway Transformer model.

    This model processes pathway data using self-attention blocks.

    Parameters
    ----------
    HVGs : int
        The number of highly variable genes.
    pathway_embedding_dim : int, optional
        The embedding dimension for the pathway data (default is 50).
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
                 HVGs: int,
                 pathway_embedding_dim: int=50,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.pathways_input = nn.Linear(HVGs, pathway_embedding_dim)
        self.normalize = norm_layer(pathway_embedding_dim)
        self.activation = act_layer()
        self.pathways_output = nn.Linear(pathway_embedding_dim, 1)

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
        pathways = self.normalize(pathways)
        pathways = self.activation(pathways)
        pathways = self.pathways_output(pathways).squeeze()

        return pathways
    
class OutputEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_pathways: int,
                 output_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.0):
        super().__init__()

        input_dim = input_dim + num_pathways
        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), int(input_dim/2))
        self.norm_layer1 = norm_layer(int(input_dim/2))
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.norm_layer2 = norm_layer(int(input_dim/4))
        self.dropout2 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(int(input_dim/4), output_dim)

    def forward(self, x, pathways):
        x = torch.cat((x, pathways), dim=1)

        # Encoder for HVGs and pathways
        x = self.norm_layer_in(x)
        x = self.linear1(x)
        x = self.norm_layer1(x)
        x = self.linear1_act(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.norm_layer2(x)
        x = self.linear2_act(x)
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
    num_pathways : int
        The number of pathways to consider.
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.BatchNorm1d).
    drop_ratio : float, optional
        The dropout ratio used within the Pathway Transformer (default is 0.2).
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
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_ratio: float=0.2,
                 pathway_embedding_dim: int=50):
        super().__init__()
        
        self.pathway_encoder = PathwayEncoder(HVGs=input_dim, 
                                                pathway_embedding_dim=pathway_embedding_dim,
                                                act_layer=nn.ReLU,
                                                norm_layer=nn.LayerNorm)
        
        self.output_encoder = OutputEncoder(input_dim=input_dim,
                                            num_pathways=num_pathways,
                                            output_dim=output_dim,
                                            drop_out=drop_ratio,
                                            act_layer=act_layer,
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
        pathways = self.pathway_encoder(pathways)

        # Output encoder 
        x = self.output_encoder(x, pathways)

        return x

