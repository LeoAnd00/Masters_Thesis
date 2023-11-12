import torch
import torch.nn as nn

#
# Model using just a encoder for HVGs
#

class OutputEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.0):
        super().__init__()

        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), int(input_dim/2))
        self.norm_layer1 = norm_layer(int(input_dim/2))
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.norm_layer2 = norm_layer(int(input_dim/4))
        self.dropout2 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(int(input_dim/4), output_dim)

    def forward(self, x):
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
    A PyTorch module for a CellType2Vec model that only consists of a Output Encoder.

    This model processes input data through a encoder to produce cell type embeddings.

    Parameters
    ----------
    input_dim : int
        The input dimension of the model. (Number of HVGs)
    output_dim : int
        The output dimension of the model, representing cell type embeddings (default is 100).
    drop_out : float, optional
        The dropout ratio used in the output projection layer (default is 0.2).
    act_layer : nn.Module, optional
        The activation function layer to use (default is nn.ReLU).
    norm_layer : nn.Module, optional
        The normalization layer to use, either nn.LayerNorm or nn.BatchNorm1d (default is nn.BatchNorm1d).

    Attributes
    ----------
    output_encoder : OutputEncoder
        The Output Encoder component for generating cell type embeddings.

    Methods
    -------
    forward(x, pathways)
        Forward pass of the CellType2Vec model.
    """

    def __init__(self, 
                 input_dim: int,
                 output_dim: int=100,
                 drop_out: float=0.2,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d):
        super().__init__()
        
        self.output_encoder = OutputEncoder(input_dim=input_dim, 
                                            output_dim=output_dim,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            drop_out=drop_out)

    def forward(self, x, pathways):
        """
        Forward pass of the CellType2Vec model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the model.
        pathways : torch.Tensor
            Input tensor containing pathway data. (Not used by this model)

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type embeddings.
        """

        # Output encoder 
        #print("Shape: ", pathways.shape)
        #x = torch.cat((x, pathways), dim=1)
        x = self.output_encoder(pathways)

        return x

