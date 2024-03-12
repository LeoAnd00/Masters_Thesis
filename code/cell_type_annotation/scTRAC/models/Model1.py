import torch
import torch.nn as nn

class OutputEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 first_layer_dim: int,
                 second_layer_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.0):
        super().__init__()

        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), first_layer_dim)
        self.norm_layer1 = norm_layer(first_layer_dim)
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.norm_layer2 = norm_layer(second_layer_dim)
        self.dropout2 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(second_layer_dim, output_dim)

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
    

class Model1(nn.Module):
    """
    A PyTorch module for Model1.

    This model processes input data through a encoder to produce cell type embeddings.

    Parameters
    ----------
    input_dim : int
        The input dimension of the model. (Number of HVGs)
    output_dim : int
        The output dimension of the model, representing cell type embeddings. Default is 100.
    drop_out : float, optional
        The dropout ratio used in the output projection layer. Default is 0.2.
    act_layer : nn.Module, optional
        The activation function layer to use. Default is nn.ReLU.
    norm_layer : nn.Module, optional
        The normalization layer to use. Default is nn.BatchNorm1d.

    Methods
    -------
    forward(x)
        Forward pass of Model1.
    """

    def __init__(self, 
                 input_dim: int,
                 output_dim: int=100,
                 drop_out: float=0.2,
                 first_layer_dim: int=1000,
                 second_layer_dim: int=500,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        
        self.output_encoder = OutputEncoder(input_dim=input_dim, 
                                            output_dim=output_dim,
                                            first_layer_dim=first_layer_dim,
                                            second_layer_dim=second_layer_dim,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            drop_out=drop_out)

    def forward(self, x):
        """
        Forward pass of Model1.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the model.

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type embeddings.
        """

        # Output encoder 
        x = self.output_encoder(x)

        # Ensure all tensors have at least two dimensions
        if x.dim() == 1:
            x = torch.unsqueeze(x, dim=0)  # Add a dimension along axis 0

        return x

