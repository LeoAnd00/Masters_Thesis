import torch
import torch.nn as nn

#
# Model for classifying cell types
#

    
class ClassifierEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 first_layer_dim: int,
                 second_layer_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm1d,
                 drop_out: float=0.2):
        super().__init__()

        self.norm_layer_in = norm_layer(int(input_dim))
        self.linear1 = nn.Linear(int(input_dim), int(first_layer_dim))
        self.norm_layer1 = norm_layer(int(first_layer_dim))
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(first_layer_dim), int(second_layer_dim))
        self.norm_layer2 = norm_layer(int(second_layer_dim))
        self.dropout2 = nn.Dropout(drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(int(second_layer_dim), int(output_dim))

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
        x = nn.functional.softmax(x)

        return x

class ModelClassifier(nn.Module):
    """
    A PyTorch module for ModelClassifier.

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
                 num_cell_types: int,
                 classifier_act_layer=nn.ReLU,
                 classifier_norm_layer=nn.BatchNorm1d,
                 first_layer_dim: int=256,
                 second_layer_dim: int=256,
                 classifier_drop_out: float=0.2):
        super().__init__()

        self.classifier = ClassifierEncoder(input_dim=input_dim,
                                            output_dim=num_cell_types,
                                            first_layer_dim=first_layer_dim,
                                            second_layer_dim=second_layer_dim,
                                            act_layer=classifier_act_layer,
                                            norm_layer=classifier_norm_layer,
                                            drop_out=classifier_drop_out)

    def forward(self, x):
        """
        Forward pass of the CellType2Vec model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the model.

        Returns
        -------
        torch.Tensor
            Output tensor representing cell type annotations distribution from Softmax.
        """

        # Classifier
        x = self.classifier(x)

        # Ensure all tensors have at least two dimensions
        if x.dim() == 1:
            x = torch.unsqueeze(x, dim=0)  # Add a dimension along axis 0

        return x

