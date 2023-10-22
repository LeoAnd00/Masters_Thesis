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
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
import time as time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import umap
from functions import data_preprocessing as dp

class prep_data(data.Dataset):

    def __init__(self, adata, HVG: bool, Scaled: bool, HVGs:int = 4000):
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
                 attn_drop_out: float=0., 
                 proj_drop_out: float=0., 
                 attn_bias: bool=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.attn_bias = attn_bias

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = CustomScaleModule(input_dim, 3*embed_dim, bias=attn_bias)
        self.o_proj = nn.Linear(embed_dim, 1)
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
        #batch_size, seq_length, _ = x.size()
        batch_size, seq_length = x.size()
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
                 act_layer=nn.GELU):
        super().__init__()
        self.mlp_linear1 = nn.Linear(in_features, hidden_features)
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
        

class CellType2VecModel(nn.Module):

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
                 act_layer=nn.Tanh,
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
        self.linear1 = nn.Linear(int(input_dim), int(input_dim/2))
        self.norm_layer1 = norm_layer(int(input_dim/2))
        self.dropout1 = nn.Dropout(proj_drop_out)
        self.linear1_act = act_layer()
        self.linear2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.norm_layer2 = norm_layer(int(input_dim/4))
        self.dropout2 = nn.Dropout(proj_drop_out)
        self.linear2_act = act_layer()
        self.output = nn.Linear(int(input_dim/4), output_dim)

    def forward(self, x):
        #for layer in self.blocks:
        #    x = layer(x)
        x = self.norm_layer_in(x)
        #x = self.dropout1(x)
        x = self.linear1(x)
        x = self.norm_layer1(x)
        x = self.linear1_act(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.norm_layer2(x)
        x = self.linear2_act(x)
        x = self.output(x)
        return x











class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
        """
        From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        """

        def __init__(self, optimizer, warmup, max_iters):
            self.warmup = warmup
            self.max_num_iters = max_iters
            super().__init__(optimizer)

        def get_lr(self):
            lr_factor = self.get_lr_factor(epoch=self.last_epoch)
            return [base_lr * lr_factor for base_lr in self.base_lrs]

        def get_lr_factor(self, epoch):
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
            if epoch <= self.warmup:
                lr_factor *= epoch * 1.0 / self.warmup
            return lr_factor
        

class EarlyStopping():
    """
    Early Stopping Callback for Training

    This class is a callback for early stopping during training based on validation loss. It monitors the validation loss and stops training if the loss does not improve for a certain number of consecutive epochs.

    Args:
        tolerance (int, optional): Number of epochs to wait for an improvement in validation loss before stopping. Default is 10.
    """
    
    def __init__(self, tolerance: int=10):

        self.tolerance = tolerance
        self.min_val = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss >= self.min_val:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.min_val = val_loss
            self.counter = 0


class SNNLoss(nn.Module):
    """
    Soft Nearest Neighbor Loss

    This PyTorch loss function computes the Soft Nearest Neighbor (SNN) loss for a given set of input vectors and their corresponding targets. The SNN loss encourages the similarity between vectors of the same class while discouraging the similarity between vectors of different classes.

    Args:
        use_weights (bool, optional): If True, calculate class weights based on label frequency. Default is True.
        targets (Tensor, optional): A tensor containing the class labels for the input vectors. Required if use_weights is True.
        temperature (float, optional): A scaling factor applied to the cosine similarity. Default is 1.0.
    """
    def __init__(self, use_weights: bool=True, targets=None, temperature: float=1.0, device: str="cuda"):
        super(SNNLoss, self).__init__()
        
        #self.temperature = temperature
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True) 
        self.device = device
        self.use_weights = use_weights

        # Calculate weights for the loss based on label frequency
        if self.use_weights:
            if targets is not None:
                self.weight = self.calculate_class_weights(targets)
            else:
                raise ValueError("'use_weights' is True, but 'targets' is not provided.")

    def calculate_class_weights(self, targets):
        """
        Calculate class weights based on label frequency.

        Args:
            targets (Tensor): A tensor containing the class labels.
        """

        class_counts = torch.bincount(targets)  # Count the occurrences of each class
        class_weights = 1.0 / class_counts.float()  # Calculate inverse class frequencies
        class_weights /= class_weights.sum()  # Normalize to sum to 1

        class_weight_dict = {class_label: weight for class_label, weight in enumerate(class_weights)}

        return class_weight_dict

    def forward(self, input, targets):
        """
        Compute the SNN loss for the input vectors and targets.

        Args:
            input (Tensor): Input vectors.
            targets (Tensor): Class labels for the input vectors.
        """

        # Restrict the temperature term
        if self.temperature.item() <= 0.1:
            self.temperature.data = torch.tensor(0.1)
        elif self.temperature.item() >= 1.0:
            self.temperature.data = torch.tensor(1.0)

        # Calculate the cosine similarity matrix
        cosine_similarity_matrix = F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperature

        # Define a loss dictionary containing the loss of each label
        loss_dict = {str(target): torch.tensor([]).to(self.device) for target in targets.unique()}
        for idx, (sim_vec, target) in enumerate(zip(cosine_similarity_matrix,targets)):
            positiv_samples = torch.sum(torch.exp(sim_vec[(targets == target)])) - torch.exp(sim_vec[idx])
            negativ_samples = torch.sum(torch.exp(sim_vec[(targets != target)]))
            loss = -torch.log(positiv_samples / (positiv_samples + negativ_samples))
            loss_dict[str(target)] = torch.cat((loss_dict[str(target)], loss.unsqueeze(0)))

        # Calculate the weighted average loss
        weighted_losses = []
        for target in targets.unique():
            if len(targets[targets==target]) > 1: # Skip if there's only one sample of given label in the batch
                losses_for_target = loss_dict[str(target)]
                if self.use_weights:
                    weighted_loss = torch.mean(losses_for_target) * self.weight[int(target)]
                else:
                    weighted_loss = torch.mean(losses_for_target)
                weighted_losses.append(weighted_loss)

        loss = torch.mean(torch.stack(weighted_losses))

        return loss
    

class train_module():
    """
    A class for training the machine learning model using single-cell RNA sequencing data as input.

    Parameters:
        data_path (str or AnnData): Path to the data file or an AnnData object containing single-cell RNA
            sequencing data. If a path is provided, the data will be loaded from the specified file. If an AnnData object is
            provided, it will be used directly.

        save_model_path (str): The path to save the trained model.

        HVG (bool): Whether to identify highly variable genes (HVGs) in the data. Highly variable genes are used to reduce
            dimensionality and improve model performance. (Default: True)

        HVGs (int): The number of highly variable genes to select. This parameter determines how many of the identified HVGs
            are used in training. (Default: 4000)

        Scaled (bool): Whether to scale (normalize) the data before training. Data scaling can help models perform better,
            especially when features have different scales. (Default: False)
    """

    def __init__(self, 
                 data_path, 
                 save_model_path: str,
                 HVG: bool=True, 
                 HVGs: int=4000, 
                 Scaled: bool=False):
        
        if type(data_path) == str:
            self.adata = sc.read(data_path, cache=True)
        else:
            self.adata = data_path

        self.data_env = prep_data(adata=self.adata, HVG=HVG, Scaled=Scaled, HVGs=HVGs)

        self.save_model_path = save_model_path

    
    def train_model(self,
                    model, 
                    optimizer, 
                    lr_scheduler, 
                    loss_module, 
                    device, 
                    out_path, 
                    train_loader, 
                    val_loader, 
                    fold_idx, 
                    print_rate, 
                    num_epochs, 
                    earlystopping_threshold):
        """
        Don't use this function by itself! It's aimed to be used in the CV_train function.
        """

        print()
        print(f"Start Training CV fold {fold_idx+1}")
        print()

        # Add model to device
        model.to(device)

        # Initiate EarlyStopping
        early_stopping = EarlyStopping(earlystopping_threshold)

        # Training loop
        best_val_loss = np.inf  
        best_val_acc = 0.0
        best_preds = []
        train_start = time.time()
        for epoch in tqdm(range(num_epochs)):

            # Training
            model.train()
            train_loss = []
            all_train_preds = []
            all_train_labels = []
            for data_inputs, data_labels in train_loader:

                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                optimizer.zero_grad()
                preds = model(data_inputs)

                loss = loss_module(preds, data_labels)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                all_train_preds.extend(preds.cpu().detach().numpy())
                all_train_labels.extend(data_labels.cpu().detach().numpy())

            # Validation
            model.eval()
            val_loss = []
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for data_inputs, data_labels in val_loader:

                    data_inputs = data_inputs.to(device)
                    data_labels = data_labels.to(device)

                    preds = model(data_inputs)

                    loss = loss_module(preds, data_labels)
                    val_loss.append(loss.item())

                    all_preds.extend(preds.cpu().detach().numpy())
                    all_labels.extend(data_labels.cpu().detach().numpy())

            # Metrics
            avg_train_loss = sum(train_loss) / len(train_loss)
            train_accuracy = 0.
            avg_val_loss = sum(val_loss) / len(val_loss)
            accuracy = 0.
            f1 = 0.

            # Check early stopping
            early_stopping(avg_val_loss)

            # Print epoch information
            if (epoch % print_rate == 0) or (epoch == (num_epochs-1)) or (early_stopping.early_stop):
                print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

            # Update learning rate
            lr_scheduler.step()

            # Apply early stopping
            if early_stopping.early_stop:
                print(f"Stopped training using EarlyStopping at epoch {epoch+1}")
                break

            # Save model if performance have improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = accuracy
                best_preds = all_preds
                torch.save(model, f'{out_path}model_CV_{fold_idx+1}.pt')

        print()
        print(f"**Finished training CV fold {fold_idx+1}**")
        print()
        train_end = time.time()
        print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

        return best_val_loss, best_val_acc, best_preds
    
    def train(self, 
                 device: str=None,
                 seed: int=42,
                 folds: int=5,
                 cv: bool=False,
                 batch_size: int=256,
                 attn_embed_dim: int=24,
                 depth: int=4,
                 num_heads: int=1,
                 output_dim: int=100,
                 attn_drop_out: int=0.,
                 proj_drop_out: float=0.3,
                 attn_bias: bool=False,
                 act_layer: torch.nn=nn.Tanh,
                 norm_layer: torch.nn=nn.LayerNorm,
                 loss_with_weights: bool=True,
                 init_temperature: float=0.15,
                 init_lr: float=0.001,
                 lr_scheduler_warmup: int=4,
                 lr_scheduler_maxiters: int=25,
                 print_rate: int=2,
                 epochs: int=20,
                 earlystopping_threshold: int=10):
        """
        Perform cross-validation training on machine learning model.

        Parameters:
        - device (str or None): The device to run the training on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
        - seed (int): Random seed for ensuring reproducibility.
        - folds (int): Number of folds for cross-validation.
        - cv (bool): Whether to do cross-validation (True) or to only just one of the folds (False)
        - batch_size (int): Batch size for data loading during training.
        - attn_embed_dim (int): Dimension of the attention embeddings in the model.
        - depth (int): Depth of the model.
        - num_heads (int): Number of attention heads in the model.
        - output_dim (int): Dimension of the model's output.
        - attn_drop_out (float): Dropout rate applied to attention layers.
        - proj_drop_out (float): Dropout rate applied to projection layers.
        - attn_bias (bool): Whether to include bias terms in attention layers.
        - act_layer (torch.nn): Activation function to be used in the model (default is nn.Tanh).
        - norm_layer (torch.nn): Normalization layer to be used in the model (default is nn.LayerNorm).
        - loss_with_weights (bool): Whether to use weights in the loss function.
        - init_temperature (float): Initial temperature for the loss function.
        - init_lr (float): Initial learning rate for the optimizer.
        - lr_scheduler_warmup (int): Number of warm-up iterations for the learning rate scheduler.
        - lr_scheduler_maxiters (int): Maximum number of iterations for the learning rate scheduler.
        - print_rate (int): Rate at which training progress is printed.
        - epochs (int): Number of training epochs.
        - earlystopping_threshold (int): Early stopping threshold.

        Returns:
        - all_preds: List of predictions.
        - all_preds_indices: List of indices of predictions.
        """

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Ensure reproducibility
        def rep_seed(seed):
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        rep_seed(seed)

        # StratifiedKFold cross-validation
        cv_losses = []
        cv_accs = []
        all_preds = []
        all_preds_indices = []
        self.models = []
        total_train_start = time.time()
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(self.data_env.X, self.data_env.labels)):

            train_data = data.Subset(self.data_env, train_indices)
            val_data = data.Subset(self.data_env, val_indices)

            train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = data.DataLoader(val_data, batch_size=batch_size)

            model = CellType2VecModel(input_dim=self.data_env.X.shape[1],
                                        attn_embed_dim=attn_embed_dim, 
                                        depth=depth,
                                        num_heads=num_heads,
                                        output_dim=output_dim,
                                        attn_drop_out=attn_drop_out,
                                        attn_bias=attn_bias,
                                        act_layer=act_layer,
                                        norm_layer=norm_layer,
                                        proj_drop_out=proj_drop_out)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {total_params}")

            loss_module = SNNLoss(use_weights=loss_with_weights, targets=torch.tensor(train_data.dataset.target), temperature=init_temperature)
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
            optimizer = optim.Adam([{'params': model.parameters(), 'lr': init_lr}, {'params': loss_module.parameters(), 'lr': init_lr}], weight_decay=5e-5)
            lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)
            out_path = self.save_model_path#"trained_models/cl_dummy_models1/"

            cv_loss, cv_acc, preds = self.train_model(model=model, 
                                        optimizer=optimizer, 
                                        lr_scheduler=lr_scheduler, 
                                        loss_module=loss_module, 
                                        device=device, 
                                        out_path=out_path,
                                        train_loader=train_loader, 
                                        val_loader=val_loader,
                                        fold_idx=fold_idx,
                                        print_rate=print_rate,
                                        num_epochs=epochs, 
                                        earlystopping_threshold=earlystopping_threshold)
            
            cv_losses.append(cv_loss)
            cv_accs.append(cv_acc)
            all_preds.extend(preds)
            all_preds_indices.extend(val_indices)

            self.models.append(model)

            del model, loss_module, optimizer, lr_scheduler

            if cv == False:
                break 

        print()
        print(f"CV loss score: {np.mean(cv_losses):.4f}")
        print(f"CV accuracy score: {np.mean(cv_accs):.4f}")
        print()

        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")

        return all_preds, all_preds_indices
    
    
    def predict(self, data_, batch_size: int=32, device: str=None):

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        all_preds = []
        for model in self.models:
            model.to(device)

            data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

            preds = []
            #all_labels = []
            for data_inputs, data_labels in data_loader:

                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                pred = model(data_inputs)

                preds.extend(pred.cpu().detach().numpy())
                #all_labels.extend(data_labels.cpu().detach().numpy())

            all_preds.append(preds)

        all_preds = np.mean(all_preds, axis=0)

        return all_preds



