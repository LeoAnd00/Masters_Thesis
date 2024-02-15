import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


def predict(data_, 
            model_name: str,
            model_path: str, 
            model: nn.Module, 
            model_classifier: nn.Module=None,
            batch_size: int=32, 
            device: str=None, 
            return_attention: bool=False,
            use_classifier: bool=False,
            detect_unknowns: bool=True,
            unknown_threshold: float=0.5):
    """
    Generate latent represntations for data using the trained model.
    Note: data_.X must contain the normalized data, as is also required for training.

    Parameters
    ----------
    data_ : AnnData
        An AnnData object containing data for prediction.
    model_path : str
        The path to the directory where the trained model is saved.
    model : nn.Module
        If the model is saved as torch.save(model.state_dict(), f'{out_path}model.pt') one have to input a instance of the model. If torch.save(model, f'{out_path}model.pt') was used then leave this as None (default is None).
    batch_size : int, optional
        Batch size for data loading during prediction (default is 32).
    device : str or None, optional
        The device to run the prediction on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
    use_classifier: bool, optional
        Whether to make cell type annotation predictions or generate latent space (defualt is False).
    unknown_threshold: float, optional
        Threshold for condfidence where if the confidence is less the data is labeled as unknown (defualt is 0.5).
        
    Returns
    -------
    preds : np.array
        Array of predicted latent embeddings.
    """

    data_ = prep_test_data(data_, model_path)

    if device is not None:
        device = device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.load_state_dict(torch.load(f'{model_path}model.pt'))
    # To run on multiple GPUs:
    #if torch.cuda.device_count() > 1:
    #    model= nn.DataParallel(model)
    model.to(device)

    if use_classifier:
        if model_classifier == None:
            raise ValueError('model_classifier needs to be defined if use_classifier=True')
        
        model_classifier.load_state_dict(torch.load(f'{model_path}model_classifier.pt'))
        
        model_classifier.to(device)

    data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

    # Define gene2vec_tensor if gene2ve is used
    if os.path.exists(f"{model_path}/ModelMetadata/gene2vec_tensor.pt"):
        #gene2vec_tensor_ref = torch.load(f"{model_path}/ModelMetadata/gene2vec_tensor.pt")
        gene2vec_tensor = torch.load(f"{model_path}/ModelMetadata/gene2vec_tensor.pt")
        #if torch.cuda.device_count() > 1:
        #    for i in range(1, torch.cuda.device_count()):
        #        gene2vec_tensor = torch.cat((gene2vec_tensor, gene2vec_tensor_ref), dim=0)
        gene2vec_tensor = gene2vec_tensor.to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        for data_inputs, data_not_tokenized in data_loader:

            data_inputs = data_inputs.to(device)
            data_not_tokenized = data_not_tokenized.to(device)

            if model_name == "Model3":
                if os.path.exists(f"{model_path}/ModelMetadata/gene2vec_tensor.pt"):
                    pred = model(data_inputs, data_not_tokenized, gene2vec_tensor)
                else:
                    pred = model(data_inputs, data_not_tokenized)
            elif model_name == "Model2":
                if os.path.exists(f"{model_path}/ModelMetadata/gene2vec_tensor.pt"):
                    pred = model(data_inputs, gene2vec_tensor)
                else:
                    pred = model(data_inputs)
            elif model_name == "Model1":
                pred = model(data_inputs)

            if use_classifier:
                preds_latent = pred.cpu().detach().to(device)
                pred = model_classifier(preds_latent)

            pred = pred.cpu().detach().numpy()

            # Ensure all tensors have at least two dimensions
            if pred.ndim == 1:
                pred = np.expand_dims(pred, axis=0)  # Add a dimension along axis 0

            preds.extend(pred)

    if use_classifier:

        if os.path.exists(f"{model_path}/ModelMetadata/onehot_label_encoder.pt"):
            label_encoder = torch.load(f"{model_path}/ModelMetadata/label_encoder.pt")
            onehot_label_encoder = torch.load(f"{model_path}/ModelMetadata/onehot_label_encoder.pt")
        else:
            raise ValueError("There's no files containing target encodings (label_encoder.pt and onehot_label_encoder.pt).")

        preds = np.array(preds)

        binary_preds = []
        pred_prob = []
        # Loop through the predictions
        for pred in preds:
            # Apply thresholding
            binary_pred = np.where(pred == np.max(pred), 1, 0)

            # Check if the max probability is below the threshold
            if (np.max(pred) < unknown_threshold) and detect_unknowns:
                # If below the threshold, classify as "unknown"
                binary_pred = np.zeros_like(binary_pred) - 1  # Set all elements to -1 to denote "unknown"

            binary_preds.append(binary_pred)
            pred_prob.append(float(pred[binary_pred==1]))

        # Convert the list of arrays to a numpy array
        binary_preds = np.array(binary_preds)

        # Reverse transform the labels
        labels = []
        for row in binary_preds:
            if np.all(row == -1):
                labels.append("Unknown")
            else:
                temp = onehot_label_encoder.inverse_transform(row.reshape(1, -1))
                labels.append(label_encoder.inverse_transform(temp))
        #labels = np.array(labels)
        labels = np.array([np.ravel(label)[0] for label in labels])

        return labels, pred_prob

    else:
        return np.array(preds)

class prep_test_data(data.Dataset):
    """
    PyTorch Dataset for preparing test data for the machine learning model.

    Parameters:
        adata : AnnData
            An AnnData object containing single-cell RNA sequencing data.
        model_path 
            Path to where model is saved.

    Methods:
        __len__()
            Returns the number of data samples.

        __getitem__(idx) 
            Retrieves a specific data sample by index.

        bucketize_expression_levels(expression_levels, num_buckets)
            Bucketize expression levels into categories based on the specified number of buckets and absolute min/max values.

        bucketize_expression_levels_per_gene(expression_levels, num_buckets)
            Bucketize expression levels into categories based on the specified number of buckets and min/max values of each individual gene.
    """

    def __init__(self, adata, model_path):

        # HVG gene names
        hvg_genes = torch.load(f"{model_path}/ModelMetadata/hvg_genes.pt")

        self.adata = adata
        self.adata = self.adata[:, hvg_genes].copy()

        self.X = self.adata.X
        self.X = torch.tensor(self.X)

        # Gene set information
        if os.path.exists(f"{model_path}/ModelMetadata/gene_set_mask.pt"):
            self.pathway_mask = torch.load(f"{model_path}/ModelMetadata/hvg_genes.pt")

        # HVG buckets thresholds
        if os.path.exists(f"{model_path}/ModelMetadata/all_bucketization_threshold_values.pt"):
            self.use_HVG_buckets = True
            self.all_thresholds_values = torch.load(f"{model_path}/ModelMetadata/all_bucketization_threshold_values.pt")

            self.X_not_tokenized = self.X.clone()
            self.X = self.bucketize_expression_levels_per_gene(self.X, self.all_thresholds_values) 
            
            # If value is above max value during tokenization, it will be put in a new bucket the model hasn't seen.
            # To fix this we simply put the value into the previous bucket, which the model has seen.
            # This make it possible to make predictions on new data outside of the training values range.
            if torch.max(self.X) == (len(self.all_thresholds_values[0])-1):
                # Mask where the specified value is located
                mask = self.X == len(self.all_thresholds_values)

                # Replace the specified value with the new value
                self.X[mask] = len(self.all_thresholds_values) - 1
        else:
            self.use_HVG_buckets = False

    def bucketize_expression_levels_per_gene(self, expression_levels, all_thresholds_values):
        """
        Bucketize expression levels into categories based on specified number of buckets and min/max values of each individual gene.

        Parameters
        ----------
        expression_levels : Tensor
            Should be the expression levels (adata.X, or in this case self.X).

        num_buckets : int
            Number of buckets to create.

        Returns
        ----------
        bucketized_levels : LongTensor
            Bucketized expression levels.
        """
        # Apply bucketization to each gene independently
        bucketized_levels = torch.zeros_like(expression_levels, dtype=torch.long)

        # Generate continuous thresholds
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            bucketized_levels[:, i] = torch.bucketize(gene_levels, all_thresholds_values[i])

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)

    def __len__(self):
        """
        Get the number of data samples in the dataset.

        Returns
        ----------
        int: The number of data samples.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get a specific data sample by index.

        Parameters
        ----------
        idx (int): Index of the data sample to retrieve.

        Returns
        ----------
        tuple: A tuple containing the data point and pathways.
        """

        data_point = self.X[idx]

        if self.use_HVG_buckets == True:
            data_not_tokenized = self.X_not_tokenized[idx] 
        else:
            data_not_tokenized = torch.tensor([])

        return data_point, data_not_tokenized

