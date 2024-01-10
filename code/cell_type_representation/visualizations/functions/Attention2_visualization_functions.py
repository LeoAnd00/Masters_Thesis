
import scanpy as sc
from functions import train as trainer
from functions import data_preprocessing as dp
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import scib
from scipy.spatial.distance import pdist, squareform
#from models import model2_tokenized_hvg_transformer as model2_tokenized_hvg_transformer
from models import model_ITSCR as model_ITSCR

class VisualizeEnv():

    def __init__(self):
        pass

    def MakeAttentionMatrix(self, 
                            attention_matrix_or_just_env: bool,
                            train_path: str, 
                            pred_path: str, 
                            gene2vec_path: str,
                            model_path: str,
                            pathway_path: str,
                            target_key: str,
                            batch_key: str,
                            HVG_attn_or_pathway_attn: str,
                            HVGs: int=2000, 
                            HVG_buckets_: int=1000,
                            batch_size: int=32):
        
        self.label_key = target_key
        self.train_adata = sc.read(train_path, cache=True)
        self.train_adata.obs["batch"] = self.train_adata.obs[batch_key]

        self.train_env = trainer.train_module(data_path=self.train_adata,
                                        pathways_file_path=pathway_path,
                                        num_pathways=300,
                                        pathway_gene_limit=20,
                                        save_model_path="",
                                        HVG=True,
                                        HVGs=HVGs,
                                        HVG_buckets=HVG_buckets_,
                                        use_HVG_buckets=True,
                                        Scaled=False,
                                        target_key=target_key,
                                        batch_keys=["batch"],
                                        use_gene2vec_emb=True,
                                        gene2vec_path=gene2vec_path)
        
        self.data_env = self.train_env.data_env
        
        self.pred_adata = sc.read(pred_path, cache=True)

        if HVG_attn_or_pathway_attn == "pathway":
            self.data_env.hvg_genes = self.data_env.pathway_names

        if attention_matrix_or_just_env:

            # Define model
            model = model_ITSCR.ITSCR_main_model(mask=self.data_env.pathway_mask,
                                                num_HVGs=min([HVGs,int(self.data_env.X.shape[1])]),
                                                output_dim=100,
                                                HVG_tokens=HVG_buckets_,
                                                HVG_embedding_dim=self.data_env.gene2vec_tensor.shape[1],
                                                use_gene2vec_emb=True)
            
            self.attention_matrices = self.predict(data_=self.pred_adata, model=model, model_path=model_path, batch_size=batch_size, HVG_attn_or_pathway_attn=HVG_attn_or_pathway_attn)
        
    def predict(self, data_, model_path: str, model=None, batch_size: int=32, device: str=None, HVG_attn_or_pathway_attn: str="HVG"):
        """
        Generate latent represntations for data using the trained model.

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

        Returns
        -------
        attn_matrices : np.array
            Matrix containing the averaged attention matrix of each cell type cluster in latent space.
        """

        data_ = prep_test_data(data_, self.data_env)

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model == None:
            model = torch.load(f'{model_path}.pt')
        else:
            model.load_state_dict(torch.load(f'{model_path}.pt'))
        # To run on multiple GPUs:
        if torch.cuda.device_count() > 1:
            model= nn.DataParallel(model)
        model.to(device)

        data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

        # Define gene2vec_tensor if gene2ve is used
        if self.data_env.use_gene2vec_emb:
            gene2vec_tensor = self.data_env.gene2vec_tensor
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    gene2vec_tensor = torch.cat((gene2vec_tensor, self.data_env.gene2vec_tensor), dim=0)
            gene2vec_tensor = gene2vec_tensor.to(device)
            #gene2vec_tensor = self.data_env.gene2vec_tensor.to(device)

        # Make dictionary for the attention matrices
        labels = data_.adata.obs[self.data_env.target_key]
        target_counts = labels.value_counts()
        unique_targets = list(set(labels))

        # Specify the size of the matrix
        matrix_size = (len(self.data_env.hvg_genes), len(self.data_env.hvg_genes)) 
        # Create a dictionary with unique targets as keys and matrices as values
        attn_matrices = {target: torch.zeros(matrix_size) for target in unique_targets}
        # Create a matrix containing the attention latent space representation for each sample
        attn_latent_space_matrix = []

        if HVG_attn_or_pathway_attn == "HVG":
            return_attention = True
            return_pathway_attention = False
        elif HVG_attn_or_pathway_attn == "pathway":
            return_attention = False
            return_pathway_attention = True

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            counter = 0
            for data_inputs, data_labels, data_pathways in tqdm(data_loader):
                counter += 1

                data_inputs = data_inputs.to(device)
                data_labels = list(data_labels)
                data_pathways = data_pathways.to(device)

                latent_space, attn_matrix = model(data_inputs, data_pathways, gene2vec_tensor, return_attention=return_attention, return_pathway_attention=return_pathway_attention)

                pred = attn_matrix.cpu().detach().numpy()

                # Ensure all tensors have at least two dimensions
                if pred.ndim == 1:
                    pred = np.expand_dims(pred, axis=0)  # Add a dimension along axis 0

                preds.extend(pred)
                labels.extend(data_labels)

                if counter == 5000:
                    break

        preds = np.array(preds)
        preds_adata = sc.AnnData(X=preds)
        if return_pathway_attention:
            self.data_env.hvg_genes = self.data_env.pathway_names
        preds_adata.index = self.data_env.hvg_genes
        preds_adata.obs["cell_type"] = labels

        return preds_adata
    
    def GetCellTypeNames(self):
        print(list(self.attention_matrices.keys()))

    def DownloadAttentionMatrixDictionary(self, name: str="AttetionMatrixDictionary"):

        # Download
        self.attention_matrices.write(f"attention_matrices/{name}.h5ad")

    def LoadAttentionMatrixDictionary(self, name: str="AttetionMatrixDictionary"):
    
        # Load the dictionary back from the .npz file
        self.attention_matrices = sc.read(f"attention_matrices/{name}.h5ad", cache=True)

    def RankGenes(self, save_path: str=None):
        adata = self.attention_matrices.copy()
        adata.var_names = self.data_env.hvg_genes

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Count the occurrences of each cell type in the 'cell_type' variable
        cell_type_counts = adata.obs["cell_type"].value_counts()

        # Extract cell types with counts greater than or equal to 2
        cell_types_to_keep = cell_type_counts[cell_type_counts >= 2].index

        # Filter samples based on the selected cell types
        adata = adata[adata.obs["cell_type"].isin(cell_types_to_keep),:].copy()

        sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')
        
        
        # Extract information from adata.uns
        result_dict = adata.uns['rank_genes_groups']
        gene_names = result_dict['names']
        gene_pvals = result_dict['pvals']
        gene_scores = result_dict['scores']

        # Create an empty DataFrame to store the results
        dc_pathway = pd.DataFrame()

        # Iterate over each column (cell type) in gene_names
        for cell_type in pd.DataFrame(gene_names).columns:
            # Create a DataFrame for the current cell type
            cell_type_df = pd.DataFrame({
                'Gene': gene_names[cell_type],
                'P-value': gene_pvals[cell_type],
                'Score': gene_scores[cell_type]
            })
            
            # Add a column indicating the cell type
            cell_type_df['Cell_Type'] = cell_type

            print(cell_type_df.head(10))
            
            # Concatenate the current cell type DataFrame to the overall DataFrame
            dc_pathway = pd.concat([dc_pathway, cell_type_df])

        # Reset index
        dc_pathway.reset_index(drop=True, inplace=True)

        if save_path is not None:
            dc_pathway.to_csv(f'{save_path}.csv', index=False)

    def RankOriginalGenes(self):
        adata = self.pred_adata[:, self.data_env.hvg_genes].copy()
        # Count the occurrences of each cell type in the 'cell_type' variable
        cell_type_counts = adata.obs["cell_type"].value_counts()

        # Extract cell types with counts greater than or equal to 2
        cell_types_to_keep = cell_type_counts[cell_type_counts >= 2].index

        # Filter samples based on the selected cell types
        adata = adata[adata.obs["cell_type"].isin(cell_types_to_keep),:].copy()

        sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')
        dc_pathway = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
        print(dc_pathway.head(10))

    def ScatterPlotGeneSetOrHVG(self, x_axis: str, y_axis: str, color: str, size: int=50, save_path: str=None):
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=10, format='svg')

        adata = self.attention_matrices.copy()

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        df_temp = pd.DataFrame(adata.X, index=adata.obs[color], columns=self.data_env.hvg_genes)
                
        adata.obs[x_axis] = df_temp[x_axis].values
        adata.obs[y_axis] = df_temp[y_axis].values
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sc.pl.scatter(adata, 
                      x=x_axis, 
                      y=y_axis, 
                      color=color, 
                      size=size, 
                      title='',
                      use_raw=False, 
                      ax=ax, 
                      show=False)
        # Remove gridlines
        ax.grid(False)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as SVG
        if save_path is not None:
            plt.savefig(f"{save_path}.svg")

        # Show the plot
        plt.show()
        
    def ViolinPlotGeneSetOrHVG(self, name: str, color: str, save_path: str=None, color_map: str="viridis", rotation: int=90):

        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=10, format='svg')

        adata = self.attention_matrices.copy()

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        df_temp = pd.DataFrame(adata.X, index=adata.obs[color], columns=self.data_env.hvg_genes)
                
        adata.obs[name] = df_temp[name].values

        # Get the number of unique categories in 'cell_type'
        num_categories = len(adata.obs[color].unique())

        # Set up a color palette based on the number of unique categories
        self.palette = sns.color_palette(color_map, num_categories)

        # Set the order of categories for the x-axis
        order = list(adata.obs.groupby(color)[name].median().sort_values(ascending=True).index)

        # Set the category order for plotting
        adata.obs[color] = pd.Categorical(adata.obs[color], categories=order, ordered=True)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sc.pl.violin(adata, [f'{name}'], groupby=color,use_raw=False, rotation=rotation, palette=self.palette, ax=ax, show=False)

        # Remove gridlines
        ax.grid(False)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as SVG
        if save_path is not None:
            plt.savefig(f"{save_path}.svg")

        # Show the plot
        plt.show()

    def StackedViolinPlotGeneSetOrHVG(self, color: str, num_top_selections: int=5, save_path: str=None):
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=10, format='svg')

        adata = self.attention_matrices.copy()
        adata.var_names = self.data_env.hvg_genes

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')

        #sc.pl.stacked_violin(adata, names, groupby=color, dendrogram=False)
        if save_path is not None:
            sc.pl.rank_genes_groups_stacked_violin(adata, 
                                                n_genes=num_top_selections, 
                                                key="rank_genes_groups", 
                                                groupby=color,
                                                save=f"{save_path}.svg")
        else:
            sc.pl.rank_genes_groups_stacked_violin(adata, 
                                                n_genes=num_top_selections, 
                                                key="rank_genes_groups", 
                                                groupby=color)
            
    def DendogramPlotGeneSetOrHVG(self, color: str, save_path: str=None):
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=10, format='svg')

        adata = self.attention_matrices.copy()
        adata.var_names = self.data_env.hvg_genes

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Convert 'cell_type' to a categorical variable
        adata.obs[color] = adata.obs[color].astype('category')

        sc.tl.dendrogram(adata, color)
        sc.pl.dendrogram(adata, color, orientation='left', save=f"{save_path}.svg")









    
    def HeatMapVisualization(self, cell_type: str, num_of_top_genes: int = 50):
        def select_positions_based_on_sum(attention_matrix, num_positions=50):
            # Calculate row and column sums
            row_sums = np.sum(attention_matrix, axis=1) / np.sum(attention_matrix)
            col_sums = np.sum(attention_matrix, axis=0) / np.sum(attention_matrix)

            # Get indices of the top rows and columns
            top_row_indices = np.argsort(row_sums)[-num_positions:]
            top_col_indices = np.argsort(col_sums)[-num_positions:]

            gene_names_row = self.data_env.hvg_genes[top_row_indices]
            row_sums = row_sums[top_row_indices]
            gene_names_col = self.data_env.hvg_genes[top_col_indices]
            col_sums = col_sums[top_col_indices]

            return gene_names_row, row_sums, gene_names_col, col_sums

        gene_names_row, row_sums, gene_names_col, col_sums = select_positions_based_on_sum(self.attention_matrices[cell_type], num_positions=num_of_top_genes)

        # Create a dataframe for seaborn
        df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

        # Select the top genes
        df = df.loc[gene_names_col, gene_names_col]

        # Visualize the heatmap with seaborn
        plt.figure(figsize=(24, 20))
        sns.heatmap(df, cmap='plasma', annot=False, linewidths=.5, fmt=".2f", cbar_kws={'label': 'Attention'})

        # Set axis titles
        plt.xlabel("Gene")
        plt.ylabel("Gene")

        plt.title(f'Attention Matrix Heatmap of Top {num_of_top_genes} Genes for {cell_type}')
        plt.show()

    def BarPlotVisualization(self, cell_type: str="Classical Monocytes", row_or_col: str="col", num_of_top_genes: int = 50, color: str="green"):
        matrix = self.attention_matrices[self.attention_matrices.obs["cell_type"] == cell_type,:]

        mean_score = np.mean(matrix.X, axis=0)
        top_indices = np.argsort(mean_score)[-num_of_top_genes:]
        mean_score = mean_score[top_indices]

        # Gene names
        gene_names = self.data_env.hvg_genes[top_indices]

        # Plot horizontal bar plots for relative sums
        fig, axes = plt.subplots(1, 1, figsize=(8, int(num_of_top_genes*3/10)))

        axes.barh(gene_names, mean_score, color=color)
        axes.set_title('Mean Attention Score of Gene')
        axes.set_xlabel('Attention Score')
        axes.set_ylabel('Gene')

        plt.tight_layout()
        plt.show()

    def RawExpressionsBarPlotVisualization(self, cell_type: str="Classical Monocytes", row_or_col: str="col", num_of_top_genes: int = 50, color: str="green"):
        matrix = self.pred_adata[:, self.data_env.hvg_genes].copy()
        matrix = matrix[matrix.obs["cell_type"] == cell_type,:]

        mean_score = np.mean(matrix.X, axis=0)
        top_indices = np.argsort(mean_score)[-num_of_top_genes:]
        mean_score = mean_score[top_indices]

        # Gene names
        gene_names = self.data_env.hvg_genes[top_indices]

        # Plot horizontal bar plots for relative sums
        fig, axes = plt.subplots(1, 1, figsize=(8, int(num_of_top_genes*3/10)))

        axes.barh(gene_names, mean_score, color=color)
        axes.set_title('Raw Expression of Gene')
        axes.set_xlabel('Normalized Expression Value')
        axes.set_ylabel('Gene')

        plt.tight_layout()
        plt.show()

    def BarPlotOfIndividualGenes(self, cell_type: str="Classical Monocytes", gene: str="LYZ", num_of_top_genes: int = 50, color: str="blue"):

        df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

        # Select the top genes
        df = df.loc[gene, :] / np.sum(df.loc[gene, :])

        top_indices = np.argsort(df)[-num_of_top_genes:]

        df = df[top_indices]
        gene_names= self.data_env.hvg_genes[top_indices]

        # Plot horizontal bar plots for relative sums
        fig, axes = plt.subplots(1, 1, figsize=(8, int(num_of_top_genes*3/10)))

        axes.barh(gene_names, df, color=color)
        axes.set_title(f'Attention between {gene} and top {num_of_top_genes} genes with highest attention score')
        axes.set_xlabel('Relative attention')
        axes.set_ylabel('Gene')

        plt.tight_layout()
        plt.show()

    def VisualizeCellTypeCorrelations(self):
        
        data = self.attention_matrices
        
        # Make a dictionary containing the vector representations of each cell type where each dimesnion is a gene.
        cell_type_vectors = {}
        for cell_type in list(data.keys()):

            matrix = data[cell_type]

            # Calculate the sum of each column
            col = np.sum(matrix, axis=0)

            # Calculate the relative sum (percentage) for each column
            relative_sums = col / np.sum(col)

            cell_type_vectors[cell_type] = relative_sums

        # Calculate the eucledian distance between all cell type vectors to create a distance matrix
        cell_types = list(cell_type_vectors.keys())
        vectors = [cell_type_vectors[cell_type] for cell_type in cell_types]

        # Calculate pairwise Euclidean distances
        distance_matrix = squareform(pdist(vectors, metric='euclidean'))

        # Create a heatmap to visualize the distance matrix
        plt.figure(figsize=(20, 16))
        heatmap = plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Euclidean Distance')
        plt.xticks(range(len(cell_types)), cell_types, rotation=45, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(cell_types)), cell_types)
        plt.title('Cell Type Correlations')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    def GeneVsCellTypeVisualization(self, num_of_top_genes: int = 20):
        
        df_all = []
        all_top_indices = []
        for cell_type in list(self.attention_matrices.keys()):

            df = pd.DataFrame(data=self.attention_matrices[cell_type], index=self.data_env.hvg_genes, columns=self.data_env.hvg_genes)

            col = np.sum(df, axis=0)
            
            relative_sums = col / np.sum(col)

            if cell_type == list(self.attention_matrices.keys())[0]:
                df_all = pd.DataFrame(data=relative_sums, index=self.data_env.hvg_genes, columns=[cell_type])
            else:
                temp = pd.DataFrame(data=relative_sums, index=self.data_env.hvg_genes, columns=[cell_type])
                df_all = pd.concat([df_all, temp], axis=1)

            top_indices = np.argsort(relative_sums)[-num_of_top_genes:]
            all_top_indices.extend(top_indices)

        unique_top_indices = list(set(all_top_indices))
        df = df_all.iloc[unique_top_indices,:]
        gene_names= df.index.tolist()
        cell_types = df.columns.tolist()

        print(f"Number of genes: {len(gene_names)}")

        # Create a heatmap to visualize 
        plt.figure(figsize=(int(2*num_of_top_genes), 26))
        heatmap = plt.imshow(df, cmap='viridis', interpolation='nearest')

        plt.colorbar(heatmap, label='Relative attention')
        plt.xticks(range(len(cell_types)), cell_types, rotation=45, ha='right')  # Adjust rotation and alignment
        plt.yticks(range(len(gene_names)), gene_names)
        plt.title('Cell Type Correlations to Genes')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()


class prep_test_data(data.Dataset):
    """
    PyTorch Dataset for preparing test data for the machine learning model.

    Parameters:
        adata : AnnData
            An AnnData object containing single-cell RNA sequencing data.
        prep_data_env 
            The data environment used for preprocessing training data.

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

    def __init__(self, adata, prep_data_env):
        self.adata = adata
        self.adata = self.adata[:, prep_data_env.hvg_genes].copy()
        if prep_data_env.scaled:
            self.adata.X = dp.scale_data(data=self.adata.X, feature_means=prep_data_env.feature_means, feature_stdevs=prep_data_env.feature_stdevs)

        self.X = self.adata.X
        self.X = torch.tensor(self.X)

        self.pathways_file_path = prep_data_env.pathways_file_path
        self.use_HVG_buckets = prep_data_env.use_HVG_buckets

        self.labels = self.adata.obs[prep_data_env.target_key]
        self.target = self.labels #prep_data_env.label_encoder.transform(self.labels)

        # Pathway information
        if self.pathways_file_path is not None:
            self.pathway_mask = prep_data_env.pathway_mask
        
        if prep_data_env.use_HVG_buckets:
            self.training_expression_levels = prep_data_env.X_not_tokenized
            self.X_not_tokenized = self.X.clone()
            #self.X = self.bucketize_expression_levels(self.X, prep_data_env.HVG_buckets) 
            self.X = self.bucketize_expression_levels_per_gene(self.X, prep_data_env.HVG_buckets) 

    def bucketize_expression_levels(self, expression_levels, num_buckets):
        """
        Bucketize expression levels into categories based on specified number of buckets and absolut min/max values.

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
        eps = 1e-6
        thresholds = torch.linspace(torch.min(self.training_expression_levels) - eps, torch.max(self.training_expression_levels) + eps, steps=num_buckets + 1)

        bucketized_levels = torch.bucketize(expression_levels, thresholds)

        bucketized_levels -= 1

        return bucketized_levels.to(torch.long)
    
    def bucketize_expression_levels_per_gene(self, expression_levels, num_buckets):
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
        eps = 1e-6
        min_values, _ = torch.min(self.training_expression_levels, dim=0)
        max_values, _ = torch.max(self.training_expression_levels, dim=0)
        for i in range(expression_levels.size(1)):
            gene_levels = expression_levels[:, i]
            min_scalar = min_values[i].item()
            max_scalar = max_values[i].item()
            thresholds = torch.linspace(min_scalar - eps, max_scalar + eps, steps=num_buckets + 1)
            bucketized_levels[:, i] = torch.bucketize(gene_levels, thresholds)

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

        # Get labels
        data_label = self.target[idx]

        #if (self.use_HVG_buckets == True) and (self.pathways_file_path is not None):
        #    data_pathways = self.X_not_tokenized[idx] * self.pathway_mask
        #elif (self.use_HVG_buckets == True) and (self.pathways_file_path is None):
        #    data_pathways = self.X_not_tokenized[idx] 
        #elif self.pathways_file_path is not None:
        #    data_pathways = self.X[idx] * self.pathway_mask
        #else:
        #    data_pathways = torch.tensor([])
        data_pathways = self.X_not_tokenized[idx] 

        return data_point, data_label, data_pathways

