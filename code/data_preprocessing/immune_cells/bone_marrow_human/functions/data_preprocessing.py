from scipy.io import mmread
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings


def read_sc_data(count_data: str, gene_data: str, barcode_data: str):

    #Load data
    data = sc.read(count_data, cache=True).transpose()
    data.X = data.X.toarray()

    # Load genes and barcodes
    genes = pd.read_csv(gene_data, sep='\t', header=None)
    barcodes = pd.read_csv(barcode_data, sep='\t', header=None)

    # set genes
    genes.rename(columns={0:'gene_id', 1:'gene_symbol'}, inplace=True)
    genes.set_index('gene_symbol', inplace=True)
    data.var = genes

    # set barcodes
    barcodes.rename(columns={0:'barcode'}, inplace=True)
    barcodes.set_index('barcode', inplace=True)
    data.obs = barcodes

    return data

def log1p_normalize(data):
    data.layers["counts"] = data.X.copy()

    # Calculate size factor
    L = data.X.sum() / data.shape[0]
    data.obs["size_factors"] = data.X.sum(1) / L

    # Normalize using shifted logarithm (log1p)
    scaled_counts = data.X / data.obs["size_factors"].values[:,None]
    data.layers["log1p_counts"] = np.log1p(scaled_counts)

    data.X = data.layers["log1p_counts"]

    return data

class QC():

    def __init__(self):
        pass

    def median_absolute_deviation(self, data):
        """
        Calculate the Median Absolute Deviation (MAD) of a dataset.

        Parameters:
        data (list or numpy.ndarray): The dataset for which MAD is calculated.

        Returns:
        float: The Median Absolute Deviation (MAD) of the dataset.
        """
        median = np.median(data)
        
        absolute_differences = np.abs(data - median)

        mad = np.median(absolute_differences)
        
        return mad

    def QC_metric_calc(self, data):

        # Sum of counts per cell
        data.obs['n_counts'] = data.X.sum(1)
        # Shifted log of n_counts
        data.obs['log_n_counts'] = np.log(data.obs['n_counts']+1)
        # Number of unique genes per cell
        data.obs['n_genes'] = (data.X > 0).sum(1)
        # Shifted lof og n_genes
        data.obs['log_n_genes'] = np.log(data.obs['n_genes']+1)

        # Fraction of total counts among the top 20 genes with highest counts
        top_20_indices = np.argpartition(data.X, -20, axis=1)[:, -20:]
        top_20_values = np.take_along_axis(data.X, top_20_indices, axis=1)
        data.obs['pct_counts_in_top_20_genes'] = (np.sum(top_20_values, axis=1)/data.obs['n_counts'])

        # Fraction of mitochondial counts
        mt_gene_mask = [gene.startswith('MT-') for gene in data.var_names]
        data.obs['mt_frac'] = data.X[:, mt_gene_mask].sum(1)/data.obs['n_counts']

        return data

    def MAD_based_outlier(self, data, metric: str, threshold: int = 5):
        data_metric = data.obs[metric]
        # calculate indexes where outliers are detected
        outlier = (data_metric < np.median(data_metric) - threshold * self.median_absolute_deviation(data_metric)) | (
                    np.median(data_metric) + threshold * self.median_absolute_deviation(data_metric) < data_metric)
        return outlier

    def QC_filter_outliers(self, data, threshold: int = 5):

        # Ignore the specific FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        data.obs["outlier"] = (self.MAD_based_outlier(data, "log_n_counts", threshold)
            | self.MAD_based_outlier(data, "log_n_genes", threshold)
            | self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold)
            | self.MAD_based_outlier(data, "mt_frac", threshold)
        )

        # Print how many detected outliers by each QC metric 
        outlier1 = (self.MAD_based_outlier(data, "log_n_genes", threshold))
        outlier2 = (self.MAD_based_outlier(data, "log_n_counts", threshold))
        outlier3 = (self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold))
        outlier4 = (self.MAD_based_outlier(data, "mt_frac", threshold))
        print(f"Number of cells before QC filtering: {data.n_obs}")
        print(f"Number of cells removed by log_n_genes filtering: {sum(1 for item in outlier1 if item)}")
        print(f"Number of cells removed by log_n_counts filtering: {sum(1 for item in outlier2 if item)}")
        print(f"Number of cells removed by pct_counts_in_top_20_genes filtering: {sum(1 for item in outlier3 if item)}")
        print(f"Number of cells removed by mt_frac filtering: {sum(1 for item in outlier4 if item)}")
        
        # Filter away outliers
        data = data[(~data.obs.outlier)].copy()
        print(f"Number of cells post QC filtering: {data.n_obs}")

        #Filter genes:
        print('Number of genes before filtering: {:d}'.format(data.n_vars))

        # Min 20 cells - filters out 0 count genes
        sc.pp.filter_genes(data, min_cells=20)
        print('Number of genes after filtering so theres min 20 unique cells per gene: {:d}'.format(data.n_vars))

        return data
    
class EDA():

    def __init__(self):
        pass

    def ViolinJitter(self, data, y_rows: list, title: str = "Violin Plots", subtitle: list = ["Unfiltered", "QC Filtered"]):
        """
        Create a violin plot and box plot with jitter on top for specified columns.

        Parameters:
        data (list): List containing adata.
        y_rows (str): The column to be plotted on the y-axis.
        title (str): The title of the plot.
        subtitle (list): List of titles for each adata in data for the plot.
        """

        # Ignore the specific FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Create a figure and axis for the plot
        num_y_columns = len(y_rows)

        # Get random colors
        colors = sns.color_palette("husl",len(y_rows))#[random.choice(sns.color_palette("husl")) for _ in range(num_y_columns)] #sns.color_palette("husl",2)
        
        # Create subplots for each Y column
        fig, axes = plt.subplots(num_y_columns, len(data), figsize=(10,3*num_y_columns))

        for n, k in enumerate(data):
            for i, (y_column, color) in enumerate(zip(y_rows, colors)):
                ax = axes[i,n]
            
                # Create a violin plot
                sns.violinplot(data=k.obs[y_column], ax=ax, color=color)
                
                # Calculate the scaling factor based on the violin plot width
                scale_factor = 0.5 * (max(ax.collections[0].get_paths()[0].vertices[:, 0]) - min(ax.collections[0].get_paths()[0].vertices[:, 0]))
                
                # Adjust jitter points to the same width as the violin plot distribution
                sns.stripplot(data=k.obs[y_column], color='black', jitter=scale_factor, alpha=1.0, size=1, ax=ax)

                # Set subplot title
                ax.set_title(subtitle[n])
        
        
        # Set overall plot title
        fig.suptitle(title, y=1.0)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        plt.show()

    def ScatterForQCMetrics(self, data, title:str = "Scatter Plot", subtitle:list = ["Unfiltered", "QC Filtered"]):
        """
        Create a scatter plot with specified x and y columns and color the points based on a color column.

        Parameters:
        data (list): List containing adata.
        title (str): The title of the scatter plot (optional).
        subtitle (list): List of titles for each adata in data for the plot.
        """

        # Ignore the specific FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        x_column=['n_counts','n_counts','mt_frac','mt_frac']
        y_column=['n_genes','n_genes','pct_counts_in_top_20_genes','pct_counts_in_top_20_genes']
        color_column=['mt_frac','pct_counts_in_top_20_genes','n_genes','n_counts']
        
        # Create a scatter plot with continuous colors using Matplotlib's pyplot
        fig, axes = plt.subplots(len(x_column), len(data), figsize=(16,6*len(x_column)))

        for n, k in enumerate(data):
            for i in range(len(x_column)):
                ax = axes[i, n]
                scatter = ax.scatter(k.obs[x_column[i]], k.obs[y_column[i]], c=k.obs[color_column[i]], cmap='coolwarm')

                # Add a colorbar to display the color scale
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_column[i])  # Set the colorbar label

                ax.set_xlabel(x_column[i])
                ax.set_ylabel(y_column[i])

                # Set subplot title
                ax.set_title(subtitle[n])
        
        # Set overall plot title
        fig.suptitle(title, y=1.0)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        # Show the plot
        plt.show()

    def VisualizeNormalization(self,data):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        p1 = sns.histplot(data.obs["n_counts"], bins=100, kde=False, ax=axes[0])
        axes[0].set_title("Raw counts")
        axes[0].set_xlabel("Sum of counts")

        p2 = sns.histplot(data.layers["log1p_counts"].sum(1), bins=100, kde=False, ax=axes[1])
        axes[1].set_xlabel("Sum of Normalized counts")
        axes[1].set_title("log1p normalised counts")

        plt.show()




        
