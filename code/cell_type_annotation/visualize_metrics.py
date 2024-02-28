import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from IPython.display import display
import seaborn as sns


class VisualizeEnv():

    def __init__(self):
        self.color_dict = None

    def read_csv(self, file: list):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        file : list, optional
            file path of the CSV files to read (don't add .csv at the end, it automatically does this).

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """

        self.metrics = pd.read_csv(f'{file}.csv', index_col=0)
        self.metrics.columns = ["Method", 
                                "Accuracy", 
                                "Balanced Accuracy",
                                "F1 Score",
                                "Dataset",
                                "Fold"]

    def BarPlotVisualization(self, dataset_name: str, image_path: str=None, version: int=1):
        """
        Generate a bar plot visualization for each metric, displaying the mean values
        with error bars representing standard deviation across different model types.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
        version : int, optional
            Which plot option to chose (Options: 1, 2, 3)
            
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()

        metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 3
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5 * 1), sharey=False)

        # Get unique model types in order of performance on Overall metric
        metrics_temp = metrics.groupby(['Method'])["Accuracy"].agg(['mean', 'std']).reset_index()
        metrics_temp = metrics_temp.sort_values(by='mean')
        unique_model_types = metrics_temp['Method'].unique()

        # Define a colormap based on unique model types
        cmap = cm.get_cmap('tab20', len(unique_model_types))

        # Map each model type to a color using the colormap
        if self.color_dict is None:
            self.color_dict = {model_type: cmap(1 - j / (len(unique_model_types) - 1)) for j, model_type in enumerate(unique_model_types)}

        columns_metrics = self.metrics.columns[1:4].to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            row_idx = 0
            col_idx = i % ncols

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics.groupby(['Method'])[metric].agg(['mean', 'std']).reset_index()
            visual_metrics = visual_metrics.sort_values(by='mean')

            # Map the colors to the model types in the sorted order
            colors = visual_metrics['Method'].map(self.color_dict)

            # Plot horizontal bars for each model_type in the specified subplot
            if version == 1:
                axs[col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', color=colors, capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 2:
                axs[col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='blue', capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 3:
                axs[col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='none', capsize=3, alpha=0.7, height=0.4, zorder=2)

            # Set labels and title for each subplot
            axs[col_idx].set_xlabel(metric)
            #axs[col_idx].set_title(metric)

            # Ensure y-axis is visible for each subplot
            axs[col_idx].tick_params(left=True)

            # Add grid
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1)

            # Set x-axis limit to 1.1
            axs[col_idx].set_xlim(right=1.1)

        # Set common ylabel for the leftmost subplot in each row
        axs[0].set_ylabel('Method')
        axs[1].set_title(dataset_name)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()

    def BoxPlotVisualization(self, image_path: str=None):
        """
        Generate a bar plot visualization for each metric, displaying the mean values
        with error bars representing standard deviation across different model types.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
            
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()

        metrics['Method'][metrics['Method'] == "Model1"] = "Model1_HVGs"
        metrics['Method'][metrics['Method'] == "TOSICA"] = "TOSICA_HVGs"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)

        columns_metrics = self.metrics.columns[1:4].to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["Model1_HVGs", 
                         "scNym", 
                         "scNym_HVGs", 
                         "Seurat", 
                         "Seurat_HVGs", 
                         "TOSICA_HVGs", 
                         "SciBet", 
                         "SciBet_HVGs", 
                         "CellID_cell", 
                         "CellID_group", 
                         "CellID_cell_HVGs", 
                         "CellID_group_HVGs"]
            
            #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
            sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
            if col_idx != 1:
                axs[col_idx].legend().remove()

            """if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                
                sns.move_legend(
                    axs[col_idx], "lower center",
                    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False,
                )
            else:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                axs[col_idx].legend().remove()"""

        sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.75), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()