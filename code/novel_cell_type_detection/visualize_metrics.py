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
                                "NovelOrNot",
                                "Fold"]

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
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows), sharey=False)

        columns_metrics = self.metrics.columns[1:4].to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset',"NovelOrNot",'Method',metric]]

            axs[col_idx].set_ylabel(metric)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Method'].to_list()
            group2 = visual_metrics['NovelOrNot'].to_list()
            hue_order = ["Novel", "Known"]
            order = ["Model1_HVGs", 
                         "scNym", 
                         "scNym_HVGs", 
                         "Seurat", 
                         "Seurat_HVGs", 
                         "TOSICA_HVGs"]
            
            #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
            """sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        order=order,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
            if col_idx != 1:
                axs[col_idx].legend().remove()

            # Add grid
            # Calculate the x positions of the grid lines to be between the ticks
            x_ticks = axs[col_idx].get_xticks()
            x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

            # Set the grid positions to be between the x ticks
            axs[col_idx].set_xticks(x_grid_positions, minor=True)

            # Add grid lines between the x positions
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')"""

            if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        order=order,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                
                # Add grid
                # Calculate the x positions of the grid lines to be between the ticks
                x_ticks = axs[col_idx].get_xticks()
                x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

                # Set the grid positions to be between the x ticks
                axs[col_idx].set_xticks(x_grid_positions, minor=True)

                # Add grid lines between the x positions
                axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

                # Rotate x-axis labels by 60 degrees
                axs[col_idx].set_xticklabels(axs[col_idx].get_xticklabels(), rotation=60)
                
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
                        order=order,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                
                # Add grid
                # Calculate the x positions of the grid lines to be between the ticks
                x_ticks = axs[col_idx].get_xticks()
                x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

                # Set the grid positions to be between the x ticks
                axs[col_idx].set_xticks(x_grid_positions, minor=True)

                # Add grid lines between the x positions
                axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')
                
                axs[col_idx].legend().remove()

                # Rotate x-axis labels by 60 degrees
                axs[col_idx].set_xticklabels(axs[col_idx].get_xticklabels(), rotation=60)
        
            # Set x-axis limit to 1.1
            axs[col_idx].set_ylim(top=1.0)
            axs[col_idx].set_ylim(bottom=0.0)

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.8), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()
