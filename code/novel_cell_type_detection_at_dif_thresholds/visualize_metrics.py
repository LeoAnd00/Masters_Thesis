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
                                "Fold",
                                "Excluded Cell Type",
                                "Threshold"]

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

        metrics['Method'][metrics['Method'] == "Model1"] = "Model1 |HVGs"
        metrics['Excluded Cell Type'][metrics['Excluded Cell Type'] == "Mature_B_Cells"] = "Mature B Cells"
        metrics['Excluded Cell Type'][metrics['Excluded Cell Type'] == "gamma-delta_T_Cells_1"] = "Gamma-delta T Cells 1"
        metrics['Excluded Cell Type'][metrics['Excluded Cell Type'] == "alpha-beta_T_Cells"] = "Alpha-beta T Cells"
        metrics['Excluded Cell Type'][metrics['Excluded Cell Type'] == "Plasma_Cells"] = "Plasma Cells"
        metrics['Excluded Cell Type'][metrics['Excluded Cell Type'] == "All_Above"] = "All Above"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 3
        nrows = len(self.metrics["Excluded Cell Type"].unique())
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), sharey=False)

        columns_metrics = self.metrics.columns[1:4].to_list()
        excluded_cell_types = metrics["Excluded Cell Type"].unique()

        for idx, novel_cell in enumerate(excluded_cell_types):
            row_idx = idx % nrows
        
            for i, metric in enumerate(columns_metrics):
                # Calculate the row and column indices
                col_idx = i % ncols

                # Group by model type, calculate mean and std, and sort by mean value of the current metric
                visual_metrics = metrics[['Dataset',"NovelOrNot",'Method',"Excluded Cell Type",'Threshold',metric]]

                visual_metrics = visual_metrics[visual_metrics['Excluded Cell Type'] == novel_cell]

                axs[row_idx,col_idx].set_ylabel(metric)
                axs[row_idx,col_idx].set_xlabel('Threshold')
                variable = visual_metrics[metric].to_list()
                group = visual_metrics['Threshold'].to_list()
                group2 = visual_metrics['NovelOrNot'].to_list()
                hue_order = ["Novel", "Known"]

                if col_idx == 1:
                    sns.boxplot(y = variable,
                            x = group,
                            hue = group2, 
                            width = 0.6,
                            linewidth=0.4,
                            #order=order,
                            hue_order = hue_order,
                            ax=axs[row_idx,col_idx], 
                            showfliers = False)
                    
                    # Add grid
                    # Calculate the x positions of the grid lines to be between the ticks
                    x_ticks = axs[row_idx,col_idx].get_xticks()
                    x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

                    # Set the grid positions to be between the x ticks
                    axs[row_idx,col_idx].set_xticks(x_grid_positions, minor=True)

                    # Add grid lines between the x positions
                    axs[row_idx,col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

                    # Rotate x-axis labels by 60 degrees
                    axs[row_idx,col_idx].set_xticklabels(axs[row_idx,col_idx].get_xticklabels(), rotation=60)
                    
                    sns.move_legend(
                        axs[row_idx,col_idx], "lower center",
                        bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False,
                    )

                    axs[row_idx,col_idx].set_title(f"Excluded Cell: {novel_cell}", y=1.2)
                else:
                    sns.boxplot(y = variable,
                            x = group,
                            hue = group2, 
                            width = 0.6,
                            linewidth=0.4,
                            #order=order,
                            hue_order = hue_order,
                            ax=axs[row_idx,col_idx], 
                            showfliers = False)
                    
                    # Add grid
                    # Calculate the x positions of the grid lines to be between the ticks
                    x_ticks = axs[row_idx,col_idx].get_xticks()
                    x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

                    # Set the grid positions to be between the x ticks
                    axs[row_idx,col_idx].set_xticks(x_grid_positions, minor=True)

                    # Add grid lines between the x positions
                    axs[row_idx,col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')
                    
                    axs[row_idx,col_idx].legend().remove()

                    # Rotate x-axis labels by 60 degrees
                    axs[row_idx,col_idx].set_xticklabels(axs[row_idx,col_idx].get_xticklabels(), rotation=60)
            
                # Set x-axis limit to 1.1
                axs[row_idx,col_idx].set_ylim(top=1.0)
                axs[row_idx,col_idx].set_ylim(bottom=0.0)

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.8), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()
