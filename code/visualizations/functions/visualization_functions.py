import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import display


class VisualizeEnv():

    def __init__(self):
        pass

    def read_csv(self, files: list):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        files : list, optional
            List of file paths of the CSV files to read (don't add .csv at the end, it automatically does this).

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """

        all_dataframes = []
        for file in files:
            metrics = pd.read_csv(f'{file}.csv', index_col=0)
            all_dataframes.append(metrics)

        self.metrics = pd.concat(all_dataframes, axis=0)
        self.metrics.columns = ["ASW | Bio", 
                                "ASW | Batch", 
                                "PCR | Batch",
                                "Isolated Label Silhouette | Bio",
                                "GC | Batch",
                                "NMI | Bio",
                                "ARI | Bio",
                                "CC | Bio",
                                "Isolated Label F1 | Bio",
                                "Overall | Batch",
                                "Overall | Bio",
                                "Overall"]

    def BarPlotVisualization(self, image_path: str=None):
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
        metrics['Model Type'] = self.metrics.index

        # Set up the figure and axis with 4 columns per row
        ncols = 3
        nrows = -(-len(self.metrics.columns) // ncols)  # Ceiling division
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharey=False)

        # Get unique model types
        unique_model_types = metrics['Model Type'].unique()

        # Define a colormap based on unique model types
        cmap = cm.get_cmap('tab20', len(unique_model_types))

        for i, metric in enumerate(reversed(self.metrics.columns)):
            # Calculate the row and column indices
            row_idx = i // ncols
            col_idx = i % ncols

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics.groupby(['Model Type'])[metric].agg(['mean', 'std']).reset_index()
            visual_metrics = visual_metrics.sort_values(by='mean')

            # Map each model type to a color using the colormap
            color_dict = {model_type: cmap(j / (len(unique_model_types) - 1)) for j, model_type in enumerate(unique_model_types)}

            # Map the colors to the model types in the sorted order
            colors = visual_metrics['Model Type'].map(color_dict)

            # Plot horizontal bars for each model_type in the specified subplot
            #axs[row_idx, col_idx].barh(visual_metrics['Model Type'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='none', capsize=3, alpha=0.7, height=0.4)
            axs[row_idx, col_idx].barh(visual_metrics['Model Type'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', color=colors, capsize=3, alpha=1.0, height=0.4, zorder=2)

            # Set labels and title for each subplot
            axs[row_idx, col_idx].set_xlabel('Score')
            axs[row_idx, col_idx].set_title(metric)

            # Ensure y-axis is visible for each subplot
            axs[row_idx, col_idx].tick_params(left=True)

            # Add grid
            axs[row_idx, col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1)

        # Set common ylabel for the leftmost subplot in each row
        for ax in axs[:, 0]:
            ax.set_ylabel('Method')

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()

    def visualize_results(self, bg_color: str="Blues"):
        """
        Visualizes the performance metrics dataframe using a colored heatmap.

        Parameters
        ----------
        bg_color : str, optional
            The colormap for the heatmap (default is "Blues").

        Returns
        -------
        None
        """
        metrics = self.metrics.copy()
        metrics['Model Type'] = self.metrics.index
        metrics = metrics.groupby(['Model Type']).agg(['mean', 'std']).reset_index()
        metrics = metrics.sort_values(('Overall', 'mean'), ascending=False)
       
        styled_metrics = metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

