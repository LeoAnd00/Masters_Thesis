import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import numpy as np
import random
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

        # Define a regular expression pattern to match the number in front of "HVGs"
        pattern = r'benchmark_(\d+)_HVGs'

        all_dataframes = []
        for file in files:
            HVGs = re.search(pattern, file)
            metrics = pd.read_csv(f'{file}.csv', index_col=0).drop_duplicates()
            metrics["HVGs"] = int(HVGs.group(1))
            all_dataframes.append(metrics)

        self.metrics = pd.concat(all_dataframes, axis=0)
        self.metrics = self.metrics.iloc[:,-4:]
        self.metrics.columns = ["Overall Batch","Overall Bio","Overall","HVGs"]

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
        metrics = metrics.groupby(['Model Type','HVGs']).agg(['mean', 'std']).reset_index()
        metrics = metrics.sort_values(('Overall', 'mean'), ascending=False)
       
        styled_metrics = metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

    def ErrorBarPlot(self, image_path: str=None, seed: int=42, metric_to_visualize: str="Overall"):

        # Ensure reproducibility
        def rep_seed(seed):
            np.random.seed(seed)
            random.seed(seed)
            
        rep_seed(seed)

        metrics = self.metrics.copy()
        metrics['Model Type'] = self.metrics.index
        metrics = metrics.groupby(['Model Type','HVGs'])[metric_to_visualize].agg(['mean', 'std']).reset_index()

        # Define a colormap
        cmap = cm.get_cmap('tab20', len(metrics['Model Type'].unique()))

        # Create a dictionary to map model types to unique colors
        color_dict = dict(zip(metrics['Model Type'].unique(), [cmap(i) for i in range(len(metrics['Model Type'].unique()))]))

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(15, 9))

        # Replace 'HVGs' with a sequence of integers for plotting
        HVGs_range = range(1, len(metrics['HVGs'].unique()) + 1)
        metrics['HVGs_temp'] = np.zeros(len(metrics['HVGs']))
        for idx, HVG_range in enumerate(metrics['HVGs'].unique()):
            metrics['HVGs_temp'][metrics['HVGs']==HVG_range] = [HVGs_range[idx]]*len(metrics['HVGs_temp'][metrics['HVGs']==HVG_range])

        # Plot all model types in the same plot
        for model_type, color in color_dict.items():
            model_df = metrics[metrics['Model Type'].str.contains(model_type)]

            # Reset the index
            model_df = model_df.reset_index(drop=True)
            
            # Add jitter to x-coordinates for each individual point
            jittered_x = model_df['HVGs_temp'] + np.random.normal(scale=0.06, size=len(model_df))

            # Plot each data point separately
            for i in range(len(model_df)):
                plt.errorbar(
                    jittered_x[i],
                    model_df['mean'].iloc[i],
                    yerr=model_df['std'].iloc[i],
                    fmt='o',  # Use 'o' for markers only, without lines
                    linestyle='',
                    label=model_type if i == 0 else "",  # Label only the first point for each model type
                    color=color,
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    alpha=1.0,
                    linewidth=2,  # Set linewidth to 0 for markers only
                )

        # Set xticks to only include the desired values
        plt.xticks(range(1, len(metrics['HVGs'].unique()) + 1), metrics['HVGs'].unique())

        #plt.xticks(model_df['HVGs'].unique())

        plt.xlabel('Nr. of HVGs')
        plt.ylabel(f'{metric_to_visualize} Metric')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.title('HVG Impact Assessment')

        # Turn off grid lines
        plt.grid(False)

        # Adjust layout to ensure the x-axis label is not cut off
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()