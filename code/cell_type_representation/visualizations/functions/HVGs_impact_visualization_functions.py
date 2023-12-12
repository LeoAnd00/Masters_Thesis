import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import re
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
            metrics = pd.read_csv(f'{file}.csv', index_col=0)
            metrics["HVGs"] = int(HVGs.group(1))
            all_dataframes.append(metrics)

        self.metrics = pd.concat(all_dataframes, axis=0)
        self.metrics = self.metrics.iloc[:,-2:]
        self.metrics.columns = ["Overall","HVGs"]

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

    def ErrorBarPlot(self, image_path: str=None):

        metrics = self.metrics.copy()
        metrics['Model Type'] = self.metrics.index
        metrics = metrics.groupby(['Model Type','HVGs'])['Overall'].agg(['mean', 'std']).reset_index()

        # Define a colormap
        cmap = cm.get_cmap('viridis', len(metrics['Model Type'].unique()))

        # Create a dictionary to map model types to unique colors
        color_dict = dict(zip(metrics['Model Type'].unique(), [cmap(i) for i in range(len(metrics['Model Type'].unique()))]))

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(15, 9))

        # Plot all model types in the same plot
        for model_type, color in color_dict.items():
            model_df = metrics[metrics['Model Type'].str.contains(model_type)]
            plt.errorbar(
                model_df['HVGs'],
                model_df['mean'],
                yerr=model_df['std'],
                fmt='-o',
                label=model_type,
                color=color,
                markersize=8,
                marker='s',
                capsize=5,  # Adjust the length of the horizontal lines at the top and bottom of the error bars
                capthick=2,      # Adjust the thickness of the horizontal lines
                alpha=0.5,        # Adjust the transparency of the plot
                linewidth=2      # Adjust the thickness of the line connecting the points
            )
        plt.xlabel('Nr. of HVGs')
        plt.ylabel('Overall Metric')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.title('HVG Impact Assessment')

        # Turn off grid lines
        plt.grid(False)

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()