import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import display
import seaborn as sns


class VisualizeEnv():

    def __init__(self):
        self.color_dict = None

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
            metrics = metrics.iloc[:,:-3]
            all_dataframes.append(metrics)

        self.metrics = pd.concat(all_dataframes, axis=0)
        self.metrics.columns = ["ASW | Bio", 
                                "ASW | Batch", 
                                "PCR | Batch",
                                "Isolated Label ASW | Bio",
                                "GC | Batch",
                                "NMI | Bio",
                                "ARI | Bio",
                                "CC | Bio",
                                "Isolated Label F1 | Bio",
                                "Overall Score | Batch",
                                "Overall Score | Bio",
                                "Overall Score"]

    def BarPlotVisualization_Appendix(self, image_path: str=None, version: int=1):
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
        metrics['Method'] = self.metrics.index
        # Replace model names
        metrics['Method'][metrics['Method'] == "Model1"] = "Model 1"

        # Set up the figure and axis with 4 columns per row
        ncols = 3
        nrows = -(-len(self.metrics.columns) // ncols)  # Ceiling division
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharey=False)

        # Get unique model types in order of performance on Overall metric
        metrics_temp = metrics.groupby(['Method'])["Overall Score"].agg(['mean', 'std']).reset_index()
        metrics_temp = metrics_temp.sort_values(by='mean')
        unique_model_types = metrics_temp['Method'].unique()

        # Define a colormap based on unique model types
        cmap = cm.get_cmap('tab20', len(unique_model_types))

        # Map each model type to a color using the colormap
        if self.color_dict is None:
            self.color_dict = {model_type: cmap(1 - j / (len(unique_model_types) - 1)) for j, model_type in enumerate(unique_model_types)}


        for i, metric in enumerate(reversed(self.metrics.columns)):
            # Calculate the row and column indices
            row_idx = i // ncols
            col_idx = i % ncols

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics.groupby(['Method'])[metric].agg(['mean', 'std']).reset_index()
            visual_metrics = visual_metrics.sort_values(by='mean')

            # Map the colors to the model types in the sorted order
            colors = visual_metrics['Method'].map(self.color_dict)

            # Plot horizontal bars for each model_type in the specified subplot
            if version == 1:
                axs[row_idx, col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', color=colors, capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 2:
                axs[row_idx, col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='blue', capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 3:
                axs[row_idx, col_idx].barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='none', capsize=3, alpha=0.7, height=0.4, zorder=2)

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

        metrics['Method'] = self.metrics.index
        # Replace model names
        metrics['Method'][metrics['Method'] == "Model1"] = "Model 1"

        metrics = metrics.loc[:,["Method", "Overall Score"]]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5 * 1), sharey=False)

        # Get unique model types in order of performance on Overall metric
        metrics_temp = metrics.groupby(['Method'])["Overall Score"].agg(['mean', 'std']).reset_index()
        metrics_temp = metrics_temp.sort_values(by='mean')
        unique_model_types = metrics_temp['Method'].unique()

        # Define a colormap based on unique model types
        cmap = cm.get_cmap('tab20', len(unique_model_types))

        # Map each model type to a color using the colormap
        if self.color_dict is None:
            self.color_dict = {model_type: cmap(1 - j / (len(unique_model_types) - 1)) for j, model_type in enumerate(unique_model_types)}

        columns_metrics = [metrics.columns[-1]]#.to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            row_idx = 0
            col_idx = 0#i % ncols

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics.groupby(['Method'])[metric].agg(['mean', 'std']).reset_index()
            visual_metrics = visual_metrics.sort_values(by='mean')

            # Map the colors to the model types in the sorted order
            colors = visual_metrics['Method'].map(self.color_dict)

            # Plot horizontal bars for each model_type in the specified subplot
            if version == 1:
                axs.barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', color=colors, capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 2:
                axs.barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='blue', capsize=3, alpha=1.0, height=0.4, zorder=2)
            elif version == 3:
                axs.barh(visual_metrics['Method'], visual_metrics['mean'], xerr=visual_metrics['std'], edgecolor='black', facecolor='none', capsize=3, alpha=0.7, height=0.4, zorder=2)

            # Set labels and title for each subplot
            axs.set_xlabel(metric)
            #axs[col_idx].set_title(metric)

            # Ensure y-axis is visible for each subplot
            axs.tick_params(left=True)

            # Add grid
            axs.grid(axis='x', linestyle='--', alpha=1.0, zorder=1)

            # Set x-axis limit to 1.1
            axs.set_xlim(right=1.1)

        # Set common ylabel for the leftmost subplot in each row
        axs.set_ylabel('Method')
        axs.set_title(dataset_name)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')

        plt.show()

    def BoxPlotVisualization(self, files, dataset_names, image_path: str=None):
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

        all_dataframes = []
        for file, dataset_name in zip(files,dataset_names):
            metrics = pd.read_csv(f'{file}.csv', index_col=0)
            metrics = metrics.iloc[:,:-3]
            metrics["Dataset"] = [dataset_name]*metrics.shape[0]
            all_dataframes.append(metrics)

        metrics = pd.concat(all_dataframes, axis=0)
        metrics.columns = ["ASW | Bio", 
                            "ASW | Batch", 
                            "PCR | Batch",
                            "Isolated Label ASW | Bio",
                            "GC | Batch",
                            "NMI | Bio",
                            "ARI | Bio",
                            "CC | Bio",
                            "Isolated Label F1 | Bio",
                            "Overall Score | Batch",
                            "Overall Score | Bio",
                            "Overall Score",
                            "Dataset"]

        metrics['Method'] = metrics.index

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 5 * nrows), sharey=False)
        axs = [axs]

        columns_metrics = ["Overall Score"]

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["Model1", 
                         "Model1_val",
                         "scGen", 
                         "scANVI", 
                         "scVI", 
                         "TOSICA", 
                         "PCA", 
                         "Unintegrated"]
            
            #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
            """sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
            if col_idx != 1:
                axs[col_idx].legend().remove()"""

            if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False,
                        showmeans=False)
                
                sns.move_legend(
                    axs[col_idx], "lower center",
                    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False, fontsize="small"
                )
            else:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False,
                        showmeans=False)
                axs[col_idx].legend().remove()

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.75), title=None, frameon=False)

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
        # Replace model names
        metrics['Model Type'][metrics['Model Type'] == "Model1"] = "Model 1"
        
        metrics = metrics.groupby(['Model Type']).agg(['mean', 'std']).reset_index()
        metrics = metrics.sort_values(('Overall Score', 'mean'), ascending=False)
       
        styled_metrics = metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

