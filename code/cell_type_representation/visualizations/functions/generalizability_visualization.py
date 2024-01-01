
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

def generalizability_error_bar_plot(csv_path: str, image_path: str):
    # Make a suitable plot to display generalizability
    metrics = pd.read_csv(f'{csv_path}.csv', index_col=0)

    #metrics['Model Type'] = [re.sub(r'\d+$', '', model_string) for model_string in metrics.index]
    metrics['Model Type'] = metrics.index

    # Group by train_num and model type, calculate mean and std
    grouped_df = metrics.groupby(['train_num', 'Model Type'])['Overall'].agg(['mean', 'std']).reset_index()

    # Define a colormap
    cmap = cm.get_cmap('viridis', len(grouped_df['Model Type'].unique()))

    # Create a dictionary to map model types to unique colors
    color_dict = dict(zip(grouped_df['Model Type'].unique(), [cmap(i) for i in range(len(grouped_df['Model Type'].unique()))]))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot all model types in the same plot
    for model_type, color in color_dict.items():
        model_df = grouped_df[grouped_df['Model Type'].str.contains(model_type)]

        # Add jitter to x-coordinates for each individual point
        jittered_x = model_df['train_num'] + np.random.normal(scale=0.1, size=len(model_df))
        jittered_x.reset_index(drop=True, inplace=True) 

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
    plt.xticks(model_df['train_num'].unique())

    plt.xlabel('Nr. of Patients for Training')
    plt.ylabel('Overall Metric')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title('Generalizability Assessment')

    # Turn off grid lines
    plt.grid(False)

    # Adjust layout to ensure the x-axis label is not cut off
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(f'{image_path}.svg', format='svg')

    plt.show()
