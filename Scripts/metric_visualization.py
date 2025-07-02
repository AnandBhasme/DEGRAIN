import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def create_radar_chart():
    # Read the metrics data
    df = pd.read_csv('Data/metrics_with_spunn.csv')
    
    # Get unique restoration methods and noise types
    restoration_methods = df['Restoration_Method'].unique()
    noise_types = ['exponential_noise', 'gaussian_noise', 'motion_blur', 'rayleigh_noise', 'salt_pepper_noise']
    
    print("\nAvailable Restoration Methods:")
    print(restoration_methods)
    
    # Create a pivot table for the radar chart
    pivot_df = df.pivot(index='Restoration_Method', columns='Noise_Type', values='SPUNN')
    
    # Handle special cases for TVM and trimmed_average
    for method in ['TVM', 'trimmed_average']:
        if method in pivot_df.index:
            # Get the rows for this method
            method_data = df[df['Restoration_Method'] == method]
            # Create a new row with the correct noise types
            new_row = {}
            for noise in noise_types:
                # Find the corresponding row with the prefixed noise type
                if method == 'TVM':
                    prefixed_noise = f'tvm_{noise}'
                else:
                    prefixed_noise = f'trimmed_avg_{noise}'
                # Get the SPUNN value
                value = method_data[method_data['Noise_Type'] == prefixed_noise]['SPUNN'].values
                if len(value) > 0:
                    new_row[noise] = value[0]
            # Update the pivot table
            pivot_df.loc[method] = pd.Series(new_row)
    
    print("\nSPUNN Scores for each method:")
    print(pivot_df)
    
    # Reorder columns to match the desired order
    pivot_df = pivot_df[noise_types]
    
    # Number of variables
    N = len(noise_types)
    
    # Create angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Set up the plot
    plt.xticks(angles[:-1], noise_types)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
    plt.ylim(0, 100)
    
    # Create a color palette with distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each restoration method
    for idx, method in enumerate(restoration_methods):
        print(f"\nPlotting method: {method}")
        values = pivot_df.loc[method].values.flatten().tolist()
        print(f"Values: {values}")
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Add legend
    legend_elements = [Patch(facecolor=colors[i], alpha=0.20, label=method) 
                      for i, method in enumerate(restoration_methods)]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Add title
    plt.title('SPUNN Scores for Different Restoration Methods\nAcross Various Noise Types', 
              size=15, y=1.1)
    
    # Save the plot
    plt.savefig('Data/spunn_radar_chart.png', bbox_inches='tight', dpi=3000)
    print("\nRadar chart has been saved as 'spunn_radar_chart.png'")

if __name__ == "__main__":
    create_radar_chart()
