import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def normalize_metric(metric_values):
    """Normalize metric values to [0,1] range"""
    return (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())

def calculate_spunn(metrics_df, weights):
    """Calculate SPUNN index using the given weights and normalize to [0,100] range"""
    # Normalize metrics
    psnr_norm = normalize_metric(metrics_df['PSNR'])
    ssim_norm = metrics_df['SSIM']  # SSIM is already in [0,1]
    uiqi_norm = metrics_df['UIQI']  # UIQI is already in [0,1]
    nmse_norm = normalize_metric(metrics_df['NMSE'])
    nae_norm = normalize_metric(metrics_df['NAE'])
    
    # Calculate individual components
    psnr_score = weights[0] * psnr_norm
    ssim_score = weights[1] * ssim_norm
    uiqi_score = weights[2] * uiqi_norm
    nmse_score = weights[3] * (1 - nmse_norm)  # Invert NMSE since lower is better
    nae_score = weights[4] * (1 - nae_norm)    # Invert NAE since lower is better
    
    # Combine scores
    raw_spunn = psnr_score + ssim_score + uiqi_score + nmse_score + nae_score
    
    # Scale to [0,100] range
    normalized_spunn = raw_spunn * 100
    
    # Ensure bounds
    normalized_spunn = np.clip(normalized_spunn, 0, 100)
    
    return normalized_spunn

def main():
    # Read the metrics data
    metrics_df = pd.read_csv('Data/metrics.csv')
    
    # Prepare data for PCA
    metrics_for_pca = metrics_df[['PSNR', 'SSIM', 'UIQI', 'NMSE', 'NAE']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(metrics_for_pca)
    
    # Perform PCA
    pca = PCA(n_components=1)
    pca.fit(scaled_data)
    
    # Get the weights from the first principal component
    weights = np.abs(pca.components_[0])
    weights = weights / weights.sum()  # Normalize weights to sum to 1
    
    print("Derived weights from PCA:")
    print(f"α (PSNR weight): {weights[0]:.4f}")
    print(f"β (SSIM weight): {weights[1]:.4f}")
    print(f"γ (UIQI weight): {weights[2]:.4f}")
    print(f"δ (NMSE weight): {weights[3]:.4f}")
    print(f"ε (NAE weight): {weights[4]:.4f}")
    
    # Calculate SPUNN index
    metrics_df['SPUNN'] = calculate_spunn(metrics_df, weights)
    
    # Save the updated metrics with SPUNN index
    metrics_df.to_csv('Data/metrics_with_spunn.csv', index=False)
    print("\nMetrics with SPUNN index have been saved to metrics_with_spunn.csv")
    
    # Print SPUNN range and statistics
    print(f"\nSPUNN index statistics:")
    print(f"Min: {metrics_df['SPUNN'].min():.2f}")
    print(f"Max: {metrics_df['SPUNN'].max():.2f}")
    print(f"Mean: {metrics_df['SPUNN'].mean():.2f}")
    print(f"Std: {metrics_df['SPUNN'].std():.2f}")
    
    # Print top 5 methods by SPUNN score
    print("\nTop 5 restoration methods by SPUNN score:")
    top_5 = metrics_df.nlargest(5, 'SPUNN')[['Restoration_Method', 'Noise_Type', 'SPUNN']]
    print(top_5.to_string(index=False))

if __name__ == "__main__":
    main()
