import os
import pandas as pd
import re

def parse_metrics_file(file_path):
    """Parse a metrics summary file and return a dictionary of metrics."""
    metrics = {}
    current_image = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is an image name line
            if line.startswith('Image:'):
                current_image = line.split(':')[1].strip()
                metrics[current_image] = {}
                
            # Parse metric values
            elif current_image and ':' in line:
                metric_name, value = line.split(':')
                metric_name = metric_name.strip()
                value = float(value.strip())
                metrics[current_image][metric_name] = value
                
    return metrics

def main():
    # Base directory containing all restoration methods
    base_dir = 'Images/Restored_Images'
    
    # Initialize list to store all metrics
    all_metrics = []
    
    # Get all restoration method directories
    restoration_methods = [d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d))]
    
    for method in restoration_methods:
        metrics_file = os.path.join(base_dir, method, 'metrics_summary.txt')
        if not os.path.exists(metrics_file):
            continue
            
        # Parse metrics for this method
        method_metrics = parse_metrics_file(metrics_file)
        
        # Process each image's metrics
        for image_name, metrics in method_metrics.items():
            # Extract noise type from image name
            noise_type = image_name.replace(f'{method}_', '')
            
            # Create row data
            row = {
                'Restoration_Method': method,
                'Noise_Type': noise_type,
                **metrics
            }
            all_metrics.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for better readability
    columns = ['Restoration_Method', 'Noise_Type', 'MSE', 'PSNR', 'NMSE', 'NAE', 'SSIM', 'UIQI']
    df = df[columns]
    
    # Save to CSV
    df.to_csv('Data/metrics.csv', index=False)
    print("Metrics have been successfully aggregated and saved to metrics.csv")

if __name__ == "__main__":
    main()
