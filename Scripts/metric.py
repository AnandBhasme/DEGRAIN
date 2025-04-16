import cv2
import numpy as np
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import pandas as pd

def calculate_metrics(original, processed):
    """
    Calculate various image quality metrics between original and processed images.
    """
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    
    metrics = {}
    
    # Mean Squared Error (MSE)
    metrics['MSE'] = mean_squared_error(original, processed)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    metrics['PSNR'] = peak_signal_noise_ratio(original, processed, data_range=255)
    
    # Normalized Mean Squared Error (NMSE)
    metrics['NMSE'] = np.sum((original - processed) ** 2) / np.sum(original ** 2)
    
    # Normalized Absolute Error (NAE)
    metrics['NAE'] = np.sum(np.abs(original - processed)) / np.sum(np.abs(original))
    
    # Structural Similarity Index (SSIM)
    ssim_values = []
    for i in range(3):
        ssim_value = ssim(original[:, :, i], processed[:, :, i], data_range=255)
        ssim_values.append(ssim_value)
    metrics['SSIM'] = np.mean(ssim_values)
    
    # Universal Image Quality Index (UIQI)
    def uiqi(x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)
        cov = np.cov(x.flatten(), y.flatten())[0, 1]
        return (4 * cov * x_mean * y_mean) / ((x_std**2 + y_std**2) * (x_mean**2 + y_mean**2))
    
    uiqi_values = []
    for i in range(3):
        uiqi_value = uiqi(original[:, :, i], processed[:, :, i])
        uiqi_values.append(uiqi_value)
    metrics['UIQI'] = np.mean(uiqi_values)
    
    return metrics

def save_metrics_to_file(metrics_data, output_path):
    """
    Save all metrics to a single text file.
    """
    with open(output_path, 'w') as f:
        f.write("Image Quality Metrics Summary\n")
        f.write("===========================\n\n")
        
        for image_name, metrics in metrics_data.items():
            f.write(f"Image: {image_name}\n")
            f.write("-" * (len(image_name) + 7) + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
            f.write("\n")

def create_visualizations(metrics_data, output_dir, method_name):
    """
    Create visualizations for the metrics.
    """
    # Convert metrics data to DataFrame for easier plotting
    df = pd.DataFrame(metrics_data).T
    
    # Create directory for visualizations if it doesn't exist
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Create bar plots for each metric
    metrics = ['MSE', 'PSNR', 'NMSE', 'NAE', 'SSIM', 'UIQI']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        df[metric].plot(kind='bar')
        plt.title(f'{metric} Comparison - {method_name}')
        plt.xlabel('Images')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(vis_dir / f'{metric.lower()}_comparison.png')
        plt.close()
    
    # Create radar chart for all metrics
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Number of metrics
    N = len(metrics)
    
    # Angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot each image's metrics
    for image_name in df.index:
        values = df.loc[image_name].values
        values = np.append(values, values[0])
        ax.plot(angles, values, linewidth=2, label=image_name)
        ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    plt.xticks(angles[:-1], metrics)
    plt.title(f'Metrics Radar Chart - {method_name}', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(vis_dir / 'metrics_radar_chart.png')
    plt.close()

def main():
    # Define paths
    original_image_path = Path("Images/Original_Image.jpg")
    noisy_images_dir = Path("Images/Noisy_Images")
    restored_images_dir = Path("Images/Restored_Images")
    
    # Read original image
    original_image = cv2.imread(str(original_image_path))
    if original_image is None:
        print("Error: Could not read original image")
        return
    
    # Process noisy images
    print("Processing noisy images...")
    noisy_metrics = {}
    for image_path in noisy_images_dir.glob("*.jpg"):
        noisy_image = cv2.imread(str(image_path))
        if noisy_image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        metrics = calculate_metrics(original_image, noisy_image)
        noisy_metrics[image_path.stem] = metrics
    
    # Save and visualize noisy images metrics
    save_metrics_to_file(noisy_metrics, noisy_images_dir / "metrics_summary.txt")
    create_visualizations(noisy_metrics, noisy_images_dir, "Noisy Images")
    
    # Process restored images
    print("\nProcessing restored images...")
    for method_dir in restored_images_dir.iterdir():
        if not method_dir.is_dir():
            continue
            
        print(f"\nProcessing {method_dir.name}...")
        restored_metrics = {}
        
        for image_path in method_dir.glob("*.jpg"):
            restored_image = cv2.imread(str(image_path))
            if restored_image is None:
                print(f"Error: Could not read image {image_path}")
                continue
            
            metrics = calculate_metrics(original_image, restored_image)
            restored_metrics[image_path.stem] = metrics
        
        # Save and visualize restored images metrics
        save_metrics_to_file(restored_metrics, method_dir / "metrics_summary.txt")
        create_visualizations(restored_metrics, method_dir, method_dir.name)
    
    print("\nAll metrics have been calculated and saved successfully!")

if __name__ == "__main__":
    main()