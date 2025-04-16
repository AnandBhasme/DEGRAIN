import cv2
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

def trimmed_average_filter(image, kernel_size=3, d=2):
    """
    Apply trimmed average filtering on an image.
    Excludes d highest and d lowest values in the neighborhood.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
        d: Number of highest and lowest values to exclude
    
    Returns:
        Filtered image
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create output image
    filtered_image = np.zeros_like(image)
    
    # Calculate padding size
    pad = kernel_size // 2
    
    # Process each channel separately for color images
    for c in range(image.shape[2]):
        # Get current channel
        channel = image[:, :, c]
        
        # Pad the image
        padded = cv2.copyMakeBorder(channel, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        # Process each pixel
        total_pixels = h * w
        processed_pixels = 0
        last_progress = 0
        
        for i in range(pad, h + pad):
            for j in range(pad, w + pad):
                # Get neighborhood
                neighborhood = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                
                # Flatten and sort the neighborhood
                sorted_values = np.sort(neighborhood.flatten())
                
                # Remove d highest and d lowest values
                trimmed = sorted_values[d:-d]
                
                # Calculate mean of remaining values
                filtered_image[i-pad, j-pad, c] = np.mean(trimmed)
                
                # Update progress
                processed_pixels += 1
                progress = (processed_pixels / total_pixels) * 100
                
                # Print progress every 5%
                if progress - last_progress >= 5:
                    print(f"Channel {c+1}/3: {progress:.1f}% complete", end='\r')
                    last_progress = progress
        
        print(f"Channel {c+1}/3: 100.0% complete")
    
    return filtered_image.astype(np.uint8)

def main():
    # Define paths
    noisy_images_dir = Path("Images/Noisy_Images")
    output_dir = Path("Images/Restored_Images/trimmed_average")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    image_files = list(noisy_images_dir.glob("*.jpg"))
    total_images = len(image_files)
    
    print(f"\nStarting processing of {total_images} images at {datetime.now().strftime('%H:%M:%S')}")
    
    # Process each image in the noisy images directory
    for idx, image_path in enumerate(image_files, 1):
        start_time = time.time()
        print(f"\nProcessing image {idx}/{total_images}: {image_path.name}")
        
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            continue
            
        # Apply trimmed average filter
        print("Applying trimmed average filter...")
        filtered_image = trimmed_average_filter(image)
        
        # Create output filename
        output_path = output_dir / f"trimmed_avg_{image_path.name}"
        
        # Save the filtered image
        cv2.imwrite(str(output_path), filtered_image)
        
        # Calculate and display processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Completed {image_path.name} in {processing_time:.2f} seconds")
        
        # Estimate remaining time
        remaining_images = total_images - idx
        if remaining_images > 0:
            avg_time_per_image = (time.time() - start_time) / idx
            estimated_remaining = avg_time_per_image * remaining_images
            print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
    
    print(f"\nAll {total_images} images processed successfully!")
    print(f"Completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
