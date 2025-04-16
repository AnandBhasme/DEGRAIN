import cv2
import numpy as np
import os
from pathlib import Path

def arithmetic_mean_filter(image, kernel_size=3):

    # Create a kernel of ones
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Apply filter to each channel
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

def main():
    # Define paths
    noisy_images_dir = Path("Images/Noisy_Images")
    output_dir = Path("Images/Restored_Images/arithmetic_mean")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image in the noisy images directory
    for image_path in noisy_images_dir.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
            
        # Apply arithmetic mean filter
        filtered_image = arithmetic_mean_filter(image)
        
        # Create output filename
        output_path = output_dir / f"arithmetic_mean_{image_path.name}"
        
        # Save the filtered image
        cv2.imwrite(str(output_path), filtered_image)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
