import cv2
import numpy as np
import os
from pathlib import Path

def median_filter(image, kernel_size=3):
    """
    Apply median filtering on an image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
    
    Returns:
        Filtered image
    """
    # Apply median filter to each channel
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image

def main():
    # Define paths
    noisy_images_dir = Path("Images/Noisy_Images")
    output_dir = Path("Images/Restored_Images/median")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image in the noisy images directory
    for image_path in noisy_images_dir.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
            
        # Apply median filter
        filtered_image = median_filter(image)
        
        # Create output filename
        output_path = output_dir / f"median_{image_path.name}"
        
        # Save the filtered image
        cv2.imwrite(str(output_path), filtered_image)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
