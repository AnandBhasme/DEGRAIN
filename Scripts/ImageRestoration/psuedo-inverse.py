import cv2
import numpy as np
import os
from pathlib import Path

def create_motion_blur_kernel(size=15):
    """Create a motion blur kernel"""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return kernel

def fix_quadrant_swapping(image):
    """
    Fix the quadrant swapping issue by rearranging the image quadrants.
    Bottom right swaps with top left, and top right swaps with bottom left.
    """
    h, w = image.shape[:2]
    mid_h = h // 2
    mid_w = w // 2
    
    # Split image into quadrants
    top_left = image[:mid_h, :mid_w]
    top_right = image[:mid_h, mid_w:]
    bottom_left = image[mid_h:, :mid_w]
    bottom_right = image[mid_h:, mid_w:]
    
    # Swap quadrants
    # Bottom right -> Top left
    # Top right -> Bottom left
    # Top left -> Bottom right
    # Bottom left -> Top right
    
    # Create new image with swapped quadrants
    new_image = np.zeros_like(image)
    new_image[:mid_h, :mid_w] = bottom_right
    new_image[:mid_h, mid_w:] = bottom_left
    new_image[mid_h:, :mid_w] = top_right
    new_image[mid_h:, mid_w:] = top_left
    
    return new_image

def pseudo_inverse_filter(image, kernel_size=15, threshold=0.1):
    """
    Apply pseudo-inverse filtering to restore motion-blurred images
    
    Args:
        image: Input image
        kernel_size: Size of the motion blur kernel
        threshold: Threshold for pseudo-inverse (to avoid division by very small numbers)
    
    Returns:
        Restored image
    """
    # Convert image to float32 for FFT
    img_float = image.astype(np.float32) / 255.0
    
    # Get image dimensions
    h, w = img_float.shape[:2]
    
    # Create motion blur kernel
    kernel = create_motion_blur_kernel(kernel_size)
    
    # Pad kernel to match image size
    kernel_padded = np.zeros((h, w))
    kernel_padded[:kernel_size, :kernel_size] = kernel
    
    # Shift kernel to center for FFT
    kernel_padded = np.fft.ifftshift(kernel_padded)
    
    # Compute FFT of kernel
    H = np.fft.fft2(kernel_padded)
    
    # Process each channel separately for color images
    restored_channels = []
    for channel in cv2.split(img_float):
        # Compute FFT of image channel
        G = np.fft.fft2(channel)
        
        # Apply pseudo-inverse filter
        # H* is the complex conjugate of H
        H_conj = np.conj(H)
        H_mag_squared = np.abs(H)**2
        
        # Add threshold to avoid division by very small numbers
        F_hat = (H_conj / (H_mag_squared + threshold)) * G
        
        # Inverse FFT
        restored_channel = np.real(np.fft.ifft2(F_hat))
        
        # Normalize
        restored_channel = np.clip(restored_channel, 0, 1)
        restored_channels.append(restored_channel)
    
    # Combine channels
    restored_image = cv2.merge(restored_channels)
    
    # Convert back to uint8
    restored_image = (restored_image * 255).astype(np.uint8)
    
    # Fix quadrant swapping
    restored_image = fix_quadrant_swapping(restored_image)
    
    return restored_image

def main():
    # Define paths
    noisy_images_dir = Path("Images/Noisy_Images")
    output_dir = Path("Images/Restored_Images/pseudo_inverse")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image in the noisy images directory
    for image_path in noisy_images_dir.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
            
        # Apply pseudo-inverse filter
        # Note: This filter works best on motion-blurred images
        restored_image = pseudo_inverse_filter(image)
        
        # Create output filename
        output_path = output_dir / f"pseudo_inverse_{image_path.name}"
        
        # Save the restored image
        cv2.imwrite(str(output_path), restored_image)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
