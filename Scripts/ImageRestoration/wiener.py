import cv2
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

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

def wiener_filter(image, kernel_size=15, noise_var=0.01):
    """
    Apply Wiener filtering to restore blurred and noisy images.
    
    Args:
        image: Input image
        kernel_size: Size of the motion blur kernel
        noise_var: Noise variance (higher = more smoothing)
    
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
    for c in range(img_float.shape[2]):
        channel = img_float[:, :, c]
        
        # Compute FFT of image channel
        G = np.fft.fft2(channel)
        
        # Compute power spectrum of the blur kernel
        H_mag_squared = np.abs(H)**2
        
        # Apply Wiener filter
        # F_hat = (H* / (|H|² + noise_var)) * G
        H_conj = np.conj(H)
        F_hat = (H_conj / (H_mag_squared + noise_var)) * G
        
        # Inverse FFT
        restored_channel = np.real(np.fft.ifft2(F_hat))
        
        # Normalize
        restored_channel = np.clip(restored_channel, 0, 1)
        restored_channels.append(restored_channel)
        
        print(f"Channel {c+1}/3 processed", end='\r')
    
    print("All channels processed")
    
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
    output_dir = Path("Images/Restored_Images/wiener")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    image_files = list(noisy_images_dir.glob("*.jpg"))
    total_images = len(image_files)
    
    print(f"\nStarting processing of {total_images} images at {datetime.now().strftime('%H:%M:%S')}")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        start_time = time.time()
        print(f"\nProcessing image {idx}/{total_images}: {image_path.name}")
        
        # Read the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            continue
            
        # Apply Wiener filter
        print("Applying Wiener filter...")
        restored_image = wiener_filter(image)
        
        # Create output filename
        output_path = output_dir / f"wiener_{image_path.name}"
        
        # Save the restored image
        cv2.imwrite(str(output_path), restored_image)
        
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
