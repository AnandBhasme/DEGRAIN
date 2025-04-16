import cv2
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

def chambolle_tv_denoise(image, weight=0.1, n_iter=100):
    """
    Apply Total Variation Minimization using Chambolle's algorithm.
    
    Args:
        image: Input image
        weight: Weight of the TV term (higher = more smoothing)
        n_iter: Number of iterations
        
    Returns:
        Denoised image
    """
    # Convert to float32 for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Initialize dual variable
    p = np.zeros((img_float.shape[0], img_float.shape[1], 2))
    
    # Gradient step
    tau = 0.25
    
    # Process each channel separately
    denoised_channels = []
    for c in range(img_float.shape[2]):
        channel = img_float[:, :, c]
        
        # Main iteration loop
        for i in range(n_iter):
            # Compute divergence of p
            div_p = np.zeros_like(channel)
            div_p[1:, :] += p[1:, :, 0] - p[:-1, :, 0]
            div_p[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
            
            # Update primal variable
            u = channel - weight * div_p
            
            # Compute gradient of u
            grad_u = np.zeros((u.shape[0], u.shape[1], 2))
            grad_u[:-1, :, 0] = u[1:, :] - u[:-1, :]
            grad_u[:, :-1, 1] = u[:, 1:] - u[:, :-1]
            
            # Update dual variable
            p = p + tau * grad_u
            norm_p = np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2)
            norm_p = np.maximum(1, norm_p)
            p[:, :, 0] /= norm_p
            p[:, :, 1] /= norm_p
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Channel {c+1}/3: Iteration {i+1}/{n_iter} ({((i+1)/n_iter)*100:.1f}%)", end='\r')
        
        print(f"Channel {c+1}/3: Iteration {n_iter}/{n_iter} (100.0%)")
        denoised_channels.append(u)
    
    # Combine channels
    denoised_image = np.stack(denoised_channels, axis=2)
    
    # Clip and convert back to uint8
    denoised_image = np.clip(denoised_image, 0, 1)
    denoised_image = (denoised_image * 255).astype(np.uint8)
    
    return denoised_image

def main():
    # Define paths
    noisy_images_dir = Path("Images/Noisy_Images")
    output_dir = Path("Images/Restored_Images/TVM")
    
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
            
        # Apply TV minimization
        print("Applying Total Variation Minimization...")
        denoised_image = chambolle_tv_denoise(image)
        
        # Create output filename
        output_path = output_dir / f"tvm_{image_path.name}"
        
        # Save the denoised image
        cv2.imwrite(str(output_path), denoised_image)
        
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
