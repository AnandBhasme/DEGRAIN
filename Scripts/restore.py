import cv2
import numpy as np
import os
from scipy.signal import wiener
import matplotlib.pyplot as plt

def create_directories():
    """Create necessary directories for restored images"""
    base_dir = 'Images/Restored_Images'
    techniques = ['arithmetic_mean', 'median', 'wiener', 'pseudo_inverse', 
                 'trimmed_average', 'gaussian']
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each technique
    for technique in techniques:
        os.makedirs(os.path.join(base_dir, technique), exist_ok=True)

def arithmetic_mean_filter(image, kernel_size=3):
    """Apply arithmetic mean filter"""
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image, kernel_size=3):
    """Apply median filter"""
    return cv2.medianBlur(image, kernel_size)

def wiener_filter(image, kernel_size=3):
    """Apply Wiener filter"""
    # Convert to float32 for processing
    image_float = image.astype(np.float32)
    # Apply Wiener filter
    restored = wiener(image_float, (kernel_size, kernel_size))
    # Convert back to uint8
    return np.clip(restored, 0, 255).astype(np.uint8)

def pseudo_inverse_filter(image, kernel_size=3):
    """Apply pseudo-inverse filter"""
    # Create a simple blur kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Convert to frequency domain
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    
    # Apply pseudo-inverse
    epsilon = 1e-6  # Small constant to avoid division by zero
    restored_fft = image_fft / (kernel_fft + epsilon)
    
    # Convert back to spatial domain
    restored = np.fft.ifft2(restored_fft)
    return np.abs(restored).astype(np.uint8)

def trimmed_average_filter(image, kernel_size=3, trim_percent=20):
    """Apply trimmed average filter"""
    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)
    
    for i in range(pad, padded.shape[0] - pad):
        for j in range(pad, padded.shape[1] - pad):
            window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
            # Flatten the window and sort
            sorted_window = np.sort(window.flatten())
            # Calculate number of pixels to trim
            trim = int(len(sorted_window) * trim_percent / 100)
            # Take average of remaining pixels
            result[i-pad, j-pad] = np.mean(sorted_window[trim:-trim])
    
    return result.astype(np.uint8)

def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Apply Gaussian filter"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def display_images(original, restored, title):
    """Display original and restored images side by side"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.title('Original Noisy Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(restored, cmap='gray')
    plt.title(f'Restored Image ({title})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Create necessary directories
    create_directories()
    
    # Read noisy images
    noisy_images = {
        'gaussian': 'Images/Noisy_Images/gaussian_noise.jpg',
        'salt_pepper': 'Images/Noisy_Images/salt_pepper_noise.jpg',
        'rayleigh': 'Images/Noisy_Images/rayleigh_noise.jpg',
        'motion_blur': 'Images/Noisy_Images/motion_blur.jpg',
        'exponential': 'Images/Noisy_Images/exponential_noise.jpg'
    }
    
    # Process each noisy image
    for noise_type, image_path in noisy_images.items():
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read {image_path}")
            continue
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\nProcessing {noise_type} noise...")
        
        # Apply each restoration technique and display results
        # Arithmetic Mean
        restored = arithmetic_mean_filter(image)
        display_images(image, restored, 'Arithmetic Mean')
        
        # Median
        restored = median_filter(image)
        display_images(image, restored, 'Median')
        
        # Wiener
        restored = wiener_filter(image)
        display_images(image, restored, 'Wiener')
        
        # Pseudo Inverse
        restored = pseudo_inverse_filter(image)
        display_images(image, restored, 'Pseudo Inverse')
        
        # Trimmed Average
        restored = trimmed_average_filter(image)
        display_images(image, restored, 'Trimmed Average')
        
        # Gaussian
        restored = gaussian_filter(image)
        display_images(image, restored, 'Gaussian')
    
    print("All images have been restored and displayed successfully!")

if __name__ == "__main__":
    main()
