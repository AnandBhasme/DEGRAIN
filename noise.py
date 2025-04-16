import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian)
    return noisy_image

def add_salt_pepper_noise(image, prob=0.05):
    """Add salt and pepper noise to the image"""
    output = np.copy(image)
    # Salt noise
    salt = np.random.random(image.shape[:2]) < prob/2
    output[salt] = 255
    # Pepper noise
    pepper = np.random.random(image.shape[:2]) < prob/2
    output[pepper] = 0
    return output

def add_rayleigh_noise(image, scale=25):
    """Add Rayleigh noise to the image"""
    noise = np.random.rayleigh(scale, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_motion_blur(image, size=15):
    """Add motion blur to the image"""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_exponential_noise(image, scale=25):
    """Add exponential noise to the image"""
    noise = np.random.exponential(scale, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def main():
    # Read the original image
    image_path = 'Images/Original_Image.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image")
        return
    
    # Create output directory if it doesn't exist
    output_dir = 'Images/Noisy_Images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Add different types of noise and save the images
    # Gaussian noise
    gaussian_noisy = add_gaussian_noise(image)
    cv2.imwrite(f'{output_dir}/gaussian_noise.jpg', gaussian_noisy)
    
    # Salt and pepper noise
    salt_pepper_noisy = add_salt_pepper_noise(image)
    cv2.imwrite(f'{output_dir}/salt_pepper_noise.jpg', salt_pepper_noisy)
    
    # Rayleigh noise
    rayleigh_noisy = add_rayleigh_noise(image)
    cv2.imwrite(f'{output_dir}/rayleigh_noise.jpg', rayleigh_noisy)
    
    # Motion blur
    motion_blurred = add_motion_blur(image)
    cv2.imwrite(f'{output_dir}/motion_blur.jpg', motion_blurred)
    
    # Exponential noise
    exponential_noisy = add_exponential_noise(image)
    cv2.imwrite(f'{output_dir}/exponential_noise.jpg', exponential_noisy)
    
    print("All noisy images have been saved successfully!")

if __name__ == "__main__":
    main()
