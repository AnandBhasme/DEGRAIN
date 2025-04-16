# Image Noise Generation

This project demonstrates different types of noise that can be added to images using Python and OpenCV. It includes implementations for:

- Gaussian Noise
- Salt and Pepper Noise
- Rayleigh Noise
- Motion Blur
- Exponential Noise

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## Usage

1. Place your original image in the `Images` directory as `Original_Image.jpg`
2. Run the script:
   ```bash
   python noise.py
   ```
3. The noisy images will be saved in the `Images/Noisy_Images` directory

## Output

The script generates five different noisy versions of the input image:
- `gaussian_noise.jpg`
- `salt_pepper_noise.jpg`
- `rayleigh_noise.jpg`
- `motion_blur.jpg`
- `exponential_noise.jpg` 