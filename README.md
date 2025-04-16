# DEGRAIN: A Comparative Study on the Restoration of Degraded Digital Images Under Various Noise Models

## Overview
This project implements and compares various image restoration techniques for different types of image degradation. The goal is to analyze the effectiveness of different restoration methods on various noise models and provide quantitative metrics for comparison.

## Noise Models
The project handles the following types of image degradation:
1. **Gaussian Noise**: Additive white noise with a normal distribution
2. **Salt & Pepper Noise**: Random black and white pixels
3. **Rayleigh Noise**: Noise with a Rayleigh distribution
4. **Motion Blur**: Linear motion blur effect
5. **Exponential Noise**: Noise with an exponential distribution

## Restoration Techniques
The following restoration methods are implemented and compared:

1. **Arithmetic Mean Filter**
   - Simple averaging filter that replaces each pixel with the mean of its neighborhood
   - Effective for Gaussian noise but may blur edges

2. **Median Filter**
   - Non-linear filter that replaces each pixel with the median of its neighborhood
   - Particularly effective for salt & pepper noise
   - Preserves edges better than mean filters

3. **Wiener Filter**
   - Optimal linear filter that minimizes mean square error
   - Works well for both noise reduction and deblurring
   - Requires knowledge of noise statistics

4. **Pseudo-Inverse Filter**
   - Frequency domain approach for deblurring
   - Effective for motion blur restoration
   - Sensitive to noise

5. **Trimmed Average Filter**
   - Removes extreme values before averaging
   - Good balance between noise removal and edge preservation
   - Effective for mixed noise types

6. **Total Variation Minimization (TVM)**
   - Advanced technique that preserves edges while removing noise
   - Based on Chambolle's algorithm
   - Particularly effective for preserving sharp edges

## Image Quality Metrics
The following metrics are used to evaluate restoration quality:

1. **Mean Squared Error (MSE)**
   - Measures average squared difference between original and processed images
   - Lower values indicate better quality

2. **Peak Signal-to-Noise Ratio (PSNR)**
   - Ratio between maximum possible power of a signal and power of noise
   - Higher values indicate better quality
   - Typically measured in decibels (dB)

3. **Normalized Mean Squared Error (NMSE)**
   - Normalized version of MSE
   - Provides scale-invariant comparison

4. **Normalized Absolute Error (NAE)**
   - Measures absolute difference between images
   - Less sensitive to outliers than MSE

5. **Structural Similarity Index (SSIM)**
   - Perceptual metric that considers structural information
   - Values range from -1 to 1, with 1 indicating perfect similarity

6. **Universal Image Quality Index (UIQI)**
   - Combines correlation, luminance, and contrast
   - Values range from -1 to 1, with 1 indicating perfect quality

## Project Structure
```
DIP-Case-Study_ImageRestoration/
├── Images/
│   ├── Original_Image.jpg
│   ├── Noisy_Images/
│   └── Restored_Images/
│       ├── arithmetic_mean/
│       ├── median/
│       ├── wiener/
│       ├── pseudo_inverse/
│       ├── trimmed_average/
│       └── TVM/
└── Scripts/
    ├── ImageRestoration/
    │   ├── Arithmatic_mean.py
    │   ├── Median.py
    │   ├── wiener.py
    │   ├── psuedo-inverse.py
    │   ├── trimmed_average.py
    │   └── tvm.py
    ├── metric.py
    └── Image_Restoration.ipynb
```

## Dependencies
- Python 3.x
- OpenCV (cv2)
- NumPy
- SciPy
- scikit-image
- Matplotlib
- Pandas

## Future Enhancements
1. **Improved Visualization**
   - Interactive comparison of restoration results
   - Real-time parameter adjustment
   - Side-by-side comparison of different methods

2. **Performance Metrics**
   - Calculation of elapsed time for each restoration method
   - Memory usage analysis

3. **Advanced Features**
   - Interactive mode for parameter tuning
   - Batch processing capabilities
   - Support for more noise models and restoration techniques
   - Machine learning-based restoration methods

4. **User Interface**
   - Web-based interface for easy interaction
   - Real-time preview of restoration effects
   - Export functionality for results and metrics

## Version 2.0 Planned Features
- Interactive parameter adjustment interface
- Support for additional noise models:
  - Speckle noise
  - Poisson noise
  - Gamma noise
- Advanced restoration techniques:
  - Non-local means denoising
  - Bilateral filtering
  - Wavelet-based restoration
- Real-time performance monitoring
- Batch processing capabilities
- Enhanced visualization tools
- Comprehensive documentation and tutorials