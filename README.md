# Image Preprocessing Pipeline for Computer Vision

This project implements a modular image preprocessing pipeline designed to transform raw, mobile-captured imagery into optimized inputs for Deep Learning models and Convolutional Neural Networks (CNNs). 

By applying these techniques, we ensure mathematical consistency across the dataset, which is critical for model convergence and efficient feature extraction.

---

## Preprocessing Techniques

The pipeline executes five core techniques commonly used in Computer Vision workflows:

### 1. Geometric & Statistical Refinement
* **Dynamic Resizing:** Standardizes inconsistent mobile camera dimensions to a uniform $256 \times 256$ input using `torchvision.transforms`.
* **Min-Max Normalization:** Scales pixel intensities to a $[0, 255]$ range across the distribution to ensure stable feature values.

### 2. Denoising & Feature Enhancement
* **Gaussian Blur:** Functions as a low-pass filter to suppress high-frequency noise using a $5 \times 5$ kernel.
* **Laplacian Sharpening:** Applies a custom Laplacian kernel to enhance edge definition, assisting models in learning structural boundaries.

### 3. Data Augmentation & Robustness
* **Salt & Pepper Noise:** Stochastic noise injection (5% density) used to simulate sensor errors and evaluate model robustness.

---

## Project Structure

The repository is organized into specific output directories to track each stage of the transformation:

```text
├── cats/                           # Raw input images (Captured via mobile)
├── output_resize/                  # Images adjusted to 256x256
├── output_gaussian_blur/           # Noise reduction results
├── output_salt_pepper/             # Synthetic noise samples
├── output_laplacian/               # Edge enhancement results
├── output_normalize/               # Pixel values scaled via Min-Max
├── output_combined_preprocessing/  # Sequential pipeline (Resize → Laplacian → Blur → Normalize)
├── image_dataset_preprocessing.py  # Main execution script
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
