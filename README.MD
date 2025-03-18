# Easy Augmentor 📸✨
A fast, configurable **image augmentation CLI tool** built with **OpenCV** and **Python**. Supports **rotations, flipping, cropping, brightness adjustment, contrast enhancement, grayscale conversion, Gaussian noise, motion blur, and Gaussian blur**. 

## 🚀 Features
✔ **Batch processing** for multiple images  
✔ **Configurable augmentations** using YAML files  
✔ **Supports CLI selection for default/user settings**  
✔ **Progress tracking with `tqdm`**  
✔ **Automatically prevents filename overwrites**  

---

## 📥 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/easy-augmentor.git
cd easy-augmentor

### **2️⃣ Install Dependencies**
pip install -r requirements.txt


## **📦 Requirements**
| **Requirement** | **Version** |
|---------------|------------|
| **Python** | `>=3.12` |
| **pip** | Latest |

🔹 Ensure Python is installed. You can check by running:
```bash
python --version
pip --version

## **🛠 Configuration**
The **`user_config.yaml`** file allows customization of image augmentations. Below is a **sample configuration file**:

```yaml
geometric_transformations:
  rotation:
    enabled: true
    angle: 45  # Increments of 45 degrees (0, 45, 90, ..., 315)
  flipping:
    enabled: true
    mode: "horizontal"  # Options: "horizontal", "vertical", "both"

color_based_augmentations:
  brightness_adjustment:
    enabled: true
    factor: 1.2  # Adjust brightness (values >1.0 increase brightness, <1.0 decrease)
  contrast_adjustment:
    enabled: true
    factor: 1.5  # Higher values increase contrast
  gray_scale:
    enabled: true  # Converts images to grayscale

noise_augmentations:
  gaussian_noise:
    enabled: true
    mean: 0
    sigma: 25  # Standard deviation of Gaussian noise
  motion_blur:
    enabled: true
    kernel_size: 15  # Kernel size for motion blur
  gaussian_blur:
    enabled: true
    kernel_size: 5  # Kernel size for Gaussian blur

## **📌 Function Use Cases**
This table explains when each augmentation should be used:

| **Function**              | **Transformation Type**        | **Best Suited For**                                    | **Explanation** |
|--------------------------|--------------------------------|--------------------------------------------------------|-----------------|
| `rotation()`             | Geometric                     | Image Classification, Object Detection, Gesture Recognition | Enhances model robustness to different object orientations. |
| `flipping()`             | Geometric                     | Image Classification, Object Detection                 | Useful for symmetrical objects; may not be suitable for asymmetrical data like text. |
| `cropping()`             | Geometric                     | Object Detection, Image Segmentation                   | Helps models handle scenarios where objects are partially visible. |
| `brightness_adjustment()`| Color-Based                   | Image Classification, Object Detection, OCR           | Prepares models for varying lighting conditions. |
| `contrast_adjustment()`  | Color-Based                   | Image Classification, OCR                             | Assists in distinguishing objects under different contrast levels. |
| `gray_scale()`           | Color-Based                   | Image Classification (where color isn't critical), OCR | Simplifies data, focusing models on shapes and textures. |
| `gaussian_noise()`       | Noise-Based                   | Image Classification, Generative Models (GANs, Diffusion Models) | Improves robustness against real-world noise and sensor imperfections. |
| `motion_blur()`          | Noise-Based                   | Image Classification (e.g., Traffic Surveillance)     | Simulates motion-induced blurriness in real-world scenarios. |
| `gaussian_blur()`        | Noise-Based                   | Image Classification, Super-Resolution Models         | Assists in training denoising models; excessive use may reduce sharpness. |


##📜 License

This project is licensed under the MIT License.