import tqdm
import pyfiglet
import cv2
import os
import yaml
import random
import numpy as np
import threading
import inquirer

# --- GLOBAL VARIABLES --- #
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
DEFAULT_CONFIG = "./default_config.yaml"
USER_CONFIG = "./user_config.yaml"

# --- UTILITY FUNCTIONS --- #
def print_banner(text, font="standard"):
    """Prints a banner with the given text using pyfiglet."""
    f = pyfiglet.Figlet(font=font)
    print(f.renderText(text))

def input_folder_exists():
    return os.path.exists(INPUT_FOLDER) and os.path.isdir(INPUT_FOLDER)

def output_folder_exists():
    return os.path.exists(OUTPUT_FOLDER) and os.path.isdir(OUTPUT_FOLDER)

def configs_exist():
    return os.path.exists(USER_CONFIG) and os.path.isfile(USER_CONFIG)

def is_output_empty():
    return not any(os.scandir(OUTPUT_FOLDER))

def is_input_empty():
    """Check if the input folder is empty. If empty, exit the program."""
    if not any(os.scandir(INPUT_FOLDER)):  
        print("❌ ERROR: Input folder is empty. Please add images and try again.")
        exit(1)  # Exit program with an error status
    return False  # Return False (not empty) if files exist

def valid_input_file_types():
    valid_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    return any(file.endswith(valid_extensions) for file in os.listdir(INPUT_FOLDER))

def read_config(use_default=False):
    """Reads the selected YAML config (default or user) and merges it."""
    config_file = DEFAULT_CONFIG if use_default else USER_CONFIG

    with open(DEFAULT_CONFIG, "r") as file:
        default_config = yaml.safe_load(file)

    with open(config_file, "r") as file:
        user_config = yaml.safe_load(file)

    for category, settings in user_config.items():
        if category in default_config:
            for key, value in settings.items():
                if isinstance(value, dict):
                    default_config[category][key].update(value)  # Merge nested dicts
                else:
                    default_config[category][key] = value  # Override with user value
        else:
            default_config[category] = settings  # Add new user-defined settings

    return default_config

# --- AUGMENTATION CLASS --- #
class Transformations:
    def __init__(self):
        pass

    # Geometric Transformations
    def rotation(self, image, image_name, angle_increments):
        total = int(360 / angle_increments)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        for i in range(total): 
            angle = i * angle_increments
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_rotated_{angle}.jpg", rotated)

    def flipping(self, image, image_name, mode):
        flip_map = {"h": 0, "v": 1, "b": -1}
        if mode.lower() in flip_map:
            flipped = cv2.flip(image, flip_map[mode.lower()])
            cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_flipped_{mode}.jpg", flipped)

    def cropping(self, image, image_name, num_of_crops, min_crop_size):
        h, w = image.shape[:2]
        for i in range(num_of_crops):
            crop_h = random.randint(min_crop_size, h // 2)
            crop_w = random.randint(min_crop_size, w // 2)
            h_start = random.randint(0, h - crop_h)
            w_start = random.randint(0, w - crop_w)

            crop = image[h_start:h_start + crop_h, w_start:w_start + crop_w]
            cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_cropped_{i}.jpg", crop)

    # Color-Based Augmentations
    def brightness_adjustment(self, image, image_name, brightness_factors):
        for i, factor in enumerate(brightness_factors):
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor)
            filename = f"{OUTPUT_FOLDER}/{image_name}_brightness_{i}_{str(factor).replace('.', '_')}.jpg"
            cv2.imwrite(filename, adjusted_image)

    def contrast_adjustment(self, image, image_name, contrast_factors):
        for i, factor in enumerate(contrast_factors):
            adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            filename = f"{OUTPUT_FOLDER}/{image_name}_contrast_{i}_{str(factor).replace('.', '_')}.jpg"
            cv2.imwrite(filename, adjusted_image)

    def gray_scale(self, image, image_name):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_gray_scale.jpg", gray_image)

    # Noise Augmentations
    def gaussian_noise(self, image, image_name, mean, sigma):
        image = image.astype(np.float32)
        gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_gaussian_noise.jpg", noisy_image)

    def motion_blur(self, image, image_name, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size) / kernel_size
        blurred_image = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_motion_blur.jpg", blurred_image)

    def gaussian_blur(self, image, image_name, kernel_size):
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        cv2.imwrite(f"{OUTPUT_FOLDER}/{image_name}_gaussian_blur.jpg", blurred_image)

# --- MAIN FUNCTION --- #
def main():
    print_banner("Easy Augmentation", font="small")

    # Ensure input folder is not empty
    is_input_empty()

    # CLI Checkbox: Ask user to select config type
    config_question = [
        inquirer.List(
            "config_choice",
            message="Choose configuration type:",
            choices=["Default Config", "User Config (See documentation for details)"],
        )
    ]
    answers = inquirer.prompt(config_question)
    
    # Load selected config
    config = read_config(use_default=(answers["config_choice"] == "Default Config"))

    # Check if output folder has files
    if not is_output_empty():
        print("⚠️  Files in the Output folder will be overwritten.")
        confirm = input("❗ Press Enter to continue or CTRL+C to cancel...")

    # Ensure input & output folders exist
    if not input_folder_exists():
        print("📁 Input folder does not exist. Creating new one...")
        os.makedirs(INPUT_FOLDER)

    if not output_folder_exists():
        print("📁 Output folder does not exist. Creating new one...")
        os.makedirs(OUTPUT_FOLDER)

    if not valid_input_file_types():
        print("❌ Invalid file types. Only .png, .jpg, .jpeg, .tiff, .bmp are allowed.")
        return

    # Process Images with Progress Bar
    transformer = Transformations()
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))]

    with tqdm.tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for file_name in image_files:
            image_path = os.path.join(INPUT_FOLDER, file_name)
            image = cv2.imread(image_path)

            image_name = os.path.splitext(file_name)[0]  # Extract filename without extension

            if config["geometric_transformations"]["rotation"]["enabled"]:
                transformer.rotation(image, image_name, config["geometric_transformations"]["rotation"]["angle"])

            if config["geometric_transformations"]["flipping"]["enabled"]:
                transformer.flipping(image, image_name, config["geometric_transformations"]["flipping"]["mode"])

            if config["color_based_augmentations"]["brightness_adjustment"]["enabled"]:
                transformer.brightness_adjustment(image, image_name, [config["color_based_augmentations"]["brightness_adjustment"]["factor"]])

            if config["color_based_augmentations"]["contrast_adjustment"]["enabled"]:
                transformer.contrast_adjustment(image, image_name, [config["color_based_augmentations"]["contrast_adjustment"]["factor"]])

            if config["color_based_augmentations"]["gray_scale"]["enabled"]:
                transformer.gray_scale(image, image_name)

            if config["noise_augmentations"]["gaussian_noise"]["enabled"]:
                transformer.gaussian_noise(image, image_name, config["noise_augmentations"]["gaussian_noise"]["mean"], config["noise_augmentations"]["gaussian_noise"]["sigma"])

            if config["noise_augmentations"]["motion_blur"]["enabled"]:
                transformer.motion_blur(image, image_name, config["noise_augmentations"]["motion_blur"]["kernel_size"])

            if config["noise_augmentations"]["gaussian_blur"]["enabled"]:
                transformer.gaussian_blur(image, image_name, config["noise_augmentations"]["gaussian_blur"]["kernel_size"])

            pbar.update(1)  # Update progress bar after each image

    print("✅ Image augmentation complete!")

if __name__ == "__main__":
    main()
