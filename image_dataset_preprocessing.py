# Libraries for working
import os
import cv2
import numpy as np
from skimage.util import random_noise
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Folder path for inputs
input_folder = "cats"

# Folder paths for outputs
save_paths = {
    "resize": "output_resize",
    "gaussian_blur": "output_gaussian_blur",
    "salt_pepper": "output_salt_pepper",
    "laplacian": "output_laplacian",
    "normalize": "output_normalize",
    "combined": "output_combined_preprocessing"
}

for folder in save_paths.values():
    os.makedirs(folder, exist_ok=True)

# transform function (RESIZE IMAGE)
resize_img = transforms.Resize((256, 256))

# NORMALIZATION
def safe_normalize(pil_img):
    img = np.array(pil_img).astype(np.float32)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(img_norm.astype(np.uint8))

# Laplacian KERNAL
laplacian_kernel = np.array([
    [0, -1,  0],
    [-1,  5, -1],
    [0, -1,  0]
])

# Load images
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Found {len(image_files)} images.")


# Processing
for filename in tqdm(image_files):

    file_path = os.path.join(input_folder, filename)

    # Load images in both formats
    original_pil = Image.open(file_path).convert("RGB")
    original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)

    # 1 RESIZE
    resized_img = resize_img(original_pil)
    resized_img.save(f"{save_paths['resize']}/{filename}")

    # 2 GAUSSIAN BLUR
    blurred_img = cv2.GaussianBlur(original_cv, (5, 5), 0)
    cv2.imwrite(f"{save_paths['gaussian_blur']}/{filename}", blurred_img)

    # 3 SALT & PEPPER
    sp_noise = random_noise(original_cv, mode='s&p', amount=0.05)
    sp_noise = (sp_noise * 255).astype(np.uint8)
    cv2.imwrite(f"{save_paths['salt_pepper']}/{filename}", sp_noise)

    # 4 Laplacian Filter
    laplacian_img = cv2.filter2D(original_cv, -1, laplacian_kernel)
    cv2.imwrite(f"{save_paths['laplacian']}/{filename}", laplacian_img)

    # 5 NORMALIZATION
    normalized_img = safe_normalize(original_pil)
    normalized_img.save(f"{save_paths['normalize']}/{filename}")


    # Combined techniques in one image
    
    # RESIZE → Laplacian Filter → GAUSSIAN BLUR → Normalize
    combined = resize_img(original_pil)
    combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

    # Laplacian Filter
    combined_cv = cv2.filter2D(combined_cv, -1, laplacian_kernel)

    # GAUSSIAN BLUR
    combined_cv = cv2.GaussianBlur(combined_cv, (3, 3), 0)

    combined_pil = Image.fromarray(cv2.cvtColor(combined_cv, cv2.COLOR_BGR2RGB))

    # Normalize
    combined_pil = safe_normalize(combined_pil)

    # Save combined result
    combined_pil.save(f"{save_paths['combined']}/{filename}")

print("Preprocessing of images completed successfully!")
