"""
Script Name: data_utils.py

Purpose: This script collects image file paths and their corresponding labels
         from a hierarchical directory structure and splits them into
         training, validation, and test datasets for a liver ultrasound
         classification task.

Author: James
Date Created: 2025-06-19
Usage: Run directly. Ensure the data directory structure is as expected.
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Set path to the root folder
root_dir = "C:\\Users\\James\\Documents\\Portfolio\\liver-ultrasound-classification\\data"  # your data folder that holds the folders for the 3 categories

# 2. Initialise list to collect file paths and labels
data = []

# 3. Walk through each class subfolder
for label in ["Benign", "Malignant", "Normal"]:
    img_folder = os.path.join(root_dir, label, "image")
    image_files = glob.glob(os.path.join(img_folder, "*.jpg"))

    for img_path in image_files:
        data.append({"image_path": img_path, "label": label})

# 4. Create a DataFrame
df = pd.DataFrame(data)

# 5. Split the data (stratify ensures balance)
train_val, test_data = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
train_data, val_data = train_test_split(
    train_val, test_size=0.1, stratify=train_val["label"], random_state=42
)

# 6. Check dataset sizes
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))
