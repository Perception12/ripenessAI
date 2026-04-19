import os
import random
import shutil

# CONFIG
dataset_path = "dataset"
images_dir = os.path.join(dataset_path, "images", "train")
labels_dir = os.path.join(dataset_path, "labels", "train")

val_split = 0.2  # 20% for validation

# TARGET DIRS
val_images_dir = os.path.join(dataset_path, "images", "val")
val_labels_dir = os.path.join(dataset_path, "labels", "val")

os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# SPLIT
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
val_size = int(len(image_files) * val_split)
val_images = random.sample(image_files, val_size)

for img in val_images:
    label = img.replace(".jpg", ".txt").replace(".png", ".txt")
    shutil.move(os.path.join(images_dir, img), os.path.join(val_images_dir, img))
    shutil.move(os.path.join(labels_dir, label), os.path.join(val_labels_dir, label))

print(f"Moved {len(val_images)} images to validation set.")