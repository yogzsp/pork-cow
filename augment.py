import os
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Path dataset asli
DATA_DIR = "./dataset_ori"
# Path dataset hasil augmentasi
OUTPUT_DIR = "./dataset"
# File output JSON
INFO_FILE = "augmentation_info.json"

# Augmentor
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Berapa variasi per gambar
AUG_PER_IMAGE = 5

# Hitung jumlah gambar asli
original_count = sum([len(files) for _, _, files in os.walk(DATA_DIR)])

# Buat folder output sesuai struktur class
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Loop semua gambar dalam class
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Load gambar
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Simpan gambar asli ke folder baru
        array_to_img(x[0]).save(os.path.join(output_class_path, img_name))

        # Generate augmentasi
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_class_path,
                                  save_prefix="aug",
                                  save_format="jpg"):
            i += 1
            if i >= AUG_PER_IMAGE:
                break  # stop setelah dapat 5 variasi per gambar

# Hitung jumlah gambar setelah augmentasi
augmented_count = sum([len(files) for _, _, files in os.walk(OUTPUT_DIR)])

# Print hasil ke terminal
print("📊 Jumlah gambar sebelum augmentasi :", original_count)
print("📊 Jumlah gambar sesudah augmentasi :", augmented_count)
print("✅ Augmentasi selesai. Dataset baru tersimpan di:", OUTPUT_DIR)

# Simpan info ke JSON
with open(INFO_FILE, "w") as f:
    json.dump({
        "original_data": original_count,
        "augmented_data": augmented_count,
        "aug_per_image": AUG_PER_IMAGE,
        "output_dir": OUTPUT_DIR
    }, f, indent=4)

print(f"💾 Info augmentasi disimpan ke {INFO_FILE}")
