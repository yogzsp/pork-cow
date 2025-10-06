import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# === KONFIGURASI ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 95
DATA_DIR = "./dataset"
MODEL_NAME = "meat_vs_pork_transfer.keras"

# === PILIH OPTIMIZER ===
OPTIMIZER_CHOICE = "adam"   # bisa diganti: "adam", "sgd", "rmsprop"

if OPTIMIZER_CHOICE == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
elif OPTIMIZER_CHOICE == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
elif OPTIMIZER_CHOICE == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
else:
    raise ValueError("Optimizer tidak dikenal, pilih: adam / sgd / rmsprop")

print(f"✅ Optimizer yang digunakan: {OPTIMIZER_CHOICE.upper()}")

# === DATA GENERATOR ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === INFO DATASET ===
total_images = sum([len(files) for _, _, files in os.walk(DATA_DIR)])
print(f"Jumlah data asli (sebelum augmentasi): {total_images}")
print(f"Jumlah sampel training per epoch: {train_gen.samples}")
print(f"Jumlah sampel validasi per epoch: {val_gen.samples}")
print(f"Classes: {train_gen.class_indices}")

# === BASE MODEL ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# === FULL MODEL ===
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === CALLBACKS ===
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-6),
    ModelCheckpoint(MODEL_NAME, save_best_only=True)
]

# === TRAIN ===
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# === GRAFIK TRAINING ===
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.close()

# === EVALUASI MODEL (Confusion Matrix, Recall, F1, dll) ===
print("\n🔍 Evaluasi model pada data validasi...")

val_gen.reset()
pred_probs = model.predict(val_gen)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.close()

# Classification Report (Precision, Recall, F1)
report = classification_report(
    true_labels,
    pred_labels,
    target_names=class_labels,
    output_dict=True
)

# Simpan evaluasi ke JSON
with open("evaluation_metrics.json", "w") as f:
    json.dump(report, f, indent=4)

# === SAVE TRAINING INFO ===
final_accuracy = history.history['val_accuracy'][-1]
with open("training_info.json", "w") as f:
    json.dump({
        "val_accuracy": float(final_accuracy),
        "epochs": len(history.history['val_accuracy']),
        "optimizer": OPTIMIZER_CHOICE,
        "original_data": total_images,
        "train_samples_per_epoch": train_gen.samples,
        "val_samples_per_epoch": val_gen.samples
    }, f, indent=4)

print(f"\n✅ Model tersimpan: {MODEL_NAME}")
print(f"📊 Akurasi Validasi Terakhir: {final_accuracy:.4f}")
print("📈 Confusion Matrix disimpan ke: confusion_matrix.png")
print("📄 Metrik evaluasi disimpan ke: evaluation_metrics.json")
