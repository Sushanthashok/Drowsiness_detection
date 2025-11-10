# training/train_eye_state.py
import os, shutil, random, json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = Path("../data/eyes")   # train/open, train/closed, val/open, val/closed
USE_EXISTING_SPLIT = True          # set False if you have data/eyes/all/{open,closed} and want auto-split

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3

MODEL_OUT = Path("../models/eye_state_cnn.h5")
LABELS_OUT = Path("../models/eye_state_labels.json")
PLOT_OUT = Path("../models/training_plot.png")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# OPTIONAL: Auto split dataset
# -----------------------------
def ensure_split():
    if USE_EXISTING_SPLIT:
        return
    src_open = DATA_ROOT / "all" / "open"
    src_closed = DATA_ROOT / "all" / "closed"
    dest_train_open = DATA_ROOT / "train" / "open"
    dest_train_closed = DATA_ROOT / "train" / "closed"
    dest_val_open = DATA_ROOT / "val" / "open"
    dest_val_closed = DATA_ROOT / "val" / "closed"
    for d in [dest_train_open, dest_train_closed, dest_val_open, dest_val_closed]:
        d.mkdir(parents=True, exist_ok=True)

    def split_and_copy(src, dst_train, dst_val, val_ratio=0.2):
        imgs = [f for f in src.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
        random.shuffle(imgs)
        split = int(len(imgs) * (1 - val_ratio))
        for i, img in enumerate(imgs):
            target = dst_train if i < split else dst_val
            shutil.copy2(img, target / img.name)

    split_and_copy(src_open,   dest_train_open,   dest_val_open)
    split_and_copy(src_closed, dest_train_closed, dest_val_closed)

# -----------------------------
# Dataset loader  (FIXED)
# -----------------------------
def make_datasets():
    """Create TF datasets and PRESERVE class_names before mapping/prefetch."""
    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "val"

    raw_train = keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=True
    )
    raw_val = keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False
    )

    class_names = raw_train.class_names  # <-- save BEFORE mapping/prefetch

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = raw_val.map(lambda x, y: (x / 255.0, y)).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names

# -----------------------------
# CNN Model
# -----------------------------
def build_model(num_classes=2):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Optional: simple training plot
# -----------------------------
def save_training_plot(history, out_path=PLOT_OUT):
    try:
        import matplotlib.pyplot as plt
        hist = history.history
        epochs = range(1, len(hist["accuracy"]) + 1)

        plt.figure()
        plt.plot(epochs, hist["accuracy"], label="train_acc")
        plt.plot(epochs, hist["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved training plot to: {out_path}")
    except Exception as e:
        print("Plot not saved (optional):", e)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_split()
    train_ds, val_ds, class_names = make_datasets()
    print(f"✅ Classes detected: {class_names}")

    model = build_model(num_classes=len(class_names))
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        )
    ]

    print("🚀 Starting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print(f"✅ Training finished. Best model saved at: {MODEL_OUT}")
    LABELS_OUT.write_text(json.dumps({"class_names": class_names, "img_size": IMG_SIZE}))
    print(f"✅ Labels saved to: {LABELS_OUT}")

    save_training_plot(history)

if __name__ == "__main__":
    main()
