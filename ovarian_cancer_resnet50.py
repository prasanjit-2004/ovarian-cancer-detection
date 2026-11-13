"""
Ovarian Cancer Detection with ResNet50 (Transfer Learning)
----------------------------------------------------------
Classifies ovarian histopathology images (e.g. cancer vs non-cancer, or
cancer subtypes) using a ResNet50 backbone pretrained on ImageNet.

Expected data layout (class names = folder names):

Option A (already split):
    data/
      train/  class_a/*.jpg  class_b/*.jpg
      val/    class_a/*.jpg  class_b/*.jpg
      test/   class_a/*.jpg  class_b/*.jpg   (optional)

Option B (single folder, split automatically 70/15/15):
    data/
      class_a/*.jpg
      class_b/*.jpg

Usage:
    python ovarian_cancer_resnet50.py --data data
    python ovarian_cancer_resnet50.py --data data --epochs 15 --fine-tune-epochs 15

NOTE: For research and educational use only. Not a medical device.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

SEED = 42
OUTPUT_DIR = Path("outputs")
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_datasets(data_dir: Path, img_size: int, batch_size: int):
    common = dict(
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        seed=SEED,
    )

    if (data_dir / "train").exists():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir / "train", shuffle=True, **common
        )
        val_name = "val" if (data_dir / "val").exists() else "valid"
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir / val_name, shuffle=False, **common
        )
        test_ds = None
        if (data_dir / "test").exists():
            test_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir / "test", shuffle=False, **common
            )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.3, subset="training", shuffle=True, **common
        )
        holdout = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.3, subset="validation", shuffle=False, **common
        )
        n = int(tf.data.experimental.cardinality(holdout).numpy())
        val_ds = holdout.take(n // 2)
        test_ds = holdout.skip(n // 2)
        train_ds.class_names  # noqa: B018 (kept before further transforms)
        class_names = train_ds.class_names
        return train_ds, val_ds, test_ds, class_names

    return train_ds, val_ds, test_ds, train_ds.class_names


def make_augmenter():
    # Histology images have no fixed orientation, so flips/rotations are safe.
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def prepare(ds, augment=None):
    if augment is not None:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_class_weights(train_ds, num_classes):
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=labels
    )
    print("Class counts:", dict(zip(range(num_classes), np.bincount(labels))))
    return dict(enumerate(weights))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(num_classes: int, img_size: int):
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="ovarian_resnet50"), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks():
    return [
        callbacks.ModelCheckpoint(
            OUTPUT_DIR / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def plot_history(histories):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val_acc = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    val_loss = sum([h.history["val_loss"] for h in histories], [])
    split = len(histories[0].history["accuracy"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, tr, va, title in [
        (ax[0], acc, val_acc, "Accuracy"),
        (ax[1], loss, val_loss, "Loss"),
    ]:
        a.plot(tr, label="train")
        a.plot(va, label="val")
        if len(histories) > 1:
            a.axvline(split - 1, color="gray", linestyle="--", label="fine-tune start")
        a.set_title(title)
        a.set_xlabel("Epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()


def evaluate(model, ds, class_names, name="test"):
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr")
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError as e:
        print("ROC-AUC not computed:", e)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", xticks_rotation=45
    )
    plt.title(f"Confusion matrix ({name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_raw, val_raw, test_raw, class_names = load_datasets(
        Path(args.data), args.img_size, args.batch_size
    )
    num_classes = len(class_names)
    print("Classes:", class_names)
    (OUTPUT_DIR / "class_names.json").write_text(json.dumps(class_names))

    class_weights = get_class_weights(train_raw, num_classes)

    train_ds = prepare(train_raw, augment=make_augmenter())
    val_ds = prepare(val_raw)
    test_ds = prepare(test_raw) if test_raw is not None else None

    model, base = build_model(num_classes, args.img_size)

    # Phase 1: train only the new classification head
    print("\n--- Phase 1: training classification head ---")
    compile_model(model, lr=1e-3)
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(),
    )
    histories = [h1]

    # Phase 2: unfreeze the top of ResNet50 and fine-tune with a low LR
    if args.fine_tune_epochs > 0:
        print("\n--- Phase 2: fine-tuning top ResNet50 layers ---")
        base.trainable = True
        for layer in base.layers[: -args.unfreeze_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        compile_model(model, lr=1e-5)
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(),
        )
        histories.append(h2)

    plot_history(histories)
    evaluate(model, val_ds, class_names, name="validation")
    if test_ds is not None:
        evaluate(model, test_ds, class_names, name="test")

    model.save(OUTPUT_DIR / "ovarian_resnet50_final.keras")
    print(f"\nSaved models and plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ovarian cancer detection - ResNet50")
    p.add_argument("--data", default="data", help="dataset folder")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10, help="head training epochs")
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--unfreeze-layers", type=int, default=30)
    main(p.parse_args())
