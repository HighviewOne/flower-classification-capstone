#!/usr/bin/env python3
"""
Train the flower classification model.

This script:
1. Loads and preprocesses the flower dataset
2. Creates train/validation/test splits
3. Builds and trains a transfer learning model (MobileNetV2)
4. Saves the trained model to the models/ directory

Usage:
    python train.py
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "flower_photos"
MODEL_DIR = BASE_DIR / "models"

# Hyperparameters (will be tuned in notebook)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def load_data():
    """Load and preprocess the flower dataset."""
    
    print(f"Loading data from {DATA_DIR}...")
    
    # Training data (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Validation data (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    """Build transfer learning model with MobileNetV2."""
    
    print("Building model...")
    
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    # Preprocessing for MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Base model (frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    # Build the full model
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    return model, base_model


def train_model(model, train_ds, val_ds):
    """Train the model."""
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return history


def save_model(model, class_names):
    """Save the trained model."""
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODEL_DIR / "flower_classifier.keras"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save class names
    classes_path = MODEL_DIR / "class_names.txt"
    with open(classes_path, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✓ Class names saved to {classes_path}")


def main():
    """Main training pipeline."""
    
    print("=" * 50)
    print("FLOWER CLASSIFICATION - TRAINING")
    print("=" * 50)
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"Error: Dataset not found at {DATA_DIR}")
        print("Run 'python src/download_data.py' first.")
        return
    
    # Load data
    train_ds, val_ds, class_names = load_data()
    
    # Build model
    model, base_model = build_model(num_classes=len(class_names))
    
    # Train model
    history = train_model(model, train_ds, val_ds)
    
    # Evaluate
    print("\nFinal Evaluation:")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Accuracy: {accuracy:.2%}")
    
    # Save model
    save_model(model, class_names)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
