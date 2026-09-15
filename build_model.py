import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Define key paths and parameters
BASE_DIR = os.path.join("dataset", "processed")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=== STEP 8: Constructing Data Loaders ===")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Performance optimization: Prefetching and caching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Data loaders ready!")

print("\n=== STEP 9: Building Transfer Learning Model (EfficientNetB0) ===")

# Data Augmentation Sequential Block
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
], name="data_augmentation")

def create_model():
    # Base model initialization
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers for feature extraction
    base_model.trainable = False

    # Input node
    inputs = layers.Input(shape=(224, 224, 3))
    
    # Pass input through augmentation
    x = data_augmentation(inputs)
    
    # Pass augmented images to EfficientNet base
    x = base_model(x, training=False)
    
    # Top Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = models.Model(inputs, outputs, name="AI_vs_Real_Discriminator")
    return model

model = create_model()

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

# Display model architectural summary
model.summary()

# Save structural model blueprint
os.makedirs("models", exist_ok=True)
print("\nModel pipeline constructed successfully.")