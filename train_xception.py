import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = 299
BATCH = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 8

# ---- Data ----
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train = train_gen.flow_from_directory(
    "dataset/faces",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    subset="training"
)

val = train_gen.flow_from_directory(
    "dataset/faces",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    subset="validation"
)

# ---- Base model ----
base = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze all layers (stage 1)
for l in base.layers:
    l.trainable = False

# ---- Custom head ----
x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---- Callbacks ----
cbs = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ModelCheckpoint("best_xception.h5", monitor="val_loss", save_best_only=True)
]

# ---- Stage 1: train head ----
model.fit(train, validation_data=val, epochs=EPOCHS_STAGE1, callbacks=cbs)

# ---- Stage 2: fine-tune top layers ----
for l in base.layers[-40:]:   # unfreeze last 40 layers
    l.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=EPOCHS_STAGE2, callbacks=cbs)

# Save final
model.save("model_xception.h5")
print(" Saved xception_deepfake.h5")