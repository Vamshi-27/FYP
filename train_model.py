import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Directories
train_dir = "data/train"
val_dir = "data/validation"
test_dir = "data/test"

output_size = len(os.listdir(train_dir))
epochs = 30

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(256, 256), color_mode='grayscale',
    class_mode='categorical', batch_size=32
)

val_generator = test_datagen.flow_from_directory(
    val_dir, target_size=(256, 256), color_mode='grayscale',
    class_mode='categorical', batch_size=32
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(output_size, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("model/gesture_model.h5", save_best_only=True)
]

# Training
history = model.fit(
    train_generator, epochs=epochs,
    validation_data=val_generator, callbacks=callbacks
)

model.save_weights("model/gesture_model_weights.h5")
