from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import os

## ENVIRONMENT
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


## GLOBAL VARIABLES 
img_shape = (64, 64)
channels = 3
batch_size = 64
classes=1081
epochs=8

## DATA DIRECTORIES
data_dir = "plantnet_300k"
train_dir = os.path.join(data_dir, "images_train")
test_dir = os.path.join(data_dir, "images_test")
val_dir = os.path.join(data_dir, "images_val")

## DATA GENERATORS
train_generator = ImageDataGenerator(rescale=1./255,
                                     horizontal_flip=True,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2).flow_from_directory(
                                                    train_dir,
                                                    target_size=img_shape,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                )
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                                                    test_dir,
                                                    target_size=img_shape,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                )
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                                                    val_dir,
                                                    target_size=img_shape,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                )

## MODEL
model = Sequential()

## FEATURE EXTRACTION
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=img_shape+(channels,)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

## DENSE LAYERS / HIDDEN LAYERS
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))

## CLASSIFICATION LAYER
model.add(Dense(units=classes, activation='softmax'))


## COMPILING
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.summary()

## TRAINING
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    verbose=1,
                    batch_size=batch_size
                    )

