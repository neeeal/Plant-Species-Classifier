from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small, MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

## ENVIRONMENT
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


## DATA DIRECTORIES
data_dir = "plantnet_300k"
train_dir = os.path.join(data_dir, "images_train")
test_dir = os.path.join(data_dir, "images_test")
val_dir = os.path.join(data_dir, "images_val")

## GLOBAL VARIABLES 
img_shape = (128,128)
channels = 3
batch_size = 8
classes=len(os.listdir(train_dir))-1
epochs=64


## DATA GENERATORS
train_generator = ImageDataGenerator(rescale=1./255,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     width_shift_range=0.4,
                                     height_shift_range=0.4,
                                     rotation_range=40,
                                     shear_range=0.2,
                                     zoom_range=0.2
                                     ).flow_from_directory(
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

## TRANSFER LEARNING MODEL
transfer_model = InceptionV3(
    input_shape=img_shape+(channels,),
    include_top=False,
    weights='imagenet',
    # classes=classes,
    # dropout_rate=0.4,
    # classifier_activation='softmax',
    # include_preprocessing=True
)


print("LAYERS ",len(transfer_model.layers))
# for layer in mobilenet.layers[:-130]:
#     layer.trainable = False

model = Sequential()

## FEATURE EXTRACTION
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', input_shape=img_shape+(channels,)))
# model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=1, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.4))

# model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
# model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.4))

# model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(transfer_model)

## DENSE LAYERS / HIDDEN LAYERS
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
# model.add(Dense(units=256, activion='relu'))
# model.add(Dense(units=256, activatation='relu'))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-7)
es = EarlyStopping(monitor='loss', patience=5)

## CLASSIFICATION LAYER
model.add(Dense(units=classes, activation='softmax'))


## COMPILING
model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['acc'])
model.summary()

## TRAINING
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    verbose=1,
                    batch_size=batch_size,
                    callbacks=[es, reduce_lr])
model_dir = "plantnet_300K//models"
model_version = len(os.listdir(model_dir))
model.save(f'{model_dir}//modelV{model_version}.h5')