import os
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, preprocessing,backend as K

import warnings
warnings.filterwarnings("ignore")
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random

DATA_DIR = "data/input/train_data"
DATA_DIR_PATH = Path(DATA_DIR)
for dirpath in DATA_DIR_PATH.glob("**/"):
    dirnames = [d for d in dirpath.iterdir() if d.is_dir()]
    filenames = [f for f in dirpath.iterdir() if f.is_file()]
    print(
        f"There are {len(dirnames)} directories and {len(filenames)} images in [{dirpath}]."
    )

print("\nTotal Training Images:", len(list(DATA_DIR_PATH.glob("*/*"))))

TRAIN_DATA_DIR = "data/input/train_data"
TRAIN_DATA_DIR = Path(TRAIN_DATA_DIR)

TEST_DATA_DIR = "data/input/test_data"
TEST_DATA_DIR = Path(TEST_DATA_DIR)

BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 180, 180
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) 
SEED = 42

train_ds = keras.utils.image_dataset_from_directory(directory=TRAIN_DATA_DIR,
                                                    label_mode='categorical',
                                                    batch_size=BATCH_SIZE,
                                                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    validation_split=0.2,
                                                    seed=SEED,
                                                    subset='training')

print()

val_ds = keras.utils.image_dataset_from_directory(directory=TRAIN_DATA_DIR,
                                                  label_mode='categorical',
                                                  batch_size=BATCH_SIZE,
                                                  image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  validation_split=0.2,
                                                  seed=SEED,
                                                  subset='validation')

class_names = train_ds.class_names

def show_first_3_images_of_3_directiories():
    plt.figure(figsize=(10,12))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[tf.argmax(labels[i])])
            plt.axis("off")

    plt.show()

NUM_CLASSES = len(class_names)

def experiment_1():
    normalizarion_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalizarion_layer(x), y))

    image_batch, label_batch = next(normalized_ds.as_numpy_iterator())

    first_image = image_batch[0]

    print(f"Image Batch Shape: {image_batch.shape}\nLabel Batch Shape: {label_batch.shape}")

    print("Min Pixel Value:", np.min(first_image),\
        "\nMax Pixel Value:", np.max(first_image))


    model = models.Sequential()
    model.add(layers.Rescaling(scale=1./255, input_shape=(180,180,3)))

    model.add(layers.Convolution2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))


    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    EPOCHS = 20

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS
                        )

    train_score = model.evaluate(train_ds)
    print(f"\nTrain score: {train_score[0]}")
    print(f'Train accuracy: {train_score[1]}\n')

    val_score = model.evaluate(val_ds)
    print("\nValidation score:", val_score[0])
    print('Validation accuracy:', val_score[1])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(loss)+1)

    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

experiment_1()

def experiment_2():
    data_augmentation = models.Sequential(name="data_augmentation")

    data_augmentation.add(layers.InputLayer(input_shape=INPUT_SHAPE))
    data_augmentation.add(layers.RandomFlip(mode='horizontal'))
    data_augmentation.add(layers.RandomRotation(0.1))
    data_augmentation.add(layers.RandomZoom(0.1))
    data_augmentation.add(layers.Rescaling(1./255))

    tf.get_logger().setLevel('ERROR')

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy())
            plt.axis("off")
        break
    plt.show()
    model = models.Sequential()

    model.add(data_augmentation)

    model.add(layers.Convolution2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())


    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax')) 


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    EPOCHS = 20

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    train_score = model.evaluate(train_ds)
    print(f"\nTrain score: {train_score[0]}")
    print(f'Train accuracy: {train_score[1]}\n')

    val_score = model.evaluate(val_ds)
    print("\nValidation score:", val_score[0])
    print('Validation accuracy:', val_score[1])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(loss)+1)

    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()


    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


    TEST_DATA_DIR = "data/input/test_data"
    TEST_DATA_DIR = Path(TEST_DATA_DIR)
    test_ds = keras.utils.image_dataset_from_directory(
                            directory=TEST_DATA_DIR,
                            label_mode="categorical",
                            image_size=(IMG_HEIGHT, IMG_WIDTH),
                            seed=SEED)

    plt.figure(figsize=(10,12))

    images, labels = next(test_ds.as_numpy_iterator())

    for i in range(9):
        ax = plt.subplot(3,3, i + 1)

        idx = random.randint(0, len(labels)-1)
        plt.imshow(images[idx].astype("uint8"))

        pred_arr = model.predict(tf.expand_dims(images[idx], axis=0), verbose=0)
        pred = tf.argmax(pred_arr, axis=-1).numpy()[0]

        temp = labels[idx].tolist()
        plt.title(f"Predicted: {class_names[pred]}\nActual: {class_names[temp.index(max(temp))]}")
        plt.axis("off")
    plt.show()



    plt.figure(figsize=(20, 30))

    images, labels = next(test_ds.as_numpy_iterator())

    for i in range(32):
        ax = plt.subplot(8, 4, i + 1)

        idx = i
        plt.imshow(images[idx].astype("uint8"))

        pred_arr = model.predict(tf.expand_dims(images[idx], axis=0), verbose=0)
        pred = tf.argmax(pred_arr, axis=-1).numpy()[0]

        temp = labels[idx].tolist()
        plt.title(f"Predicted: {class_names[pred]}\nActual: {class_names[temp.index(max(temp))]}")
        plt.axis("off")
    plt.show()

experiment_2()
