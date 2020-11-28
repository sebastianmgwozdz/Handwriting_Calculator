import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import os
import cv2
from sklearn.model_selection import train_test_split


class ConvolutionalNN:
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(500, activation='relu'))
        self.model.add(layers.Dense(13, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    def train(self, x, y, epochs):
        self.model.fit(x, y, epochs=epochs, batch_size=10)

    def predict(self, x):
        return self.model.predict(x)

    def predicted_value(self, x):
        pred = self.model.predict(x)
        max = 0
        for i in range(13):
            if pred[0,i] > pred[0,max]:
                max = i

        return max


    def load(self, json_path, weight_path):
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = models.model_from_json(loaded_model_json)
        self.model.load_weights(weight_path)
        print("loaded weights")

    def save(self, json_path, weight_path):
        model_json = self.model.to_json()
        with open(json_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(weight_path)
        print("saved model")

def load_images_and_labels(directory, labels, dimensions):
    x = None
    y = []
    for folder in os.listdir(directory):
        if folder.startswith("."):
            continue
        images = load_images_from_folder(os.path.join(directory, folder), dimensions)

        if x is None:
            x = images
        else:
            x = combine_rows(x, images)

        label = labels[folder]
        for i in range(len(images)):
            y.append(label)

    y_as_np = np.reshape(y, (len(y),))
    return x, y_as_np

def combine_rows(old, new):
    return np.append(old, new, axis=0)

def load_images_from_folder(path, dimensions):
    images = None
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img)

        if img is not None:
            img = cv2.resize(img, dimensions)
            img = np.reshape(img, (1,) + dimensions)

            if images is None:
                images = img
            else:
                images = combine_rows(images, img)
    return images

def normalize(data):
    scaled = data / 255
    return np.expand_dims(scaled, -1)