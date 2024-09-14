import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

# Load in Data
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = keras.utils.normalize(X_train, axis=1)
X_test = keras.utils.normalize(X_test, axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.2, random_state=0)

# ======================MODEL BUILDING=========================
def invert_image(image):
    image = tf.cast(image * 255, tf.uint8)
    inverted_image = tf.numpy_function(np.invert, [image], tf.uint8)
    inverted_image = tf.ensure_shape(inverted_image, image.shape)
    return tf.cast(inverted_image, tf.float32) / 255.0

model = keras.Sequential([
    # Convolutional base
    layers.InputLayer(input_shape=(28, 28, 1)),

    # layers.Lambda(lambda x: tf.where(tf.random.uniform([]) > .5, invert_image(x), x)),

    # Block One
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(.3),

    # Block Two
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(.3),

    # Block Three
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(.3),

    # Block Four
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(.3),

    # Head
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(10, activation='softmax')
])

early_stopping = EarlyStopping(
    min_delta=.001,
    patience=3,
    restore_best_weights=True
)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)



history = model.fit(
    X_train, y_train, 
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping],
    epochs=30
)
history_frame = pd.DataFrame(history.history)
history_frame

history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()




# model.save('handwritten.keras')

# model = keras.models.load_model('handwritten.keras')




# ===================MODEL EVALUTION===================
loss, accuracy = model.evaluate(X_test, y_test)
print(f'loss: {loss}')
print(f'accuracy: {accuracy}')

# Best scores: 
# loss: 0.04188231751322746
# accuracy: 0.9879999756813049




# ===================PREDICTION===================
current_dir = os.path.dirname(os.path.abspath(__file__))
digits = os.path.join(current_dir, 'digits')

for image in os.listdir(digits):
    img_path = os.path.join(digits, image)
    img = Image.open(img_path).convert('L')

    img = ImageOps.fit(img, (28, 28), method=Image.LANCZOS, centering=(.5, .5))

    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display the image and the model's prediction
    plt.figure(figsize=(2, 2))
    plt.imshow(img[0], cmap='gray')
    plt.title(f'Model Prediction: {predicted_class}')
    plt.show()


