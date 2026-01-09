import os
import tensorflow as tf

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = 2

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

"""Seeding"""
# tf.random.seed(42)
tf.random.set_seed(42)

"""Dataset"""
dataset = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = dataset.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

"""Hyperparameters"""

H = 28
W = 28
C = 1
latent_dimension = 128
lr = 1e-3
batch_size = 32
epochs = 5

"""Model"""
inputs = Input(shape = (H,W,C))
x = inputs

##
x = Conv2D(32, (3,3), padding='same')(x)    #28
x = BatchNormalization()(x)                 #28
x = LeakyReLU(alpha=0.2)(x)                 #28
x = MaxPool2D((2,2))(x)                     #14

##
x = Conv2D(64, (3,3), padding='same')(x)    #14
x = BatchNormalization()(x)                 #14
x = LeakyReLU(alpha=0.2)(x)                 #14
x = MaxPool2D((2,2))(x)                     #7,7,64 -- 64 layers
print(x.shape)

##
x = Flatten()(x)                            # (?, 7*7*4)
units = x.shape[1]                          # 7*7*64
x = Dense(latent_dimension, name = 'Latent')(x)
x = Dense(units)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = Reshape((7,7,64))(x)
print(x.shape)

##
x = Conv2DTranspose(32, (4,4), strides = 2, padding = "same")(x)   # (?,14,14,32)
print(x.shape)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = 0.2)(x)

x = Conv2DTranspose(1, (4,4), strides = 2, padding = "same")(x)   # (?,28,28,32)
print(x.shape)
x = BatchNormalization()(x)
x = Activation("sigmoid")(x)

outputs = x
print(x.shape)

autoencoder = Model(inputs, outputs, name = "Conv_Autoencoder")
autoencoder.compile(optimizer = Adam(lr), loss = "binary_crossentropy")
autoencoder.summary()

"""Training"""
autoencoder.fit(
    x_train,
    x_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (x_test, x_test)
)

"""Prediction"""
test_pred_y = autoencoder.predict(x_test)

"""Visulaize"""
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    """Original"""
    ax = plt.subplot(2, n, i+1)
    ax.set_title("Original Image")
    plt.imshow(x_test[i].reshape(H,W))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    """Predicted"""
    ax = plt.subplot(2, n, i+1+n)
    ax.set_title("Original Image")
    plt.imshow(test_pred_y[i].reshape(H,W))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig("conv_autoencoder.png")