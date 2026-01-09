from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np

def load_kth_tips2_textures(root_dir, img_size=28, max_images=1000):
    images = []

    for material in os.listdir(root_dir):
        material_path = os.path.join(root_dir, material)
        if not os.path.isdir(material_path):
            continue

        # NEW: loop over sample folders (sample_a, sample_b, ...)
        for sample in os.listdir(material_path):
            sample_path = os.path.join(material_path, sample)
            if not os.path.isdir(sample_path):
                continue

            for fname in os.listdir(sample_path):
                img_path = os.path.join(sample_path, fname)

                # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if img is None:
                    continue

                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                images.append(img)

                if len(images) >= max_images:
                    break

            if len(images) >= max_images:
                break

        if len(images) >= max_images:
            break

    images = np.array(images)  # (N, H, W, 1)
    print(f"Loaded {len(images)} texture images")
    return images

dataset = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = dataset.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = np.repeat(x_train[..., None], 3, axis=-1)
x_test  = np.repeat(x_test[..., None], 3, axis=-1)


# adding channel dimensions
# x_train = x_train[..., None]
# x_test  = x_test[..., None]

# Load Fashion-MNIST for style
(fashion_train, _), (fashion_test, _) = tf.keras.datasets.fashion_mnist.load_data()

fashion_train = fashion_train / 255.0
fashion_test  = fashion_test / 255.0

# add channel dimension
# fashion_train = fashion_train[..., None]
# fashion_test  = fashion_test[..., None]

kth_style_images = load_kth_tips2_textures(
    root_dir="KTH-TIPS2",
    img_size=28,
    max_images=1000
)

content_input = Input(shape=(None,None,3), name="content_image")
style_input = Input(shape=(None,None,3), name="style_image")

def encoder_block(inputs):
    x = Conv2D(32, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool2D((2,2))(x)   # 14x14

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPool2D((2,2))(x)   # 7x7

    # x = Conv2D(128, (3,3), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = MaxPool2D((2,2))(x)   # 3x3

    return x  # shape: (7,7,64) now (3,3,128)

content_features = encoder_block(content_input)
style_features   = encoder_block(style_input)

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        content, style = inputs

        c_mean = tf.reduce_mean(content, axis=[1, 2], keepdims=True)
        c_std  = tf.math.reduce_std(content, axis=[1, 2], keepdims=True)

        s_mean = tf.reduce_mean(style, axis=[1, 2], keepdims=True)
        s_std  = tf.math.reduce_std(style, axis=[1, 2], keepdims=True)

        normalized = (content - c_mean) / (c_std + self.epsilon)
        return normalized * s_std + s_mean

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon
        })
        return config


t = AdaIN()([content_features, style_features])

def decoder_block(inputs):
    # x = Conv2DTranspose(128, 3, strides=2, padding="valid")(inputs)  # 7x7
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(64, (4,4), strides=2, padding="same")(inputs)  # 14x14
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(32, (4,4), strides=2, padding="same")(x)  # 28x28
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return x

output_image = decoder_block(t)

style_transfer_model = Model(
    inputs=[content_input, style_input],
    outputs=output_image,
    name="Simple_Style_Transfer_Model"
)

style_transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

style_transfer_model.summary()

style_indices = np.random.randint(
    0, len(kth_style_images), size=len(x_train)
)
style_train = kth_style_images[style_indices]

"""Training"""
style_transfer_model.fit(
    # [x_train, style_train],   # the content and style are different now
    [x_train, x_train]
    x_train,
    epochs=3,
    batch_size=16
)

"""Prediction"""
n = 10
content_images = x_test[:n]
style_images = kth_style_images[:n]
style_images = style_images * 1.25
style_images = np.clip(style_images, 0.0, 1.0)
stylized_pred = style_transfer_model.predict(
    [content_images, style_images]
)

plt.figure(figsize=(20,6))

for i in range(n):
    # Content Image
    ax = plt.subplot(3, n, i + 1)
    ax.set_title("Content")
    plt.imshow(content_images[i])
    ax.axis("off")

    # Style Image
    ax = plt.subplot(3, n, i + 1 + n)
    ax.set_title("Style")
    plt.imshow(style_images[i])
    ax.axis("off")

    # Stylized Output
    ax = plt.subplot(3, n, i + 1 + 2*n)
    ax.set_title("Output")
    plt.imshow(stylized_pred[i])
    ax.axis("off")

plt.tight_layout()
plt.savefig("style_transfer_result_12.png")
plt.show()

style_transfer_model.save("style_transfer_adain_3_layers_no_input_dim_wd_updated_now_training_wd_styles_mse_12.keras")