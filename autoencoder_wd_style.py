# pyright: reportUndefinedVariable=false
from tensorflow.keras.layers import * # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
from datetime import datetime

# loading syle_dataset
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

# Load Fashion-MNIST for style
(fashion_train, _), (fashion_test, _) = tf.keras.datasets.fashion_mnist.load_data()

fashion_train = fashion_train / 255.0
fashion_test  = fashion_test / 255.0

kth_style_images = load_kth_tips2_textures(
    root_dir="KTH-TIPS2",
    img_size=28,
    max_images=1000
)

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

encoder_input = Input(shape=(None, None, 3))
encoder_output = encoder_block(encoder_input)

encoder = Model(encoder_input, encoder_output, name="shared_encoder")
encoder.trainable = False

content_input = Input(shape=(None,None,3), name="content_image")
style_input = Input(shape=(None,None,3), name="style_image")

content_features = encoder(content_input)
style_features   = encoder(style_input)


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

# style_transfer_model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss="mse"
# )

style_transfer_model.summary()

"""Loss Definitions"""

# generated image - content image
def content_loss(content, generated, encoder):
    c_feat = encoder(content, training=False)
    g_feat = encoder(generated, training=False)
    return tf.reduce_mean(tf.square(c_feat - g_feat))

def gram_matrix(x):
    # x: (B, H, W, C)
    b, h, w, c = x.shape
    x = tf.reshape(x, [b, h*w, c])
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(h * w * c, tf.float32)

# using gram-matrices (apperently doesnt align very well with adain)
# def style_loss(style, generated, encoder):
#     s_feat = encoder(style, training=False)
#     g_feat = encoder(generated, training=False)

#     S = gram_matrix(s_feat)
#     G = gram_matrix(g_feat)

#     return tf.reduce_mean(tf.square(S - G))

# using mean_variance
def style_loss(style, generated, encoder, eps=1e-5):
    s_feat = encoder(style, training=False)
    g_feat = encoder(generated, training=False)

    s_mean, s_var = tf.nn.moments(s_feat, axes=[1, 2])
    g_mean, g_var = tf.nn.moments(g_feat, axes=[1, 2])

    mean_loss = tf.reduce_mean(tf.square(s_mean - g_mean))
    var_loss  = tf.reduce_mean(tf.square(s_var - g_var))

    return mean_loss + var_loss


# train_step definition
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(content, style):
    with tf.GradientTape() as tape:
        generated = style_transfer_model(
            [content, style], training=True
        )

        c_loss = content_loss(content, generated, encoder)
        s_loss = style_loss(style, generated, encoder)

        total_loss = c_loss + 2.0 * s_loss

    grads = tape.gradient(
        total_loss,
        style_transfer_model.trainable_variables
    )
    optimizer.apply_gradients(
        zip(grads, style_transfer_model.trainable_variables)
    )

    return total_loss

"""Dataset & Training"""

# shuffling training dataset, so it doesnt have sequential corelations
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")

    style_indices = np.random.randint(
        0, len(kth_style_images), size=len(x_train)
    )
    style_train = kth_style_images[style_indices]

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, style_train)
    ).shuffle(1000).batch(16)

    for step, (c, s) in enumerate(dataset):
        loss = train_step(c, s)
        if step % 100 == 0:
            print(f"step {step} | loss {loss.numpy():.4f}")


# old
# style_transfer_model.fit(
#     # [x_train, style_train],   # the content and style are different now
#     [x_train, x_train],
#     x_train,
#     epochs=3,
#     batch_size=16
# )

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


from datetime import datetime

name = datetime.now().strftime("model_%Y%m%d_%H%M.keras")
style_transfer_model.save(f"models/{name}")


# style_transfer_model.save("style_transfer_adain_3_layers_no_input_dim_wd_updated_now_training_wd_styles_mse_12.keras")