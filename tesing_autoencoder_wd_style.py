import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# =========================
# Custom AdaIN Layer
# =========================
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

# =========================
# Load Model
# =========================
MODEL_PATH = "style_transfer_adain_3_layers_no_input_dim.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AdaIN": AdaIN}
)

print("✅ Model loaded successfully")


# =========================
# Image Loader
# =========================
def load_image(path, img_size=28):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"❌ Could not read image: {path}")

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    # img = cv2.resize(img, (img_size, img_size))

    # normalize
    img = img / 255.0

    # add channel + batch dim
    img = img[..., None]
    img = np.expand_dims(img, axis=0)

    return img


# =========================
# Input Paths
# =========================
CONTENT_IMAGE_PATH = "inputs/content.jpg"
# STYLE_IMAGE_PATH   = "inputs/style.jpg"
STYLE_IMAGE_PATH   = "inputs/rainbow.png"

OUTPUT_PATH = "outputs/result.png"
os.makedirs("outputs", exist_ok=True)


# =========================
# Load Images
# =========================
content_img = load_image(CONTENT_IMAGE_PATH)
style_img   = load_image(STYLE_IMAGE_PATH)

print("✅ Images loaded")


# =========================
# Run Style Transfer
# =========================
output = model.predict([content_img, style_img])

output_img = output[0].squeeze()  # (28,28)


# =========================
# Save Output
# =========================
plt.imsave(OUTPUT_PATH, output_img, cmap="gray")
print(f"✅ Output saved to {OUTPUT_PATH}")


# =========================
# Show Result
# =========================
plt.figure(figsize=(8,3))

plt.subplot(1,3,1)
plt.title("Content")
plt.imshow(content_img[0].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Style")
plt.imshow(style_img[0].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Output")
plt.imshow(output_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()