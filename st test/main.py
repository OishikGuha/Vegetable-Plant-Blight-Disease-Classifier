import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

tf.random.set_seed(42)
np.random.seed(42)


MODEL = tf.keras.models.load_model(
    "../saved_models/modelv6")

CLASS_NAMES = ['Potato with Early blight',
               'Potato with Late blight',
               'Healthy Potato',
               'Tomato with Early blight',
               'Tomato with Late blight',
               'Healthy Tomato']

st.header("Plant Disease Predictor")

image = st.file_uploader("Upload the image you want to be predicted here: ")


# After hours of hair tearing, I found this code on a github with the link:
# https://github.com/alvarobartt/tensorflow-serving-streamlit/blob/master/src/streamlit/utils.py
# So, thanks to @alvarrobatt for saving me hours with this. And yes, I learnt this function, not just simple copy-pasted it.
def image2tensor(image_as_bytes):
    """
    Receives a image as bytes as input, that will be loaded,
    preprocessed and turned into a Tensor so as to include it
    in the TF-Serving request data.
    """

    # Apply the same preprocessing as during training (resize and rescale)
    image = tf.io.decode_image(image_as_bytes, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image/255.

    # Convert the Tensor to a batch of Tensors and then to a list
    image = np.expand_dims(image, 0)
    return image


if image:
    st.image(image)
    if st.button("Predict!"):
        img = image2tensor(image.read())
        preds = MODEL.predict(img)

        print(preds)
        final_pred = CLASS_NAMES[np.argmax(preds)]
        st.write(final_pred)
