from tkinter import N
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

# bucket_name = "veg-plant-classifier"
CLASS_NAMES = ["Potato with Early blight",
               "Potato with Late blight",
               "Healthy Potato",
               "Tomato with Early blight",
               "Tomato with Late blight",
               "Healthy Tomato"]

model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def predict(request):
    global model
    if model is not None:
        download_blob(
            bucket_name,
            "models/modelv5. longer, stronger, bonger",
            "/tmp/modelv5. longer, stronger, bonger"
        )
        model = tf.keras.load_model("/tmp/modelv5. longer, stronger bonger")

    image = request.files["files"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image/255
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    print(predictions)

    class_name = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": class_name, "confidence": confidence}
