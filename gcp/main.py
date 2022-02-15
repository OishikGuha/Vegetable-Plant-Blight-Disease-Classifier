from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from google.cloud import storage

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None

CLASS_NAMES = ['Potato with Early blight',
               'Potato with Late blight',
               'Healthy Potato',
               'Tomato with Early blight',
               'Tomato with Late blight',
               'Healthy Tomato']


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


class FileData(BaseModel):
    file: UploadFile


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    # file: FileData = File(...)
):
    global MODEL

    if MODEL is None:
        download_blob("veg-plant-classifier",
                      "/models/5.h5", "/tmp/")
        MODEL = tf.keras.models.load_model(
            "/tmp/5.h5")

    image = tf.image.resize(read_file_as_image(await file.read()), (256, 256))
    img_batch = np.expand_dims(image, 0)

    print(image)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': str(float(confidence) * 100) + "%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
