from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(
    "D:\machine learning projects\\agriculture\saved_models\modelv5. longer, stronger, bonger")

CLASS_NAMES = ['Potato with Early blight',
               'Potato with Late blight',
               'Healthy Potato',
               'Tomato with Early blight',
               'Tomato with Late blight',
               'Healthy Tomato']


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
    image = tf.image.resize(read_file_as_image(await file.read()), (256, 256))
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
        # 'confidence': str(float("{0:.2f}".format(confidence)) * 100) + "%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
