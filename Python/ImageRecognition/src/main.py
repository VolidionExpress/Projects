import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 \
import MobileNetV2, preprocess_input, \
decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model = MobileNetV2(weights='imagenet')

def load_and_prepare_image(img_path):
    img = image.load_img(img_path,
                         target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image_class(img_path):
    prepared_image = load_and_prepare_image(img_path)
    predictions = model.predict(prepared_image)
    decoded_predictions = decode_predictions(
		predictions, top=3)[0]
    
    for i, pred in enumerate(decoded_predictions):
        print(f"Prediction {i+1}: {pred[1]} "
           f"with a confidence of "
           f"{pred[2]*100:.2f}%")
     
img_path = "data/dog.jpg"
predict_image_class(img_path)