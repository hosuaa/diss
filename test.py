import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications.imagenet_utils import decode_predictions

import matplotlib.pyplot as plt

#image(s) to classify
img = load_img("img/toaster.jpg", target_size = (224, 224))
plt.imshow(img)

#image preprocessing
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

#instantiate model, using pre trained classifier
model = ResNet50(weights="imagenet")

#predict image
preds = model.predict(img)
print("Predicted:", decode_predictions(preds, top=5)[0])