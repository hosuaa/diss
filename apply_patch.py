#classify image in img/
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet50 import decode_predictions

import matplotlib.pyplot as plt

#image(s) to classify
img = load_img("img/toaster.jpg", target_size = (224, 224))
plt.imshow(img)

#instantiate model, using pre trained classifier
model = ResNet50(weights="imagenet")
model.trainable = False

decode_predictions = tf.keras.applications.resnet50.decode_predictions

#image preprocessing
# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.resnet50.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

#img = img_to_array(img)
#img = np.expand_dims(img, axis=0)
img = preprocess(img)

img_probs = model.predict(img)

plt.figure()
plt.imshow(img[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(img_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()

#implement fast gradient sign 
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

#display patch
# Get the input label of the image.
img_index = 208 #208 #??? 
label = tf.one_hot(img_index, img_probs.shape[-1])
label = tf.reshape(label, (1, img_probs.shape[-1]))
perturbations = create_adversarial_pattern(img, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  #To change [-1, 1] to [0,1]

#display (patched) image
def display_images(image, description):
  _, label, confidence = get_imagenet_label(model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()

epsilons = [0, 0.00001, 0.005, 0.01]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = img + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])