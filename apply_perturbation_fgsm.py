#classify image in img/
import numpy as np
import tensorflow as tf
import matplotlib as mpl

import random

import csv

from tensorflow import keras

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications.imagenet_utils import decode_predictions
#from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]



loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    #print(prediction)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


def display_images(image, description, ground_truth, eps):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.subplot(1,2,1)
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  #what top classes the classifier thinks the adversarial image is                                                
  plt.subplot(1,2,2)
  plt.xlabel('Prediction')
  plt.ylabel('Probability')
  plt.title('What am I looking at?')
  top = 5
  prediction_decode = decode_predictions(pretrained_model.predict(image), top=top)[0]    
  lengths = list()
  for i in range(top):
      lengths.append(len(prediction_decode[i][1]))
  max_length = max(lengths)    
  x=[]
  y=[]
  for i in range(top):
      name = prediction_decode[i][1]
      name = name.ljust(max_length, " ")
      probability = prediction_decode[i][2]
      x.append(name)
      y.append(probability)
  plt.bar(x,y)
  plt.show()

  if ground_truth==label:
    print("Correct prediction at "+str(eps)+" epsilon")
  else:
    print("Incorrect prediction at "+str(eps)+" epsilon")
for i in range(10):
  #choose a random image
  i = random.randrange(1,50000)
  print(i)
  with open("ILSVRC/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt") as f:
    lines = f.readlines()
    image_index = int(lines[i-1])  #line no. i in validation ground truth
    print(image_index)

  image_num=str(i)
  while i<10000:
    i=i*10
    image_num="0"+image_num
  #do the whole process on the image
  #image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
  image_path = "kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_000"+image_num+".JPEG"

  #image_name = "banana"
  a = np.loadtxt("ILSVRC/devkit/data/map_clsloc.txt", dtype='str')
  print(a)
  image_name_index=np.where(a[:,1]==str(image_index))
  image_name=a[image_name_index][0][2]
  #print(image_name)

  image_raw = tf.io.read_file(image_path)
  image = tf.image.decode_image(image_raw)

  image = preprocess(image)
  image_probs = pretrained_model.predict(image)

  plt.figure()
  plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
  _, image_class, class_confidence = get_imagenet_label(image_probs)
  plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
  plt.show()

  label = tf.one_hot(image_index, image_probs.shape[-1])
  label = tf.reshape(label, (1, image_probs.shape[-1]))

  perturbations = create_adversarial_pattern(image, label)
  #plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

  epsilons = [0, 0.01, 0.1, 0.5, 1, 5, 30]
  descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                  for eps in epsilons]

  for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i], image_name, eps)
