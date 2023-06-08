#given an image and its label output what it is (whether it is correct)
import cv2, socket, pickle #,numpy
import time
#---------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

#run pip install git+https://github.com/nottombrown/imagenet_stubs in the directory of the notebooks server
import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import name_to_label

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from art.estimators.classification import TensorFlowV2Classifier, EnsembleClassifier
from art.attacks.evasion import AdversarialPatch

target_name = 'toaster'
image_shape = (224, 224, 3)
clip_values = (0, 255)
nb_classes  =1000
batch_size = 16
scale_min = 0.4
scale_max = 1.0
rotation_max = 22.5
learning_rate = 5000.
max_iter = 500

model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

mean_b = 103.939
mean_g = 116.779
mean_r = 123.680

tfc = TensorFlowV2Classifier(model=model, loss_object=None, train_step=None, nb_classes=nb_classes,
                             input_shape=image_shape, clip_values=clip_values, 
                             preprocessing=([mean_b, mean_g, mean_r], np.array([1.0, 1.0, 1.0])))

images_list = list()

for image_path in imagenet_stubs.get_image_paths():
    im = image.load_img(image_path, target_size=(224, 224))
    im = image.img_to_array(im)
    im = im[:, :, ::-1].astype(np.float32) # RGB to BGR
    im = np.expand_dims(im, axis=0)
    images_list.append(im)

images = np.vstack(images_list)

#for displaying patch
def bgr_to_rgb(x):
    return x[:, :, ::-1]

#def gen_patch_and_apply(image, img_name):
#    ap = AdversarialPatch(classifier=tfc, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max,
#                      learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
#                      patch_shape=(224, 224, 3))
#
#    label = name_to_label(target_name)
#    y_one_hot = np.zeros(nb_classes)
#    y_one_hot[label] = 1.0
#    y_target = np.tile(y_one_hot, (images.shape[0], 1))
#
#   patch, patch_mask = ap.generate(x=image, y=y_target)
#    patched_image = ap.apply_patch(image, scale=0.5)
#
#    path_to_save = "img_with_patch/" + img_name
#    cv2.imwrite(path_to_save,patched_image)

def predict_model(classifier, image):
    #plt.imshow(bgr_to_rgb(image.astype(np.uint8)))
    #plt.show()
    
    image = np.copy(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = classifier.predict(image)
    
    top = 5
    prediction_decode = decode_predictions(prediction, top=top)[0]
    print('Predictions:')
    
    lengths = list()
    for i in range(top):
        lengths.append(len(prediction_decode[i][1]))
    max_length = max(lengths)
    
    for i in range(top):
        name = prediction_decode[i][1]
        name = name.ljust(max_length, " ")
        probability = prediction_decode[i][2]
        output_str = "{} {:.2f}".format(name, probability)
        print(output_str)
#------------------------------------------------------------
while True:
    img_name = input("Enter image to classify (in img/...):")
    img_path = "img/" + img_name
    img = cv2.imread(img_path)
    resize_img=cv2.resize(img, (224,224))
    predict_model(tfc, resize_img)


