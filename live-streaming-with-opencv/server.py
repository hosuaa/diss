#recieve video stream from client. classify objects per frame (every 0.5s) and output the top 5 predictions in a bar chart (likelyhood of being correct)
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


model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

#for displaying patch
def bgr_to_rgb(x):
    return x[:, :, ::-1]

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
    
    x=[]
    y=[]
    for i in range(top):
        name = prediction_decode[i][1]
        name = name.ljust(max_length, " ")
        probability = prediction_decode[i][2]
        x.append(name)
        y.append(probability)
        #output_str = "{} {:.2f}".format(name, probability)
        #print(output_str)
    print(x)
    print(y)
    plt.bar(x,y)
    plt.pause(0.5)
    
plt.xlabel('Prediction')
plt.ylabel('Probability')
plt.title('What am I looking at?')
plt.show(block=False)  
#------------------------------------------------------------
s=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
ip="127.0.0.1"
port=6666
s.bind((ip,port))
while True:
    print("recieving camera input")
    x=s.recvfrom(1000000)
    print("1")
    clientip = x[1][0]
    data=x[0]
    #print(data)
    data=pickle.loads(data)
    #print(type(data))
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print("2")
    resize=cv2.resize(data, (224,224))
    print("3")
    #cv2.imshow('server', resize) to open image
    #----------
    predict_model(model, resize)
    plt.clf()
    #-----------
    #time.sleep(5)
    print("4")
    if cv2.waitKey(10) == 13:
        plt.close('all')
        break
cv2.destroyAllWindows()
