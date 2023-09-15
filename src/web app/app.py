import streamlit as st
import numpy as np
import cv2
from skimage import io
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array

st.markdown('<h1 style="color:black;">Image Classification Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> COVID and Non-COVID</h3>', unsafe_allow_html=True)

img_height = 224
img_width = 224
n_classes=1

model = Sequential()
# add the convolutional layer
# filters, size of filters,padding,activation_function,input_shape
model.add(Input(shape=(img_width,img_height,3), name="layer1"))
model.add(Conv2D(64, (3,3), padding='SAME', activation='relu', name="layer2"))
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2), name="max_pool_layer1"))
# place a dropout layer
#model.add(Dropout(0.2))
# add another convolutional layer
model.add(Conv2D(64, (3,3), padding='SAME', activation='relu', name="layer3"))
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2), name="max_pool_layer2"))
# place a dropout layer
#model.add(Dropout(0.5))
# add another convolutional layer
model.add(Conv2D(64, (3,3), padding='SAME', activation='relu', name="layer4"))
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2), name="max_pool_layer3"))
# Flatten layer
model.add(Flatten(name="flatten_layer"))
# add a dense layer : amount of nodes, activation
model.add(Dense(512, activation='relu', name="layer6"))
model.add(Dense(256,activation='relu', name="layer7"))
model.add(Dense(128,activation='relu', name="layer8"))
model.add(Dropout(0.5, name="drop_out_layer"))
model.add(Dense(n_classes,activation='sigmoid', name="output_layer"))
model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the model weights
model.load_weights('covid_classification_new_model_weights_omdena_custom_clahe.hdf5')

upload= st.file_uploader('Insert X-ray image of a patient for classification', type=["png", "jpg", "jpeg"])
c1, c2= st.columns(2)
c1.header('Input Image')
c2.header('Output')
if upload is not None:
  img = io.imread(upload)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(img)
  image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  image = cv2.resize(image, (img_width,img_height))
  # img = load_img(image, target_size=(img_width,img_height))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  c1.image(img)
  prediction = model.predict(image)
  if prediction == 0:
    c2.write('COVID Not Detected')
  else:
    c2.write('COVID Detected')
