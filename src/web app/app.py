import streamlit as st
import numpy as np
from skimage import io
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

st.markdown('<h1 style="color:black;">COVID X-Ray Classification Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">This classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> COVID and Non-COVID</h3>', unsafe_allow_html=True)

img_height = 224
img_width = 224


# Load the model
model = load_model('xception.h5', compile=False)

upload= st.file_uploader('Upload X-ray image of a patient for classification', type=["png", "jpg", "jpeg"])
c1, c2= st.columns(2)
c1.header('Uploaded Image')
c2.header('Predicted Class')
if upload is not None:
  img = load_img(upload, target_size=(img_width,img_height))
  image = img_to_array(img)
  image = np.expand_dims(image, axis=0)
  c1.image(img)
  prediction = model.predict(image)
  print(prediction)
  if prediction == 0:
    c2.write('COVID Not Detected')
  else:
    c2.write('COVID Detected')


