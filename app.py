# libraries
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import plotly.express as px






# st.header

st.header("COVID detector based on chest-X-ray")

#st.image('./Downloads/12.png')

# sidebar-info
st.sidebar.title('Control Panel')



# Load the model
model = load_model('keras_model.h5')



# labels
labels=['COVID-19', 'Normal', 'Viral pneumonia']

file = st.sidebar.file_uploader('Upload chest-X-ray', type=['png', 'jpeg', 'gif', 'svg'])

st.sidebar.title('About Project')
st.sidebar.text("""This Covid diagnostic program 
is Artificial Intelligence 
Technology which was built 
on the basis of more than 1,000
X-ray data """)
#st.sidebar.image('./Downloads/1.png')

st.sidebar.code('Author: Musaeva Gulrukhsor\nEmail: sarah579395@gmail.com ')

tips_normal = "Your X-Ray is Normal, Stay Safe 👋🏻"
tips_covid = " Covid was found on your X-Ray, please contact nearest healthcare centre🏥 "
tips_viral = " Viral Prenumonia was found in your X-Ray, please contact nearest healthcare centre 🏥  "





if file:
    st.image(file)
    # PIL convert
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(file).convert('RGB')
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # prediction
    pred=model.predict(data)[0]
    st.text([i*100 for i in pred]) # to display probs
    st.success(f"Prediction: {labels[pred.argmax()]}")

    #plotting
    fig=px.bar(x=[i*100 for i in pred], y=labels)
    st.plotly_chart(fig)
    
