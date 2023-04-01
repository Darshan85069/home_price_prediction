
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import streamlit as st


#Header
st.title('Bangalore House prediction')
#select box
data = pd.read_json('columns.json')
data = data['data columns'].values
locations = data[3:]


location = st.selectbox('Select Location',locations)
Sqft = st.slider('Select Area SqFt',300,3000)
bath = st.slider('No of Batroom',1,16)
bhk = st.slider('Bhk',1,16)
   

with open('model.pickle','rb') as f:
    model = pickle.load(f)
# model function define 
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(data==location)[0]

    x = np.zeros(len(data)) # creating empty array == x columns len
    x[0] = sqft # defining starting 3 columns
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0: 
        x[loc_index] = 1 #defining value 1 on the index from above other are remaing zeros

    return model.predict([x])[0]
Price = predict_price(location,Sqft,bath,bhk)

if st.button('Click to predict price'):
    st.success(f"Rs.{np.round(Price,decimals=5)} lakhs")


