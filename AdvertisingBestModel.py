import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Advertising Best Model App")
st.write("This app predicts the **Sales** !")

st.sidebar.header('User Input Parameters') #sidebar

def user_input_features():
    TV = st.sidebar.slider('Tv', 17.2, 230.1, 180.80)
    Radio = st.sidebar.slider('Radio', 10.8, 45.9, 37.8)
    Newspaper = st.sidebar.slider('Newspaper', 45.1, 69.3, 58.4)
    data = {'TV': TV, #key:value
            'Radio': Radio,
            'Newspaper': Newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingBestModel.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(df) # testing (examination)

st.subheader('Prediction')
st.write(new_pred)


