#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Boston Dataset
fname = "data//boston.csv"
df = pd.read_csv(fname) 

#create IVs and DVs
X = df[['CRIM','CHAS','NOX', 'RM','AGE', 'DIS']]
y = df['MEDV']

#Create Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_regression = LinearRegression()

lrMod = lin_regression.fit(X, y)

#Predict the price for test data
#lrPred = lrMod.predict(X[1:5])

#print(lrPred)

#Let's convert this into a streamlit applicaiton
#pip install streamlit
import streamlit as st
import plotly.express as px
import altair as alt #pip install altair
from plotly import graph_objs as go

st.title("Boston Housing")
st.subheader("Predicting the price of a house in Boston: Using Linear Regression Model")

st.image("data//boston_house.png")

#Create a streamlit sidebar with radio button for navigation
nav = st.sidebar.radio("Navigation",["Home", "Prediction"])

if nav == 'Home':
    if st.checkbox("Show data"):
        #Display the dataframe in table format
         st.dataframe(df)
    
    if st.checkbox("Show map"):
        #Display the map of Boston
        city_data = df[['LON', 'LAT', 'MEDV']]
        city_data.columns = ['longitude', 'latitude', 'price']
        st.map(city_data)

if nav == 'Prediction':
    st.write("This is Prediction Screen")
    #Create a form for user input
    v_name = st.text_input("Enter Your Name")
    v_dob = st.date_input("Enter Date of Birth")
    v_email = st.text_input("Enter Your Email")

    #'CRIM','CHAS','NOX', 'RM','AGE', 'DIS'
    v_room = st.number_input("Enter Number of Rooms", min_value=1, max_value=10, value=5)
    v_age = st.slider("Enter Age of House in Years", min_value=0, max_value=100, value=10, step=1)
    v_dis = st.slider("Enter Distance to Employment Centers in KMs", min_value=0.0, max_value=25.0, value=5.0, step=0.5)
    on = st.toggle("The property is next to Charles River")
    if on:
        v_chas = 1
    else:
        v_chas = 0
    v_crim = st.number_input("Enter Crime Rate", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    v_NOX = st.number_input("Enter NOX", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    if st.button("Predict Price"):
        #Create a dataframe for the input values
        input_data = pd.DataFrame([[v_crim, v_chas, v_NOX, v_room, v_age, v_dis]], columns=['CRIM', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS'])
        #Predict the price using the model
        price = lrMod.predict(input_data)[0] * 1000  # Convert to dollars
        #Display the predicted price in dollars
        st.success(f"The predicted price of the house is ${price:,.2f}")
