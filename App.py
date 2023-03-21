import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# load the pre-trained linear regression model
model = joblib.load('model.pkl')
# Read devaluation_ratio.csv
devaluation_ratio = pd.read_csv('devaluation_ratio.csv')
# Import the scaler
scaler = joblib.load('scaler.pkl')

# Devaluation ratio
def get_devaluation_ratio(car_name):
    try:
        return devaluation_ratio[devaluation_ratio['Car_Name'] == car_name]['devaluation_ratio'].values[0]
    except:
        return "Car model not found"
    
# define a function to predict the selling price of the car
def predict_selling_price(year, present_price, car_model):
    car = pd.DataFrame([[year, present_price, car_model]], columns=['Year', 'Present_Price', 'Car_Name'])
    # create a new column 'Years_Old' based on the year of purchase
    car['Years_Old'] = 2023 - car['Year']
    # create a new column 'Devaluation_Ratio' based on the car model
    car['Devaluation_Ratio'] = car['Car_Name'].apply(get_devaluation_ratio)
    # Apply the scaler
    car = scaler.transform(car[['Present_Price','Years_Old', 'Devaluation_Ratio']])

    prediction = model.predict(car)
    return prediction[0]

# create a streamlit app
def app():
    st.title("Car Selling Price Predictor")

    # define input fields for user input
    year = st.slider('Select Year of Purchase', 2003, 2023, 2010)
    present_price = st.number_input('Enter Present Price', min_value=0.1, max_value=100.0, step=0.1, value=5.0)
    car_models = devaluation_ratio['Car_Name'].unique()
    car_model = st.selectbox('Select Car Model', car_models)

    # when the user clicks the 'Predict Selling Price' button, predict the selling price of the car
    if st.button('Predict Selling Price'):
        selling_price = predict_selling_price(year, present_price, car_model)
        st.write('Estimated Selling Price: {:.2f}'.format(selling_price))

if __name__ == '__main__':
    app()


