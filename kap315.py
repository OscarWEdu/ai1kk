import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def dataframe_encoding(fuel_encoder, dataset):
    encoded_fuel = fuel_encoder.transform(dataset[['Fuel_Type']])
    encoded_df = pd.DataFrame(
        encoded_fuel,
        columns=fuel_encoder.get_feature_names_out(['Fuel_Type']),
        index=dataset.index
    )

    formatted_dataset = pd.concat(
        [dataset.drop(columns='Fuel_Type'), encoded_df],
        axis=1
    )

    return formatted_dataset

@st.cache_resource
def initialize_model():

    carDataset = pd.read_csv(r'C:car_price_dataset.csv', sep=";")
    print(carDataset.head())

    carDataset = carDataset.drop(columns=['Brand', 'Model', 'Transmission'])
    X = carDataset.drop(columns=['Price'])
    y = carDataset['Price']
    carDataset.head()

    fuel_encoder = OneHotEncoder(sparse_output=False)
    fuel_encoder.fit(X[['Fuel_Type']])

    X = dataframe_encoding(fuel_encoder, X)

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    return lin_reg, fuel_encoder

def perform_prediction(car_inp):
    lin_reg, fuel_enc = initialize_model()
    formatted_car = dataframe_encoding(fuel_enc, car_inp)
    return lin_reg.predict(formatted_car)


st.title("Car Price Predictor")
st.markdown("Add your car info below:")

year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2026,
    value=2020,
    step=1
)

engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Diesel", "Electric", "Hybrid", "Petrol"]
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=50000,
    step=1000
)

doors = st.number_input(
    "Number of Doors",
    min_value=2,
    max_value=6,
    value=4,
    step=1
)

owner_count = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    value=1,
    step=1
)


if st.button("Predict"):
    car_inp = pd.DataFrame({
        'Year': [year],
        'Engine_Size': [engine_size],
        'Fuel_Type': [fuel_type],
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count]
    })
    st.write(f"Predicted price: {perform_prediction(car_inp)[0]:,.0f}")