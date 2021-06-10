import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe=pickle.load(open('pipe.pkl','rb'))
st.header('Car Price Predictor')

# Inputs t0o be taken
# year
year=st.number_input('Enter Year')

# km_driven
kms=st.number_input('Kilometer Travelled')

#fuel
fuel= st.selectbox('Choose Fuel type',('Diesel','Petrol'))

# seller_type
seller_type= st.selectbox('Choose Seller type',('Individual','Dealer'))

# transmission
transmission= st.selectbox('Transmission',('Manual','Automatic'))

# owner
owner= st.selectbox('Owner',('First Owner','Second Owner','Third Owner'))

# mileage
mileage=st.number_input('Mileage of Car')

# engine
engine=st.number_input('Engine')

# max_power
max_power=st.number_input('Maximum Power')

# seats
seats=st.number_input('Seats in the Car')

# brand
brand= st.selectbox('Select Car Brand',('Maruti','Hyundai','Mahindra','Tata','Ford','Honda','Toyota','Renault','Chevrolet','Volkswagon'))

if st.button('Predict Price'):
    # form a numpy array(1,11)
    input= np.array([[year,kms,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats,brand]])
    input= pd.DataFrame(input,columns=['year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','brand'])
    y_pred= pipe.predict(input)
    st.title("Rs "+str(np.round(y_pred[0])))

