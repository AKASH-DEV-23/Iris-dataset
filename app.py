import pickle 
import streamlit as st
import numpy as np


scaler_file = pickle.load(open('scaler.pkl', 'rb')) 
model_file = pickle.load(open('model.pkl', 'rb'))

flower_names = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}

def pred_output(user_input):

    scaled_input = scaler_file.transform(np.array(user_input).reshape(1, -1))
    y_pred = model_file.predict(scaled_input)

    return flower_names.get(y_pred[0], "Unknown")

def main(): 
    st.title('Iris Flower Prediction')

    
    sepal_length = st.number_input('Enter the Sepal Length (Cm)')
    sepal_width = st.number_input('Enter the Sepal Width (Cm)')
    petal_length = st.number_input('Enter the Petal Length (Cm)')
    petal_width = st.number_input('Enter the Petal Width (Cm)')

 
    if st.button('Predict'):
        user_input = [sepal_length, sepal_width, petal_length, petal_width]
        make_prediction = pred_output(user_input)
        st.success(make_prediction)
    
if __name__ == '__main__':
    main()
