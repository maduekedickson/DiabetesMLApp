import numpy  as np
import pickle
import streamlit as st

# load the saved model
#loded_model = pickle.load(open('Diabetes/trained_model.sav', 'rb'))

import os
model_path = os.path.join(os.path.dirname(__file__), 'Diabetes/trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))





# Create the function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return'The person is diabetic'
    

def main():
    # title
    st.title("Diabetes Prediction Application")

    #getting information

    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose level")
    BloodPressure = st.number_input("Blood Pressure Value")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin level")
    BMI = st.number_input("Body Max Index")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

    # Code for Prediction
    Diagnosis = ''

    # Create the predictions button
    if st.button("Diabetes Test Result"):
        Diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


    st.success(Diagnosis)

if __name__ == '__main__':
    main()
