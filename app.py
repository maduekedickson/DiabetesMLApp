import numpy as np
import pickle
import streamlit as st
import os

# load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))



# Load the image at the top of the app
image = Image.open('./image.png')

# Display the image
st.image(image, use_column_width=True)





# Create the function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # title
    st.title("Diabetes Prediction Application")

    # getting information from user input
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    Glucose = st.number_input("Glucose level", min_value=0, max_value=200, value=100, step=1)
    BloodPressure = st.number_input("Blood Pressure Value", min_value=0, max_value=150, value=80, step=1)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
    Insulin = st.number_input("Insulin level", min_value=0, max_value=900, value=30, step=1)
    BMI = st.number_input("Body Max Index", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    # Code for Prediction
    Diagnosis = ''

    # Create the predictions button
    if st.button("Diabetes Test Result"):
        Diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(Diagnosis)

if __name__ == '__main__':
    main()
