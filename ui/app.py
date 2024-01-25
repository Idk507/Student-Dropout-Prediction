import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
# Load the saved model
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



model = joblib.load('log_model.pkl')

# Function to preprocess input data
def preprocess_input(input_data):
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled

# Streamlit UI
st.title("Student Dropout Prediction Web App")

# Sidebar for user input
st.sidebar.header("User Input")

# Allow user to upload a CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    user_data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader("Uploaded Data")
    st.write(user_data)

    # Preprocess user input
    user_data_processed = preprocess_input(user_data)

    # Make predictions using the loaded model
    prediction = model.predict(user_data_processed)

    # Display the prediction
    st.subheader("Prediction Result")
    class_mapping = {0: 'Dropout', 1: 'Graduate', 2: 'Enrolled'}  
    predicted_class = prediction[0]
    predicted_class_label = class_mapping[predicted_class]
    st.write(f"The predicted class is: {predicted_class_label}")


