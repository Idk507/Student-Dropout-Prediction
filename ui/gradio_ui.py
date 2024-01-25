import gradio as gr
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the best model
best_model = load_model('best_model.h5')

# Instantiate StandardScaler for preprocessing
scaler = StandardScaler()

def predict_dropout(file):
    try:
        # Read the CSV file
        data = pd.read_csv(file)
        
        # Check if the file is empty
        if data.empty:
            raise ValueError("The uploaded CSV file is empty.")
        
        # Preprocess the input data
        data = data.drop(['International', 'Nacionality', "Father's qualification", 'Curricular units 1st sem (credited)',
                          'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Course',
                          'Educational special needs', 'Unemployment rate', 'Inflation rate'], axis=1)
        data = scaler.transform(data)

        # Make predictions
        prediction = best_model.predict(data)
        predicted_classes = tf.argmax(prediction, axis=1).numpy()

        # Map the predicted classes to the corresponding labels
        labels = {0: 'Dropout', 1: 'Graduate'}
        predicted_labels = [labels[class_] for class_ in predicted_classes]

        return predicted_labels
    except Exception as e:
        return f"Error: {str(e)}"


# Define Gradio interface
iface = gr.Interface(fn=predict_dropout,
                     inputs="file",
                     outputs="text",
                     live=True)

# Launch the Gradio interface
iface.launch()
