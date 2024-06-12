import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Class names and mappings specific to Yakshagana project
class_names = ['Bheemamudi', 'Devi mudi', 'Kesaritatti', 'Kirana', 'Kolukireeta', 'Kuttari', 'Kombu', 'Pakdi Shaiva vaishnava', 'Turai']
vesha_mapping = {
    'Bheemamudi': 'Rakshasa vesha',
    'Devi mudi': 'Strivesha/Punduvesha',
    'Kesaritatti': 'Rakshasa vesha',
    'Kirana': 'Strivesha',
    'Kolukireeta': 'Raja Vesha',
    'Kuttari': 'Hennu Banna',
    'Kombu': 'Rakshasa vesha',
    'Pakdi Shaiva vaishnava': 'Pundu Vesha',
    'Turai': 'Pundu Vesha'
}
vesha_mapping2 = {
    'Bheemamudi': 'Rakshasa vesha',
    'Devi mudi' : 'Devi/Vishnu',
    'Kesaritatti': 'Rakshasa vesha',
    'Kolukireeta': 'Raja Vesha',
    'Kuttari': 'Hennu Banna',
    'Kombu' : 'Mahisha',
    'Pakdi Shaiva vaishnava': 'Pundu Vesha',
    'Turai' : 'Vishnu/Pakdi'
}
vesha_mapping3 = {
    'Bheemamudi': 'Based on the Theme',
    'Devi mudi' : 'Chandrakruthi/Vishnu',
    'Kesaritatti': 'Based on the Theme',
    'Kolukireeta': 'Based on the Theme',
    'Kuttari': 'Based on the Theme',
    'Kombu' : 'Based on the Theme',
    'Pakdi Shaiva vaishnava': 'Based on the Theme',
    'Turai' : 'Vaishnava'
}

# Function to preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions using the trained model
def predict(image, model):
    # Preprocess the image
    img_width, img_height = model.input_shape[1:3]
    image = preprocess_image(image, (img_width, img_height))
    # Make predictions
    prediction = model.predict(image)
    return prediction

# Streamlit app
def main():
    # Set title and page layout
    st.set_page_config(page_title="Yakshagana Character Identification")
    st.title("Yakshagana Character Identification")
    st.write("Welcome to the Yakshagana Character Identification App")
    st.sidebar.title("Options")

    # Load the trained model
    model_path = st.sidebar.text_input("Model Path", "yakshagana.h5")
    if model_path:
        model = tf.keras.models.load_model(model_path)

    # File upload section
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image and predictions in columns
        col1, col2 = st.columns(2)

        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("Predict"):
                # Make predictions
                prediction = predict(image, model)

                # Get the predicted class name
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                predicted_vesha = vesha_mapping.get(predicted_class_name)
                predicted_vesha2 = vesha_mapping2.get(predicted_class_name)
                predicted_vesha3 = vesha_mapping3.get(predicted_class_name)

                # Display result
                st.header("Predictions:-")
                st.write(f"Makeup Pattern: {predicted_vesha3}")
                st.write(f"Predicted Crown: {predicted_class_name}")
                st.write(f"Predicted Character Type: {predicted_vesha}")
                st.write(f"Character: {predicted_vesha2}")

if __name__ == "__main__":
    main()
