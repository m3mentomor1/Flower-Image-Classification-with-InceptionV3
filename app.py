import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# Class mapping for flower classification
class_mapping_flower = {
    0: 'Astilbe',
    1: 'Black-Eyed Susan',
    2: 'Bellflower',
    3: 'Common Daisy',
    4: 'Coreopsis',
    5: 'Dandelion',
    6: 'Water Lily',
    7: 'Carnation',
    8: 'Calendula',
    9: 'California Poppy',
    10: 'Sunflower',
    11: 'Tulip',
    12: 'Rose',
    13: 'Iris',
}

# Function to load the flower classification model
@st.cache(allow_output_mutation=True)
def load_flower_model():
    # Google Drive direct link to the shared model file
    drive_url = 'https://drive.google.com/drive/folders/1aTpSSn11zzGbMZMWbixy1tKJJng6eU0P?usp=sharing'
    
    # Local path to save the downloaded model file
    local_model_path = './model.ckpt'  # You can adjust the path as needed
    
    # Download the model file using gdown
    response = gdown.download(drive_url, output=local_model_path, quiet=False)

    # Load the InceptionV3 base model
    input_shape = (224, 224, 3)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Create a sequential model
    model = tf.keras.models.Sequential()

    # Add the InceptionV3 base model to the sequential model
    model.add(base_model)

    # Add a global average pooling layer to reduce the spatial dimensions of the output
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Add a dense layer with 256 units and ReLU activation function
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    # Add a dropout layer to prevent overfitting
    model.add(tf.keras.layers.Dropout(0.5))

    # Add the final dense layer with the number of labels and softmax activation function
    total_labels = len(class_mapping_flower)
    model.add(tf.keras.layers.Dense(total_labels, activation='softmax'))

    # Load the weights of the trained model
    model.load_weights(local_model_path)

    return model

# Function to preprocess and make predictions for flower classification
def predict_flower(image, model):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize images to the specified target size: 224x224
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label from the mapping
    predicted_class_label = class_mapping_flower.get(predicted_class_index, 'Unknown')

    return predicted_class_label

# Streamlit app
st.title('Flower Image Classification')
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load the flower classification model
    flower_model = load_flower_model()

    # Make predictions for flower classification
    predicted_class_flower = predict_flower(image, flower_model)
    st.write(f"Prediction: {predicted_class_flower}")
