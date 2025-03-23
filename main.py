import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Initialize session state variables if they don't exist
if 'result_index' not in st.session_state:
    st.session_state.result_index = None
if 'disease_name' not in st.session_state:
    st.session_state.disease_name = None

@st.cache_data
def load_treatment_data():
    return pd.read_csv('/Users/sneharaut/Desktop/Plant_Disease_Dataset-2/treatments.csv', encoding='utf-8')


treatment_df = load_treatment_data()


def get_treatment(Disease_Name):
    treatment = treatment_df[treatment_df['Disease_Name'] == Disease_Name]['Treatment']
    return treatment.values[0] if not treatment.empty else "No treatment details available."
    
#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('/Users/sneharaut/Desktop/Plant_Disease_Dataset-2/plant_trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Define class names
class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy',
 'not_a_leaf']

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!


""")

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
""")
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    # Display current prediction if available
    if st.session_state.disease_name is not None:
        st.success(f"Current prediction: {st.session_state.disease_name}")
    
    #Predict Button
    if(st.button("Predict")):
        if test_image is not None:
            with st.spinner("Please Wait.."):
                st.write("Our Prediction")
                st.session_state.result_index = model_prediction(test_image)
                st.session_state.disease_name = class_name[st.session_state.result_index]
                st.success(f"Model is Predicting it's a {st.session_state.disease_name}")
        else:
            st.warning("Please upload an image first")
        
    if(st.button("Show Image")):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first")
            
    if st.button("Show Treatment"):
        if st.session_state.disease_name is not None:
            treatment = get_treatment(st.session_state.disease_name)
            st.info(f"**Recommended Treatment:** {treatment}")
        else:
            st.warning("Please predict the disease first")