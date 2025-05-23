from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
import os
import logging
import numpy as np
import gc
import tempfile
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from urllib.parse import unquote
from .models import DetectionHistory
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import requests
import json
from django.conf import settings
from datetime import datetime
import pytz
from .utils import render_to_pdf



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log TensorFlow versions immediately
logger.info(f"TensorFlow: {tf.__version__}, Keras: {tf.keras.__version__}")

# Define class names at module level - ensure this matches your training order exactly
class_names = ['Apple___Apple_scab',
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

# Model path
MODEL_PATH = os.path.abspath("/Users/sneharaut/Desktop/Plant_Disease_Dataset-2/plant_trained_model.keras")
logger.info(f"Model path set to: {MODEL_PATH}")
logger.info(f"Model file exists: {os.path.exists(MODEL_PATH)}")

# Debug flag
DEBUG = True

# Global model variable
model = None
treatment_df = pd.read_csv('/Users/sneharaut/Desktop/Plant_Disease_Dataset-2/treatments.csv', encoding='utf-8')
requirements_df = pd.read_csv('/Users/sneharaut/Desktop/Plant_Disease_Dataset-2/treatment/plant_requirements.csv', encoding='utf-8')
def preprocess_image(image_file):
    """
    Preprocess the image using the exact same approach as in Streamlit
    """
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            
            # If image is a file-like object from request.FILES
            if hasattr(image_file, 'read'):
                # Save uploaded file to temporary file
                with open(temp_path, 'wb') as f:
                    f.write(image_file.read())
                logger.info(f"Saved uploaded image to temporary file: {temp_path}")
            else:
                # If image is already a string path
                temp_path = image_file
        
        # Use tf.keras preprocessing just like in Streamlit
        image = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
        logger.info(f"Image loaded and resized to 128x128")
        
        # Convert to array and add batch dimension
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch
        
        logger.info(f"Final input shape: {input_arr.shape}")
        return input_arr, temp_path
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def load_model_on_demand():
    """Load the model on demand with proper initialization"""
    global model
    try:
        # Check if model is already loaded
        if model is not None:
            logger.info("Using previously loaded model")
            return model
            
        # Load model using tf.keras.models.load_model directly
        logger.info("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Warmup prediction
        dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = model.predict(dummy_input)
        logger.info("Model warmup complete")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@login_required
def predict(request):
    if request.method == 'POST':
        try:
            image = request.FILES['image']
            # Load the model
            model = load_model_on_demand()
            if model is None:
                return JsonResponse({"error": "Model could not be loaded"}, status=500)
                
            # Process the uploaded image using same method as Streamlit
            logger.info("Processing uploaded image file")
            input_array, temp_path = preprocess_image(image)
            if input_array is None:
                return JsonResponse({"error": "Image preprocessing failed"}, status=400)
            
            # Make prediction
            logger.info("Running prediction...")
            predictions = model.predict(input_array)
            
            # Log raw prediction data
            logger.info(f"Prediction shape: {predictions.shape}")
            
            # Log all predictions for debugging
            if DEBUG:
                logger.info("All class predictions:")
                for i, pred in enumerate(predictions[0]):
                    logger.info(f"{class_names[i]}: {pred:.6f}")
                
                # Log sorted prediction values to see distribution
                sorted_indices = np.argsort(predictions[0])[-10:][::-1]  # Top 10 indices
                logger.info("Top 10 predictions:")
                for idx in sorted_indices:
                    logger.info(f"Class {class_names[idx]}: {predictions[0][idx]:.6f}")
            
            # Get predicted class
            predicted_class_index = np.argmax(predictions[0])
            predicted_confidence = float(predictions[0][predicted_class_index])
            predicted_class_name = class_names[predicted_class_index]
            
            logger.info(f"Predicted class: {predicted_class_name} (index: {predicted_class_index})")
            logger.info(f"Confidence: {predicted_confidence:.6f}")
            
            # Get top 3 predictions for display
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = {class_names[i]: float(predictions[0][i]) for i in top_indices}
            

            treatment_info = treatment_df[treatment_df['Disease_Name'] == predicted_class_name]
            if not treatment_info.empty:
                treatment = treatment_info.iloc[0]['Treatment']  # Assuming 'treatment' column in CSV
            else:
                treatment = "Treatment information not available"
            
            
            # Clean up resources
            if os.path.exists(temp_path) and temp_path.startswith(tempfile.gettempdir()):
                os.unlink(temp_path)
                logger.info(f"Temporary file deleted: {temp_path}")
            
            # Clean memory
            gc.collect()
            
            # Save to history
            history_item = DetectionHistory.objects.create(
                user=request.user,
                image=image,
                prediction=predicted_class_name
            )
            
            # Return detection_id in the response
            return JsonResponse({
                'prediction': predicted_class_name,
                'confidence': predicted_confidence,
                'treatment': treatment,
                'top_predictions': top_predictions,
                'detection_id': history_item.id
            })
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({"error": f"Prediction error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def get_treatment(request, disease_name):
    """Retrieve treatment information for a disease"""
    try:
        # Decode URL-encoded disease name
        disease_name = unquote(disease_name)
        
        # Find the treatment for the disease
        treatment_info = treatment_df[treatment_df['Disease_Name'].str.lower() == disease_name.lower()]
        
        if not treatment_info.empty:
            treatment = treatment_info.iloc[0]['Treatment']  # Changed 'treatment' to 'Treatment'
            return JsonResponse({'treatment': treatment})
        else:
            return JsonResponse({'error': 'Treatment not found for this disease'}, status=404)
    
    except Exception as e:
        logger.error(f"Error in get_treatment: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'error': 'An error occurred while retrieving treatment information'}, status=500)
    
def get_requirements(request, plant_name):
    """Retrieve growing requirements for a plant"""
    try:
        # Decode URL-encoded plant name
        plant_name = unquote(plant_name)
        
        # Find the requirements for the plant
        req_info = requirements_df[requirements_df['Plant Name'].str.lower() == plant_name.lower()]
        
        if not req_info.empty:
            # Use consistent keys in the JSON response with standardized names
            requirements = {
                'optimal_temperature': req_info.iloc[0]['Optimal_temperature'],
                'sunlight_requirements': req_info.iloc[0]['Sunlight_requirement'],
                'watering_requirements': req_info.iloc[0]['Watering _requirement']  # Maintain the space as in CSV
            }
            return JsonResponse(requirements)
        else:
            return JsonResponse({'error': 'Requirements not found for this plant'}, status=404)
    
    except Exception as e:
        logger.error(f"Error in get_requirements: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'error': 'An error occurred while retrieving plant requirements'}, status=500)
def home(request):
    # Get weather data
    weather = get_weather_data()
    
    # Add debug print
    print(f"Weather data being sent to template: {weather}")
    
    return render(request, 'home.html', {'weather': weather})
def history(request):
    history_items = DetectionHistory.objects.filter(user=request.user)
    return render(request, 'history.html', {'history_items': history_items})

def test_model(request):
    """
    Test endpoint to verify model loading and prediction with a dummy input
    Access this via /test-model/ URL
    """
    try:
        # Load model
        model = load_model_on_demand()
        if model is None:
            return JsonResponse({"status": "error", "message": "Failed to load model"})
            
        # Create dummy input (all zeros)
        dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
        
        # Run prediction
        predictions = model.predict(dummy_input)
        
        # Get basic info
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        predicted_confidence = float(predictions[0][predicted_class_index])
        
        # Return basic model info and prediction result
        return JsonResponse({
            "status": "success",
            "model_info": {
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "total_layers": len(model.layers)
            },
            "dummy_prediction": {
                "class": predicted_class_name,
                "confidence": predicted_confidence,
            },
            "num_classes": len(class_names)
        })
        
    except Exception as e:
        logger.error(f"Error in test_model: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "status": "error", 
            "message": f"Test failed: {str(e)}",
            "traceback": traceback.format_exc() if DEBUG else "Set DEBUG=True for details"
        })

@login_required
def delete_history(request, history_id):
    history_item = get_object_or_404(DetectionHistory, id=history_id)
    
    # Security check - only allow users to delete their own history
    if history_item.user != request.user and not request.user.userprofile.is_admin:
        messages.error(request, "You don't have permission to delete this item.")
        return redirect('history')
    
    # Delete the image file if it exists
    if history_item.image:
        if os.path.isfile(history_item.image.path):
            os.remove(history_item.image.path)
    
    # Delete the item
    history_item.delete()
    
    # Show a success message
    messages.success(request, "History item deleted successfully!")
    
    # Check if there's a next parameter for redirect
    next_url = request.GET.get('next')
    if next_url:
        return redirect(next_url)
    
    # Otherwise redirect back to history page
    return redirect('history')

def get_weather_data(city=None):
    """
    Get real-time weather data from OpenWeatherMap API with Nepal timestamp
    """
    if not city:
        city = 'Kathmandu'  # Default to Nepal's capital
        
    # Get Nepal time regardless of API success
    nepal_tz = pytz.timezone('Asia/Kathmandu')
    nepal_time = datetime.now(nepal_tz)
    nepal_time_str = nepal_time.strftime('%b %d, %Y %I:%M %p')
    
    # Hardcoded API key (for testing only - move to settings.py in production)
    api_key = settings.WEATHER_API_KEY  # Example: "a1b2c3d4e5f6g7h8i9j0"
    
    try:
        # Make the API request
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
        response = requests.get(url, timeout=5)
        
        # For debugging
        print(f"Weather API response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # For debugging
            print(f"Weather data received: {data}")
            
            weather = {
                'city': data['name'],
                'temperature': int(round(data['main']['temp'])),  # Convert to integer
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'humidity': int(data['main']['humidity']),  # Convert to integer
                'wind_speed': round(data['wind']['speed'], 1),  # Round to 1 decimal
                'feels_like': int(round(data['main']['feels_like'])),  # Convert to integer
                'nepal_time': nepal_time_str
            }
            return weather
        else:
            print(f"Weather API error: {response.status_code}, {response.text}")
            return get_default_weather(nepal_time_str)
            
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return get_default_weather(nepal_time_str)

def get_default_weather(nepal_time_str):
    """Return default weather data with actual values if API fails"""
    return {
        'city': 'Kathmandu',
        'temperature': 25,  # Numeric value, not string
        'description': 'Weather information unavailable',
        'icon': '01d',
        'humidity': 60,  # Numeric value, not string
        'wind_speed': 5.0,  # Numeric value, not string
        'feels_like': 27,  # Numeric value, not string
        'nepal_time': nepal_time_str
    }

@login_required
def generate_report(request, detection_id):
    # Get the detection record
    detection = get_object_or_404(DetectionHistory, id=detection_id, user=request.user)
    
    # Get treatment data (similar to your existing code)
    prediction = detection.prediction
    plant_name = prediction.split('___')[0] if '___' in prediction else prediction
    
    # Get treatment data
    treatment = "No treatment information available."
    try:
        # Use your existing treatment logic here
        # Example: treatment = get_treatment_for_disease(prediction)
        pass
    except Exception as e:
        print(f"Error getting treatment: {e}")
    
    # Get growing requirements data
    requirements = {
        'optimal_temperature': 'Not available',
        'sunlight_requirements': 'Not available',
        'watering_requirements': 'Not available'
    }
    
    try:
        # Use your existing growing requirements logic here
        # Example: requirements = get_requirements_for_plant(plant_name)
        pass
    except Exception as e:
        print(f"Error getting requirements: {e}")
    
    # Prepare context for PDF
    context = {
        'detection': detection,
        'prediction': prediction,
        'treatment': treatment,
        'requirements': requirements,
        'date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'user': request.user,
        'plant_name': plant_name
    }
    
    # Generate PDF
    pdf = render_to_pdf('pdf_report.html', context)
    
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"Plant_Disease_Report_{detection.id}.pdf"
        content = f"attachment; filename={filename}"
        response['Content-Disposition'] = content
        return response
    
    return HttpResponse("Error generating PDF", status=400)
