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
from django.utils.translation import gettext as _
import pytz
from .utils import render_to_pdf
from django.template.loader import render_to_string
from weasyprint import HTML  # Ensure you have WeasyPrint installed

WEATHER_TRANSLATIONS = {
    "clear sky": "खुला आकाश",
    "few clouds": "थोरै बादल",
    "scattered clouds": "छरिएका बादल",
    "broken clouds": "फुटेका बादल",
    "shower rain": "झरी",
    "rain": "वर्षा",
    "thunderstorm": "गडगडाहटसहितको वर्षा",
    "snow": "हिउँ",
    "mist": "कुहिरो"
}

SUNLIGHT_TRANSLATIONS = {
    "Full": "पूरा",
    "Partial": "आंशिक",
    "Low": "कम"
}

WATERING_TRANSLATIONS = {
    "Medium": "मध्यम",
    "High": "धेरै",
    "Low": "कम"
}

def translate_range(value, to_word_nepali="देखि"):
    if isinstance(value, str):
        return value.replace("to", to_word_nepali)
    return value

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

# After normalization, all columns are lowercase and spaces are preserved unless you also replace spaces with underscores.
# If you want to be extra robust, you can also replace spaces with underscores:
requirements_df.columns = requirements_df.columns.str.strip().str.lower().str.replace(' ', '_')

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
            
            # Get top 3 predictions for display
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = {class_names[i]: float(predictions[0][i]) for i in top_indices}
            
            # Get treatment info
            treatment_info = treatment_df[treatment_df['Disease_Name'] == predicted_class_name]
            if not treatment_info.empty:
                treatment = treatment_info.iloc[0]['Treatment']
            else:
                treatment = _("Treatment information not available")
            
            # Clean up resources
            if os.path.exists(temp_path) and temp_path.startswith(tempfile.gettempdir()):
                os.unlink(temp_path)
                logger.info(f"Temporary file deleted: {temp_path}")
            gc.collect()
            
            
            # Save to history
            history_item = DetectionHistory.objects.create(
                user=request.user,
                image=image,
                prediction=predicted_class_name
            )
            
            # Return detection_id in the response
            return JsonResponse({
                
           'prediction': predicted_class_name,  # English, for backend use
           'prediction_translated': _(predicted_class_name),  # Nepali, for display

                'confidence': predicted_confidence,
                'treatment': _(treatment),              
                'top_predictions': {_(k): v for k, v in top_predictions.items()},  
                'detection_id': history_item.id
            })
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({"error": _("Prediction error: ") + str(e)}, status=500)
    
    return JsonResponse({"error": _("Invalid request")}, status=400)

def get_treatment(request, disease_name):
    """Retrieve treatment information for a disease"""
    try:
        # Decode URL-encoded disease name
        disease_name = unquote(disease_name)
        
        # Find the treatment for the disease
        treatment_info = treatment_df[treatment_df['Disease_Name'].str.lower() == disease_name.lower()]
        
        if not treatment_info.empty:
            treatment = treatment_info.iloc[0]['Treatment']  # Changed 'treatment' to 'Treatment'
            return JsonResponse({'treatment': _(treatment)})
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

        # Find the actual plant name column in a case-insensitive and space-insensitive way in requirements_df.
        # Use requirements_df.columns directly as they are globally normalized to lowercase with underscores.
        plant_column = None
        for col in requirements_df.columns:
            # Check if 'plant' and 'name' are in the lowercase column name and if it's the normalized form.
            if 'plant' in col.lower() and 'name' in col.lower() and 'plant_name' in col.lower():
                 plant_column = col
                 break # Found the likely column, break the loop

        if not plant_column:
            logger.error("Cannot find a suitable 'plant name' column in requirements_df.")
            logger.error(f"Available columns in requirements_df: {requirements_df.columns.tolist()}")
            return JsonResponse({'error': 'Internal configuration error: Plant name column not found'}, status=500)

        # Filter requirements_df by plant name (case-insensitive) using the identified column.
        # Ensure we are definitely using requirements_df here and not accidentally treatment_df.
        req_info = requirements_df[requirements_df[plant_column].str.lower() == plant_name.lower()]

        if not req_info.empty:
            # Use consistent lowercase keys for the response, matching the global normalization of requirements_df columns.
            # Correcting the .iloc[0].iloc[0] mistake from the previous edit.
            requirements = {
                'optimal_temperature': req_info.iloc[0].get('optimal_temperature', 'Not available'),
                'sunlight_requirements': req_info.iloc[0].get('sunlight_requirement', 'Not available'),
                'watering_requirements': req_info.iloc[0].get('watering__requirement', 'Not available'),
                'humidity': req_info.iloc[0].get('humidity', 'Not available')
            }
            return JsonResponse(requirements);
        else:
            # Log that requirements were not found for the specific plant name.
            logger.info(f"Requirements not found for plant: {plant_name}")
            # Also log the plant names available in the dataframe for debugging
            if plant_column:
                logger.info(f"Available plant names in requirements_df: {requirements_df[plant_column].unique().tolist()}")
            return JsonResponse({'error': 'Requirements not found for this plant'}, status=404);

    except Exception as e:
        logger.error(f"Error in get_requirements: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'error': 'An error occurred while retrieving plant requirements'}, status=500);

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
def generate_report(request, id):
    # Get the detection record
    detection = get_object_or_404(DetectionHistory, id=id, user=request.user)
    
    # Get prediction and plant name
    prediction = detection.prediction
    plant_name = prediction.split('___')[0] if '___' in prediction else prediction
    
    # Get treatment data
    treatment = "No treatment information available."
    try:
        # Don't use the view function directly - extract the treatment info
        # Find the treatment for the disease from your dataframe
        treatment_info = treatment_df[treatment_df['Disease_Name'].str.lower() == prediction.lower()]
        if not treatment_info.empty:
            treatment = treatment_info.iloc[0]['Treatment']
    except Exception as e:
        print(f"Error getting treatment: {e}")
    
    # Get growing requirements data
    requirements = {
        'optimal_temperature': 'Not available',
        'sunlight_requirements': 'Not available',
        'watering_requirements': 'Not available',
        'humidity': 'Not available'
    }
    try:
        # Always compare lowercased values
        req_info = requirements_df[requirements_df['plant_name'].str.lower() == plant_name.lower()]
        print("Looking for plant_name:", plant_name)
        print("Available plant names:", requirements_df['plant_name'].unique())
        print("Filtered row:", req_info)
        if not req_info.empty:
            requirements = {
                'optimal_temperature': req_info.iloc[0].get('optimal_temperature', 'Not available'),
                'sunlight_requirements': req_info.iloc[0].get('sunlight_requirement', 'Not available'),
                'watering_requirements': req_info.iloc[0].get('watering__requirement', 'Not available'),
                'humidity': req_info.iloc[0].get('humidity', 'Not available')
            }
    except Exception as e:
        print(f"Error getting requirements: {e}")
    
    # Prepare context for PDF
    context = {
        'detection': detection,
        'prediction': prediction,
        'treatment': treatment,
        'requirements': requirements,
        'date': datetime.now().strftime("%Y-%m-%d"),
        'user': request.user,
        'plant_name': plant_name
    }
    
    # Render the PDF
    html_string = render_to_string('pdf_report.html', context)
    pdf = HTML(string=html_string).write_pdf()  # Generate PDF from HTML
    
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = f"Plant_Disease_Report_{id}.pdf"
        content = f"attachment; filename={filename}"
        response['Content-Disposition'] = content
        return response
    
    return HttpResponse("Error generating PDF", status=400)
# Add this to your views.py file

def get_yield_prediction(request):
    """Get yield prediction based on plant requirements and current weather including temperature and humidity"""
    try:
        # Get parameters from request
        if request.method == 'POST':
            data = json.loads(request.body)
            plant_name = data.get('plant_name')
            location = data.get('location', 'Kathmandu')  # Default to Kathmandu if not specified
            disease_detected = data.get('disease_detected', 'healthy')
        else:
            plant_name = request.GET.get('plant_name')
            location = request.GET.get('location', 'Kathmandu')
            disease_detected = request.GET.get('disease_detected', 'healthy')
            
        if not plant_name:
            return JsonResponse({'error': 'Plant name is required'}, status=400)
            
        # Get real-time weather data for the location
        weather_data = get_weather_data(location)
        weather_desc_en = weather_data.get('description', '')  # e.g., "broken clouds"
        
        # Debugging - log the request data and column names
        logger.info(f"Yield prediction request for plant: {plant_name}, location: {location}")
        logger.info(f"Available columns in requirements_df: {requirements_df.columns.tolist()}")
        
        # Normalize column names to handle potential case sensitivity issues
        normalized_df = requirements_df.copy()
        normalized_df.columns = normalized_df.columns.str.strip().str.lower()
        
        # Look for plant name in a case-insensitive way
        plant_column = None
        for col in normalized_df.columns:
            if 'plant' in col and 'name' in col:
                plant_column = col
                break
                
        if not plant_column:
            logger.error("Cannot find 'plant name' column in requirements_df")
            return JsonResponse({'error': 'Internal configuration error: Plant name column not found'}, status=500)
            
        # Find the plant using the identified column
        plant_req = normalized_df[normalized_df[plant_column].str.lower() == plant_name.lower()]
        
        if plant_req.empty:
            # Log available plant names for debugging
            available_plants = normalized_df[plant_column].unique().tolist()
            logger.info(f"Available plants: {available_plants}")
            return JsonResponse({'error': f'Requirements not found for plant: {plant_name}'}, status=404)
            
        optimal_temp_col = next((col for col in normalized_df.columns if 'optimal' in col and 'temp' in col), None)
        sunlight_col = next((col for col in normalized_df.columns if 'sun' in col), None)
        watering_col = next((col for col in normalized_df.columns if 'water' in col), None)
        humidity_col = next((col for col in normalized_df.columns if 'humid' in col), None)
        
        # Get values with safe fallbacks
        optimal_temp = plant_req.iloc[0][optimal_temp_col] if optimal_temp_col else "20 to 30°C"
        sunlight_req = plant_req.iloc[0][sunlight_col] if sunlight_col else "Full sun"
        watering_req = plant_req.iloc[0][watering_col] if watering_col else "Regular watering"
        optimal_humidity = plant_req.iloc[0][humidity_col] if humidity_col else "50% to 70%"
        
        # Parse optimal temperature range
        temp_range = str(optimal_temp).replace('°C', '').replace('˚C', '').strip()
        try:
            if 'to' in temp_range:
                parts = temp_range.split('to')
                min_temp_str = ''.join(c for c in parts[0] if c.isdigit() or c == '.')
                max_temp_str = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                min_temp = float(min_temp_str) if min_temp_str else 20.0
                max_temp = float(max_temp_str) if max_temp_str else 30.0
            elif '-' in temp_range:
                min_temp, max_temp = map(float, temp_range.split('-'))
            else:
                # Handle single temperature value
                temp_str = ''.join(c for c in temp_range if c.isdigit() or c == '.')
                min_temp = max_temp = float(temp_str) if temp_str else 25.0
        except Exception as e:
            logger.warning(f"Error parsing temperature: {e}, using default values")
            min_temp, max_temp = 20.0, 30.0
            
        # Parse optimal humidity range with error handling
        humidity_range = str(optimal_humidity).strip()
        try:
            if '%' in humidity_range and 'to' in humidity_range:
                parts = humidity_range.split('to')
                min_humidity_str = parts[0].replace('%', '').strip()
                max_humidity_str = parts[1].replace('%', '').strip()
                min_humidity = float(min_humidity_str) if min_humidity_str.replace('.', '', 1).isdigit() else 50.0
                max_humidity = float(max_humidity_str) if max_humidity_str.replace('.', '', 1).isdigit() else 70.0
            elif 'to' in humidity_range:
                parts = humidity_range.split('to')
                min_humidity_str = ''.join(c for c in parts[0] if c.isdigit() or c == '.')
                max_humidity_str = ''.join(c for c in parts[1] if c.isdigit() or c == '.')
                min_humidity = float(min_humidity_str) if min_humidity_str else 50.0
                max_humidity = float(max_humidity_str) if max_humidity_str else 70.0
            elif '-' in humidity_range:
                parts = humidity_range.replace('%', '').split('-')
                min_humidity = float(parts[0].strip())
                max_humidity = float(parts[1].strip())
            else:
                # Handle single humidity value
                humidity_str = ''.join(c for c in humidity_range if c.isdigit() or c == '.')
                min_humidity = max_humidity = float(humidity_str) if humidity_str else 60.0
        except Exception as e:
            logger.warning(f"Error parsing humidity: {e}, using default values")
            min_humidity, max_humidity = 50.0, 70.0
            
        # Current temperature and humidity (with safety checks)
        current_temp = weather_data.get('temperature', 25)
        current_humidity = weather_data.get('humidity', 60)
        
        # Calculate temperature score (0-1)
        if min_temp <= current_temp <= max_temp:
            temp_score = 1.0
        else:
            # The further from optimal range, the lower the score
            closest_optimal = min_temp if abs(current_temp - min_temp) < abs(current_temp - max_temp) else max_temp
            temp_difference = abs(current_temp - closest_optimal)
            temp_score = max(0, 1 - (temp_difference / 10))  # Decrease by 0.1 for each degree away
            
        # Calculate humidity score (0-1)
        if min_humidity <= current_humidity <= max_humidity:
            humidity_score = 1.0
        else:
            # The further from optimal range, the lower the score
            closest_optimal = min_humidity if abs(current_humidity - min_humidity) < abs(current_humidity - max_humidity) else max_humidity
            humidity_difference = abs(current_humidity - closest_optimal)
            humidity_score = max(0, 1 - (humidity_difference / 20))  # Decrease by 0.05 for each percentage point away
            
        # Disease impact (if disease detected, yield will be reduced)
        disease_factor = 0.6 if disease_detected.lower() != 'healthy' and '___healthy' not in disease_detected.lower() else 1.0
            
        # Calculate yield score (0-100%)
        # Consider both temperature and humidity with equal weights (50% each)
        environmental_score = (temp_score * 0.5) + (humidity_score * 0.5)
        yield_score = environmental_score * disease_factor * 100
        
        # Yield prediction levels
        if yield_score >= 80:
            yield_level = "Excellent"
            yield_description = "Expected to produce maximum yield"
        elif yield_score >= 60:
            yield_level = "Good"
            yield_description = "Expected to produce good yield"
        elif yield_score >= 40:
            yield_level = "Average"
            yield_description = "Expected to produce average yield"
        elif yield_score >= 20:
            yield_level = "Below Average"
            yield_description = "Expected to produce below average yield"
        else:
            yield_level = "Poor"
            yield_description = "Expected to produce poor yield"
            
        # Recommendations based on conditions
        recommendations = []
        
        if current_temp < min_temp:
            recommendations.append(f"Temperature is too low ({current_temp}°C). Consider using greenhouse or temperature control methods to raise temperature to {min_temp}-{max_temp}°C.")
        elif current_temp > max_temp:
            recommendations.append(f"Temperature is too high ({current_temp}°C). Consider shade or cooling methods to lower temperature to {min_temp}-{max_temp}°C.")
            
        if current_humidity < min_humidity:
            recommendations.append(f"Humidity is too low ({current_humidity}%). Consider using humidity-increasing methods like misting to achieve {min_humidity}-{max_humidity}%.")
        elif current_humidity > max_humidity:
            recommendations.append(f"Humidity is too high ({current_humidity}%). Consider improving ventilation or using dehumidifiers to lower humidity to {min_humidity}-{max_humidity}%.")
            
        if disease_detected.lower() != 'healthy' and '___healthy' not in disease_detected.lower():
            recommendations.append(f"Treat the detected disease ({disease_detected}) promptly to improve yield.")
        
        response_data = {
            'plant_name': plant_name,
            'location': location,
            'current_weather': {
                'temperature': current_temp,
                'humidity': current_humidity,
                'description': weather_desc_en
            },
            'plant_requirements': {
                'optimal_temperature': optimal_temp,
                'optimal_humidity': optimal_humidity,
                'sunlight_requirements': sunlight_req,
                'watering_requirements': watering_req
            },
            'yield_prediction': {
                'score': round(yield_score, 1),
                'temperature_score': round(temp_score * 100, 1),
                'humidity_score': round(humidity_score * 100, 1),
                'level': yield_level,
                'description': yield_description
            },
            'recommendations': recommendations
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error in yield prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)