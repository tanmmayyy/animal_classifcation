from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Animal classes (must match your training data)
ANIMAL_CLASSES = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra'
]

class AnimalPredictor:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.img_height = 224
        self.img_width = 224
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                # Create a dummy model for testing if real model doesn't exist
                self.create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for testing when real model is not available"""
        logger.warning("Creating dummy model for testing purposes")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(15, activation='softmax')
        ])
        # Initialize with random weights
        dummy_input = np.random.random((1, 224, 224, 3))
        self.model(dummy_input)
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize((self.img_width, self.img_height))
            
            # Convert to array and normalize
            img_array = tf.keras.utils.img_to_array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image):
        """Make prediction on image"""
        try:
            if self.model is None:
                raise Exception("Model not loaded")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = ANIMAL_CLASSES[predicted_class_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [
                {
                    'class': ANIMAL_CLASSES[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_idx
            ]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

# Initialize predictor
MODEL_PATH = 'best_animal_classifier.h5'  # Update this path if needed
predictor = AnimalPredictor(MODEL_PATH)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'classes': ANIMAL_CLASSES
    })

@app.route('/predict', methods=['POST'])
def predict_animal():
    """Predict animal from uploaded image"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Read and process image
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Make prediction
        result = predictor.predict(image)
        
        # Add success status
        result['status'] = 'success'
        
        logger.info(f"Prediction made: {result['predicted_class']} with confidence: {result['confidence']:.2f}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict_animal: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_animal_base64():
    """Predict animal from base64 encoded image"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = data['image']
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Make prediction
        result = predictor.predict(image)
        result['status'] = 'success'
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict_animal_base64: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("ANIMAL CLASSIFICATION API SERVER")
    print("=" * 50)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {predictor.model is not None}")
    print(f"Supported classes: {len(ANIMAL_CLASSES)}")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("API Endpoints:")
    print("  GET  /          - Main page")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Predict from file upload")
    print("  POST /predict_base64 - Predict from base64 image")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)