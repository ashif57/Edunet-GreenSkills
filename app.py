from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = 'Farm_Irrigation_System.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Error: index.html file not found!", 404

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Machine learning model not available. Please ensure Farm_Irrigation_System.pkl exists.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'sensor_values' not in data:
            return jsonify({
                'error': 'Invalid request. Please provide sensor_values array.'
            }), 400
        
        sensor_values = data['sensor_values']
        
        # Validate sensor values
        if not isinstance(sensor_values, list) or len(sensor_values) != 20:
            return jsonify({
                'error': 'Invalid sensor_values. Expected array of 20 numeric values.'
            }), 400
        
        # Convert to numpy array and validate range
        try:
            sensor_array = np.array(sensor_values, dtype=float)
            
            # Check if all values are in valid range [0, 1]
            if not all(0 <= val <= 1 for val in sensor_array):
                return jsonify({
                    'error': 'All sensor values must be between 0.0 and 1.0'
                }), 400
                
        except (ValueError, TypeError):
            return jsonify({
                'error': 'All sensor values must be numeric.'
            }), 400
        
        # Reshape for prediction (model expects 2D array)
        sensor_array = sensor_array.reshape(1, -1)
        
        # Make prediction
        try:
            prediction = model.predict(sensor_array)
            
            # Convert prediction to readable format
            # The model returns an array with 3 values (for 3 sprinklers/parcels)
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                if isinstance(prediction[0], (list, np.ndarray)) and len(prediction[0]) >= 3:
                    # Multi-output prediction
                    pred_values = prediction[0]
                elif len(prediction) >= 3:
                    # Single array with 3 values
                    pred_values = prediction
                else:
                    # Single prediction, replicate for all sprinklers
                    pred_values = [prediction[0]] * 3
            else:
                # Single value prediction
                pred_values = [prediction] * 3
            
            # Convert to ON/OFF based on threshold (assuming binary classification or regression)
            predictions = {}
            parcel_names = ['parcel_0', 'parcel_1', 'parcel_2']
            for i in range(3):
                # If the prediction is already binary (0/1), use it directly
                # If it's a probability, use 0.5 as threshold
                if isinstance(pred_values[i], (int, np.integer)):
                    status = "ON" if pred_values[i] == 1 else "OFF"
                else:
                    status = "ON" if float(pred_values[i]) > 0.5 else "OFF"
                
                predictions[parcel_names[i]] = status
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'sensor_values': sensor_values
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Prediction failed: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Smart Irrigation System Flask Server...")
    print(f"Model status: {'Loaded' if model is not None else 'Not loaded'}")
    print("Server will be available at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/predict")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)