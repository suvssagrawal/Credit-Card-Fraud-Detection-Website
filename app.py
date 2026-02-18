from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the model package once when server starts
print("Loading model package...")
package = pickle.load(open('model/fraud_detection_complete.pkl', 'rb'))
model = package['model']
scaler = package['scaler']
le_transaction = package['le_transaction']
le_merchant = package['le_merchant']
le_country = package['le_country']
features = package['features']
print("✅ Model loaded successfully!")

def predict_fraud(amount, transaction_type, merchant_category, country, hour):
    """
    Predict if a transaction is fraudulent
    """
    try:
        # Step 1: Encode categorical features
        trans_enc = le_transaction.transform([transaction_type])[0]
        merch_enc = le_merchant.transform([merchant_category])[0]
        country_enc = le_country.transform([country])[0]
        
        # Step 2: Create additional features
        is_night = 1 if (hour <= 5 or hour >= 22) else 0
        is_high = 1 if amount > 1000 else 0
        
        # Step 3: Create input DataFrame
        input_data = pd.DataFrame([[
            amount,
            hour,
            is_night,
            is_high,
            trans_enc,
            merch_enc,
            country_enc
        ]], columns=features)
        
        # Step 4: Scale features
        input_data[['amount', 'hour']] = scaler.transform(
            input_data[['amount', 'hour']]
        )
        
        # Step 5: Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            risk_color = "#e74c3c"  # Red
        elif probability >= 0.4:
            risk_level = "MEDIUM"
            risk_color = "#f39c12"  # Orange
        else:
            risk_level = "LOW"
            risk_color = "#2ecc71"  # Green
        
        return {
            'success': True,
            'is_fraud': int(prediction),
            'fraud_probability': round(float(probability) * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'message': 'Prediction completed successfully'
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract fields
        amount = float(data.get('amount'))
        transaction_type = data.get('transaction_type')
        merchant_category = data.get('merchant_category')
        country = data.get('country')
        hour = int(data.get('hour'))
        
        # Validate inputs
        if not all([amount, transaction_type, merchant_category, country, hour is not None]):
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400
        
        if amount <= 0:
            return jsonify({
                'success': False,
                'message': 'Amount must be greater than 0'
            }), 400
        
        if hour < 0 or hour > 23:
            return jsonify({
                'success': False,
                'message': 'Hour must be between 0 and 23'
            }), 400
        
        # Make prediction
        result = predict_fraud(amount, transaction_type, merchant_category, country, hour)
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'message': 'Invalid input format'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)