# 💳 Credit Card Fraud Detection System

A comprehensive machine learning solution for detecting fraudulent credit card transactions with a user-friendly web interface for real-time fraud prediction.

**Author:** Suvan Agrawal

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Data Analysis](#data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Overview

Credit card fraud represents a significant threat to the integrity of financial transactions and consumer trust in digital commerce. As the reliance on credit cards for everyday purchases continues to grow, so does the sophistication of fraudsters exploiting vulnerabilities in the system.

This project aims to:
- Analyze patterns of credit card fraud
- Understand factors contributing to fraudulent activities
- Build effective machine learning models for detection
- Deploy a real-time fraud detection web application

---

## 📊 Dataset Description

The dataset comprises **100,000 simulated credit card transactions** designed to mirror real-world activity patterns.

### Features

| Feature | Description |
|---------|-------------|
| **TransactionID** | Unique identifier for each transaction |
| **TransactionDate** | Date and time of transaction |
| **Amount** | Monetary value ($29 - $5,000) |
| **MerchantID** | Merchant identifier (1-999) |
| **TransactionType** | Purchase or Refund |
| **Location** | Geographic location (10 US cities) |
| **IsFraud** | Target variable (0 = Genuine, 1 = Fraud) |

### Class Distribution

- **Total Transactions:** 100,000
- **Genuine Transactions:** 99,000 (99.0%)
- **Fraudulent Transactions:** 1,000 (1.0%)
- **Imbalance Ratio:** 99:1

> **Note:** The 1% fraud rate aligns with real-world credit card fraud statistics (typically 0.5-2%), making this dataset realistic for training production-ready models.

---

## ✨ Features

### User-Friendly Input Fields
- Transaction Amount
- Transaction Type (Purchase/Refund)
- Merchant Category
- Location (City dropdown)
- Date & Time (auto-filled)

### Model Predictions
- Real-time fraud risk assessment
- Probability score (0-100%)
- Risk level classification (Low/Medium/High)
- Actionable recommendations

---

## 🔍 Data Analysis

### Exploratory Data Analysis (EDA)

1. **Transaction Distribution Analysis**
   - Class imbalance visualization
   - Amount distribution by fraud status
   
2. **Temporal Analysis**
   - Transaction patterns by hour
   - Weekday vs weekend trends
   
3. **Geographic Analysis**
   - Fraud rates by location
   - High-risk cities identification

4. **Merchant Analysis**
   - Fraud rates per merchant
   - Risk categorization

### Key Findings

✅ **No missing values** - Clean dataset ready for modeling  
✅ **Balanced fraud ratio** - 1% fraud rate enables effective training  
✅ **Diverse locations** - 10 major US cities represented  
✅ **Realistic patterns** - Transaction amounts and types mirror real-world data

---

## 🛠️ Feature Engineering

### Enhanced Features Created

1. **Time-Based Features**
```python
   - Hour (0-23)
   - IsNightTime (22:00-06:00 flag)
   - DayOfWeek (0=Monday, 6=Sunday)
   - IsWeekend (Saturday/Sunday flag)
```

2. **Transaction-Based Features**
```python
   - HighAmount (>$3000 flag)
   - IsRefund (Refund transaction flag)
```

3. **Merchant-Based Features**
```python
   - MerchantRisk (Fraud rate per merchant)
```

### Feature Importance

These engineered features significantly improve model performance by:
- Capturing temporal fraud patterns (night transactions are riskier)
- Identifying high-value transaction anomalies
- Quantifying merchant-specific risk profiles

---

## 🤖 Model Development

### Algorithms Implemented

1. **Logistic Regression**
   - Baseline model for interpretability
   
2. **Decision Tree**
   - Non-linear pattern detection
   
3. **Random Forest** ⭐
   - Best performing model
   - Handles categorical features well
   - Robust against overfitting

4. **Support Vector Machine (SVM)**
   - Advanced classification

### Handling Class Imbalance

**Technique:** SMOTE (Synthetic Minority Over-sampling Technique)
- Applied only to training data (prevents data leakage)
- Balances classes from 99:1 to 1:1 ratio
- Generates synthetic fraud samples for better learning

### Evaluation Metrics

**Primary Focus:**
- **Recall:** Maximize fraud detection (minimize missed frauds)
- **Precision:** Reduce false alarms
- **F1-Score:** Balance between recall and precision
- **ROC-AUC:** Overall discrimination ability

**Expected Performance:**
- Recall: 85-92% (catch most fraudulent transactions)
- Precision: 70-85% (reasonable false alarm rate)
- F1-Score: 80-88% (strong balanced performance)

---

## 💻 Technologies Used

### Machine Learning
- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - ML algorithms and preprocessing
- **imbalanced-learn** - SMOTE implementation
- **matplotlib & seaborn** - Data visualization

### Web Development
- **Flask** - Backend framework
- **HTML/CSS** - Frontend design
- **JavaScript** - Interactive user experience
- **Bootstrap** - Responsive UI components

### Model Deployment
- **pickle** - Model serialization
- **REST API** - Prediction endpoint

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the application**
```bash
   python app.py
```

4. **Access the web interface**
```
   http://localhost:5000
```

---

## 📱 Usage

### Training the Model
```python
# Load and preprocess data
df = pd.read_csv('credit_card_fraud_dataset.csv')

# Apply feature engineering
# Train model with SMOTE
# Save model using pickle
```

### Making Predictions
```python
# Load saved model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare transaction data
transaction = {
    'Amount': 4500,
    'TransactionType': 'purchase',
    'Location': 'New York',
    'Hour': 23,
    'IsNightTime': 1,
    # ... other features
}

# Get prediction
prediction = model.predict([transaction])
probability = model.predict_proba([transaction])[:, 1][0]

print(f"Fraud Probability: {probability*100:.2f}%")
```

### Web Interface

1. Enter transaction details in the form
2. Click "Check Transaction"
3. View fraud risk assessment
4. Follow recommended actions

---

## 📈 Results

### Model Performance (To Be Updated)

| Model | Recall | Precision | F1-Score | Status |
|-------|--------|-----------|----------|--------|
| Logistic Regression | TBD | TBD | TBD | ⏳ In Progress |
| Decision Tree | TBD | TBD | TBD | ⏳ In Progress |
| Random Forest | TBD | TBD | TBD | ⏳ In Progress |
| SVM | TBD | TBD | TBD | ⏳ Planned |

---

## 🔮 Future Enhancements

- [ ] Implement additional sampling techniques (ADASYN, Random Undersampling)
- [ ] Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- [ ] Ensemble methods combining multiple models
- [ ] Deploy as REST API with authentication
- [ ] Real-time transaction monitoring dashboard
- [ ] Integration with payment gateways
- [ ] Mobile application development
- [ ] Explainable AI (SHAP values) for prediction interpretation

---

## 🎓 Project Objectives

1. **Exploratory Data Analysis**
   - Distribution analysis of transaction amounts and types
   - Temporal and geographic trend identification
   - Fraud ratio analysis

2. **Pattern Recognition**
   - Clustering techniques for unusual pattern detection
   - Feature correlation analysis

3. **Fraud Detection Modeling**
   - Multiple ML algorithm implementation
   - Model performance comparison
   - Feature importance analysis

4. **Real-World Application**
   - User-friendly web interface
   - Real-time fraud prediction
   - Practical deployment solution

---

## 📝 Key Learnings

- Handling imbalanced datasets in fraud detection
- Feature engineering impact on model performance
- Trade-offs between recall and precision in fraud systems
- Importance of data leakage prevention
- Real-time ML model deployment

---

## 👨‍💻 Author

**Suvan Agrawal**

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset inspired by real-world credit card transaction patterns
- Built as a college project demonstrating end-to-end ML development
- Focus on practical, deployable fraud detection solutions

---

**Project Status:** 🔄 In Development | **Last Updated:** February 2026