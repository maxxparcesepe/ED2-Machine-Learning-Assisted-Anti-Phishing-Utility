# 🛡️ Machine Learning-Assisted Anti-Phishing Utility

This project is an academic and applied initiative to develop a real-time phishing detection system using a machine learning model (Random Forest), integrated into a browser extension environment. It aims to address the limitations of traditional blacklist-based phishing protection by learning patterns from phishing URLs and HTML content.

---
## Prequel Tasks (Pre 5/10/2025)

-Designed Simple Website for Demonstration 
-Created Backend server with simple requests for login etc.
-Created functioning webpage based on demonstration that can be accesed at https://anti-phishing-flask-server.onrender.com/


## 📁 Project Structure

```
phishing-api/
├── rf_train.py                    # Random Forest training script
├── rf_model_combined.pkl         # Trained ML model
├── feature_order_combined.txt    # Ordered feature list for inference
├── feature_importance_plot.png   # Top 15 features chart
├── app.py                        # Flask API for predictions (planned)
├── requirements.txt              # Python dependencies
├── Training-Results(Folder)      # Model performance notes
```

---

## ✅ Completed Work as of 5/15/2025

### 1. Dataset Integration
- Loaded and analyzed two phishing datasets:
  - UCI PhiUSIIL Phishing URL Dataset
  - Kaggle Phising_Detection_Dataset.csv
- Unified column structures and removed non-numeric/textual data.
- Cleaned and merged datasets, resolving label inconsistencies.

### 2. Data Preprocessing
- Dropped missing or malformed rows.
- Downsampled majority class (legitimate) to balance phishing/legitimate ratio.
- Randomized and stratified dataset splits.

### 3. Model Training
- Trained a Random Forest classifier using 200 estimators.
- Exported model using joblib.
- Saved feature order for API compatibility.

### 4. Evaluation
- Achieved ~89.4% accuracy overall.
- Detected 97% of legitimate URLs and 52% of phishing URLs (recall).
- Created a full classification report and saved as TXT.
- Plotted top 15 features for transparency.

---

## 🔧 In Progress / To Be Implemented

### 🔜 5. API Deployment (Flask)
- Flask-based server for real-time model inference.
- Endpoint: `POST /predict` with JSON input vector
- Output: `"phishing"` or `"legitimate"`

### 🔜 6. Browser Extension Integration
- Frontend UI + content script
- Extracts features from current tab
- Sends them to Flask API and receives classification

### 🔜 7. Dataset Expansion
- Incorporate PhishStats and OpenPhish feeds
- Collect Alexa/Tranco top 1M for legitimate URLs
- Apply feature engineering to raw URLs in real time

### 🔜 8. Model Improvements
- Test Gradient Boosting / XGBoost for higher recall
- Apply class_weight="balanced" to improve phishing detection
- Evaluate performance using unseen 3rd-party datasets

---

## 🧪 Evaluation Metrics

- Precision, Recall, F1 Score (per class)
- Class Imbalance Awareness
- Feature importance ranking
- Confusion matrix plotting (future)

---

## 🔐 Project Goals

- Deliver a privacy-respecting, lightweight, ML-powered phishing detection engine.
- Support real-time analysis without storing user URLs.
- Be extensible: allow feature updates and model retraining via pipeline.

---

## 📌 Future Enhancements

- Integration with WHOIS or DNS APIs for real-time domain features
- Chrome Web Store deployment for testing
- User reporting feature (feedback loop)
- Continuous model updating with automated labeling

---

Developed by Project Lead Maxx Parcesepe — Florida Atlantic University, 2025  
Senior Design Capstone Project
