# CSC311 Final Project: City Prediction from Survey Data

## 📌 Project Overview
This project focuses on building a machine learning system that predicts the **city a respondent is describing** based on their **survey responses**. The task is formulated as a **multiclass classification problem**, where each class represents a different city. The project emphasizes end-to-end machine learning workflows, including data preprocessing, model development, evaluation, and comparison.

---

## 🎯 Objectives
- Predict the target city from textual and structured survey responses
- Compare multiple machine learning algorithms for multiclass classification
- Improve predictive performance through feature engineering and model tuning
- Evaluate models using appropriate performance metrics

---

## 🗂️ Dataset
- **Source**: Survey data collected for the CSC311 course project
- **Input Features**:
  - Structured survey responses describing characteristics of cities
  - Encoded numerical and categorical features derived from responses
- **Target Variable**:
  - City label (multiclass)

---

## 🧠 Methodology
### 1. Data Preprocessing
- Cleaned and structured raw survey responses
- Encoded categorical variables and normalized numerical features
- Split data into training and validation sets

### 2. Model Development
The following models were implemented and evaluated:
- **Multinomial Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

All models were implemented using **NumPy**, **Pandas**, and **scikit-learn**, with careful control of hyperparameters and regularization.

### 3. Model Evaluation
- Compared models based on **classification accuracy** and error patterns
- Analyzed misclassifications to understand model limitations
- Selected the best-performing models for ensemble learning

### 4. Ensemble Learning
- Combined predictions from multiple models
- Achieved improved generalization performance compared to individual models

---

## 📈 Results
- Achieved approximately **80% prediction accuracy** on the validation set
- Ensemble approach outperformed individual baseline models
- Demonstrated the effectiveness of combining linear and nonlinear classifiers

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - NumPy
  - Pandas
  - scikit-learn
- **Techniques**:
  - Multiclass classification
  - Feature engineering
  - Model comparison and ensemble learning

---

## 🔍 Key Takeaways
- Proper feature engineering significantly impacts model performance
- Different models capture different aspects of survey data
- Ensemble methods can effectively leverage the strengths of multiple classifiers

---

## 🚀 Future Improvements
- Incorporate NLP techniques for free-text survey responses
- Apply advanced ensemble methods such as stacking
- Explore deep learning models for improved representation learning

---

## 📎 Notes
This project was completed as part of **CSC311 (Introduction to Machine Learning)** and demonstrates practical experience with supervised learning and model evaluation in a real-world classification setting.

