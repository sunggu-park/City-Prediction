# City Prediction from Survey Data

## 📌 Project Overview

---

## 🔧 Installation and Execution
1. Clone the repository
```bash
git clone https://github.com/sunggu-park/City-Prediction.git
cd City-Prediction
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python pred.py
```
---

## 🎯 Objectives
- Predict the target city from textual and structured survey responses
- Compare multiple machine learning algorithms for multiclass classification
- Improve predictive performance through feature engineering and hyperparameter tuning
- Evaluate models using appropriate performance metrics

---

## 🗂️ Dataset
- **Source**: [Survey data](data/clean_dataset.csv) collected for this project  
- **Size**: 1,469 rows × 12 columns  

### Dataset Structure
- **Input Features**: Responses to 10 survey questions capturing perceptions and characteristics of cities  
- **Target Variable**: City label (New York City, Rio de Janeiro, Dubai, Paris)

### Survey Questions
- **Q1**: From a scale of 1 to 5, how popular is this city? (1 = least popular, 5 = most popular)  
- **Q2**: On a scale of 1 to 5, how efficient is this city at turning everyday occurrences into potential viral moments on social media?  
- **Q3**: Rate the city's architectural uniqueness from 1 to 5, where 5 represents a blend of futuristic innovation and historical charm.  
- **Q4**: Rate the city's enthusiasm for spontaneous street parties on a scale of 1 to 5.  
- **Q5**: If you were to travel to this city, who would most likely accompany you?  
  - Partner, Friends, Siblings, Co-worker  
- **Q6**: Rank the following attributes from least to most relatable to this city  
  (1 = least relatable, 6 = most relatable):  
  - Skyscrapers  
  - Sport  
  - Art & Music  
  - Carnival  
  - Cuisine  
  - Economics  
- **Q7**: What is the average temperature of this city in January? (°C)  
- **Q8**: How many different languages might you overhear while walking through the city?  
- **Q9**: How many different fashion styles might you observe within a 10-minute walk?  
- **Q10**: What quote comes to mind when you think of this city? (Free-text response)

---

## 🧠 Methodology
### Data Preprocessing
- Cleaned and structured raw survey responses
- Handled outliers and missing values
- Encoded categorical variables and normalized numerical features
- Vectorized text data using the Bag-of-Words (BoW) technique
- Split data into training and validation sets

### Model Development
The following models were implemented and evaluated:
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Multi-layer Perceptron (MLP) Classifier**
- **Linear Regression model**
- **Logistic Regression model**
- **Multinomial Logistic Regression**

### Model Evaluation
- Compared models based on **classification accuracy** and error patterns
- Analyzed misclassifications to understand model limitations
- Selected the best-performing models for ensemble learning

### Hyperparameter Optimization
- Applied hyperparameter tuning to improve model generalization and stability
- Used grid search and cross-validation to identify optimal hyperparameter
- Tuned key parameters including regularization strength, tree depth, number of estimators, and kernel settings
- Observed measurable performance gains after optimization, contributing to improved ensemble results

### Ensemble Learning
- Combined predictions from multiple models
- Achieved improved generalization performance compared to individual models

---

## 📈 Results
- Achieved **~80% prediction accuracy** on the test set
- Ensemble approach outperformed individual baseline models
- Demonstrated the effectiveness of combining linear and nonlinear classifiers

---
