# 🏙️ City Prediction from Survey Data

## 📌 Project Overview

This project develops predictive machine learning models to classify cities based on structured and unstructured survey responses. The dataset consists of qualitative and quantitative features reflecting respondents’ perceptions and rankings of four global cities: Dubai, New York City, Paris, and Rio de Janeiro.

The workflow includes:
- Exploratory Data Analysis
- Data Preprocessing
- Baseline Model Training and Evaluation
- Hyperparameter Optimization
- Ensembling and Generalization Performance Evaluation
- Final Model Selection

---

## ▶️ How to Reproduce

1. Clone the repository
```
git clone https://github.com/sunggu-park/City-Prediction.git
cd City-Prediction
```

3. Install required dependencies
```
pip install -r requirements.txt
```

4. Run preprocessing and training:
   python main.py

---

## 🗂️ Dataset
**Source**: [Survey data](clean_dataset.csv) collected for this project  
**Size**: 1,469 rows × 12 columns  
**Input Features**: Responses to 10 survey questions capturing perceptions and city characteristics
**Target Variable**: City label (New York City, Rio de Janeiro, Dubai, Paris)

 **Feature Types**:

| Question | Type | Description |
|----------|------|-------------|
| Q1–Q4 | Ordinal | Ratings (1–5 scale) |
| Q5 | Categorical | Travel companion choice |
| Q6 | Structured string | Ranked attributes |
| Q7–Q9 | Numerical | Continuous user-entered values |
| Q10 | Text | Free-text response |

### Questionnaires used to collect data
- **Q1**: From a scale of 1 to 5, how popular is this city? (1 = least popular, 5 = most popular)  
- **Q2**: On a scale of 1 to 5, how efficient is this city at turning everyday occurrences into potential viral moments on social media?  
- **Q3**: Rate the city's architectural uniqueness from 1 to 5, where 5 represents a blend of futuristic innovation and historical charm.  
- **Q4**: Rate the city's enthusiasm for spontaneous street parties on a scale of 1 to 5.  
- **Q5**: If you were to travel to this city, who would most likely accompany you? (Partner, Friends, Siblings, Co-worker) 
- **Q6**: Rank the following attributes from least to most relatable (1 = least, 6 = most): Skyscrapers, Sport, Art & Music, Carnival, Cuisine, Economics
- **Q7**: What is the average temperature of this city in January? (°C)  
- **Q8**: How many different languages might you overhear while walking through the city?  
- **Q9**: How many different fashion styles might you observe within a 10-minute walk?  
- **Q10**: What quote comes to mind when you think of this city?

---

## 🔎 Exploratory Data Analysis (EDA)
EDA focused on understanding the distribution of survey responses across the four cities and identifying discriminative features.

### 🔹 Q1-Q4 (Perception-Based Ratings)
Grouped bar charts illustrate how cities are perceived in terms of popularity, social media virality, architectural uniqueness, and street party enthusiasm.

<p align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/616c3f5b-eadf-4d9a-aff2-e032c5bef047" width="500" alt="q1"></td>
      <td><img src="https://github.com/user-attachments/assets/f2f7815b-c790-48e3-b4a4-a1777a249982" width="500" alt="q2"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/38b6a435-001f-4812-b279-b50d2ec54951" width="500" alt="q3"></td>
      <td><img src="https://github.com/user-attachments/assets/91898b44-0b85-4de2-9c0b-99bd1364d221" width="500" alt="q4"></td>
    </tr>
  </table>
</p>

- New York City and Paris receive overwhelmingly high popularity ratings, whereas Dubai and Rio de Janeiro exhibit more balanced distributions. Consequently, Q1 (Popularity) appears to be a strong discriminative feature, effectively separating highly favored cities from those with more varied perceptions.

### 🔹 Q5 (Relationship Types)

<p align="center"> <img width="800" height="500" alt="q5" src="https://github.com/user-attachments/assets/aa766932-c75c-4466-9de2-6c9f3f449afc" /> </p>

- Paris has the highest proportion of respondents selecting “partner,” while “friends” is the most frequently chosen option for the other cities. Since responses are heavily concentrated in two categories across all cities, Q5 is expected to contribute limited additional predictive value and may introduce noise.

### 🔹 Q7-Q9 (Numerical Attributes)

<p align="center"> <img width="334" height="600" alt="image" src="https://github.com/user-attachments/assets/1e6f6819-91af-4867-a385-0ec85998cf80" /> </p>

- Significant outliers are observed in all three variables. For instance, in Q7 (Average January Temperature), the maximum value reaches 10,000, while the mean is only 21.8.

<p align="center">
  <table>
    <tr> 
      <td><img width="800" height="500" alt="q7" src="https://github.com/user-attachments/assets/5b824687-6a75-4d25-baf7-4329f002512e" /> </td>
      <td><img width="800" height="500" alt="q8" src="https://github.com/user-attachments/assets/e5483c98-80ef-461d-8f71-88baa260086c" /> </td>
    </tr>
    <tr>
      <td><img width="800" height="500" alt="q9" src="https://github.com/user-attachments/assets/c777a80c-f6dd-457c-b5bc-f646f72acd7c" /> </td>
    </tr>
  </table>
</p>

- Because these questions allow open numerical input, they exhibit substantial variance. Constraining the data to the Interquartile Range (IQR) improves stability and ensures more representative feature distributions.
   
### 🔹 Q10 (Free-text Responses)
<p align="center"> <img width="300" height="600" alt="스크린샷 2026-02-12 오전 3 05 25" src="https://github.com/user-attachments/assets/e0e90bad-b547-4f1e-babe-d8e6dc99b35c" /> </p>

- Text analysis reveals distinct and city-specific linguistic patterns. For example, Dubai is frequently associated with economically oriented words such as “rich,” “money,” and “oil,” while Paris is strongly linked to sentimental terms such as “love” and “romance.”
- City names themselves (e.g., “dubai,” “rio,” “paris”) appear frequently. Although this may introduce mild label leakage and artificially inflate predictive performance, retaining these tokens is justifiable, as they reflect natural linguistic behavior in survey responses.

### 🔹 Correlation Heatmap
<p align="center"> <img width="900" height="800" alt="corr" src="https://github.com/user-attachments/assets/b6ec4c69-507f-46de-9afc-a38708f54e23" /> </p>

- The highest absolute correlation (0.62) remains well below common multicollinearity thresholds (0.8–0.9), indicating low risk of instability in linear models.

- The predominance of weak-to-moderate correlations suggests that features convey largely independent information. This broadens predictive signals and allows models—particularly tree-based and neural network architectures—to learn complex relationships without excessive redundancy.

- Given the absence of severe multicollinearity, there is no immediate need for aggressive feature pruning based solely on inter-feature correlations.

---

## ⏳ Data Preprocessing

The preprocessing pipeline was modularized into a `process_data()` orchestration function, utilizing helper functions for:

 - **Q10 Feature Extraction**: Cleaning the raw text from Q10 responses, removing stopwords (e.g., “the,” “of,” “and”), and converting the text into numerical features using TF-IDF vectorization.
   
 - **Q6 Feature Extraction**: Parsing numerical rankings from structured string data (e.g., “Skyscrapers=>6, Sport=>1, Art and Music=>3, Carnival=>3, Cuisine=>3, Economics=>3”) and converting them into separate numerical features.
   
 - **Feature Stacking**: Combining all processed features (Q1-Q4, Q5 breakdowns, Q6 ranks, Q7-Q9 values, and Q10 TF-IDF features).
   
 - **Handling Missing Values and Format Errors**: Removing responses with invalid data types or incorrect formats.
   
 - **Handling Outliers**: Removing extreme outliers from numerical features (Q7-Q9) to improve model stability.
   
 - **Feature Selection and Normalization**: Selecting features and applying standardization to numerical features.
   
The processed dataset was split into training, validation, and test sets using an 80/15/5 ratio, and target labels were numerically encoded.

<p align="center"> <img width="301" height="162" alt="image" src="https://github.com/user-attachments/assets/64760d1c-6101-48b9-9814-be22ce427548" /> </p>

---

## ⚙️ Baseline Model Training and Evaluation

The following baseline models were trained and evaluated on the validation set:	K-Nearest Neighbors (KNN), Decision Tree,	Multinomial Logistic Regression (MLR), Multi-Layer Perceptron (MLP), Random Forest

| Model | Training Accuracy | Validation Accuracy | F1 Score |
|-------|-------------------|---------------------|----------|
| KNN | 0.9048 | 0.9005 | 0.90 |
| Decision Tree | 1.0 | 0.8294 | 0.83 |
| MLR | 0.9377 | 0.9384 | 0.94 |
| MLP | 1.0 | 0.9336 | 0.93 |
| Random Forest | 1.0 | 0.9147 | 0.92 |

Best-performing models:


1️⃣ **Multinomial Logistic Regression** achieved the highest validation accuracy of 0.9384. It showed strong and balanced performance with minimal difference between training and validation accuracy, indicating good generalization and no signs of significant overfitting.

2️⃣ **MLP Classifier** demonstrated a high validation accuracy of 0.9336. Although it achieved a perfect training accuracy (1.0) suggesting some overfitting, its strong performance on the validation set indicates good generalization.

3️⃣ **Random Forest Classifier** secured a validation accuracy of 0.9147. Similar to the MLP Classifier, it achieved perfect training accuracy but maintained a high validation accuracy, demonstrating robust performance and generalization capability.

Although KNN showed strong initial performance with a validation accuracy of around 0.9, it was consistently outperformed by Multinomial Logistic Regression and the MLP Classifier, both of which achieved higher validation accuracies. In addition, KNN’s performance is sensitive to the choice of k and to the curse of dimensionality, making it less robust in some scenarios compared to models that learn more complex decision boundaries.

In the case of the Decision Tree, the model achieved perfect training accuracy (1.0) but a much lower validation accuracy of 0.8294. This large gap between training and validation performance indicates overfitting, where the model learned the training data too well, including its noise, and failed to generalize effectively to unseen data. As a result, the Decision Tree is a less reliable choice for prediction compared to models with stronger generalization capabilities.

---

## 🛠️ Hyperparameter Tuning

Grid Search with K-Fold cross-validation was conducted for the three best-performing models. The performance of the tuned models was then compared to their pre-tuned counterparts.

### 🔹 Multinomial Logistic Regression

Best parameters: {'C': 0.5, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'saga'}

<table>
<tr>
<td width="50%">

### Pre-tuned Multinomial Logistic Regression

**Training Accuracy:** 0.9377  
**Validation Accuracy:** 0.9384  

**Classification Report (Validation Set):**
```
                 precision    recall  f1-score   support

         Dubai       0.93      0.89      0.91        56
 New York City       0.98      0.92      0.95        59
Rio de Janeiro       0.92      0.96      0.94        50
         Paris       0.92      1.00      0.96        46

      accuracy                           0.94       211
     macro avg       0.94      0.94      0.94       211
  weighted avg       0.94      0.94      0.94       211
```

**Confusion Matrix:**
```
[[50  1  3  2]
 [ 3 54  1  1]
 [ 1  0 48  1]
 [ 0  0  0 46]]
```

</td>
<td width="50%">

### Tuned Multinomial Logistic Regression

**Training Accuracy:** 0.9351  
**Validation Accuracy:** 0.9289  

**Classification Report (Validation Set):**
```
                 precision    recall  f1-score   support

         Dubai       0.92      0.86      0.89        56
 New York City       0.98      0.92      0.95        59
Rio de Janeiro       0.94      0.96      0.95        50
         Paris       0.87      1.00      0.93        46

      accuracy                           0.93       211
     macro avg       0.93      0.93      0.93       211
  weighted avg       0.93      0.93      0.93       211
```

**Confusion Matrix:**
```
[[48  1  2  5]
 [ 3 54  1  1]
 [ 1  0 48  1]
 [ 0  0  0 46]]
```

</td>
</tr>
</table>

Hyperparameter tuning did not improve validation performance. The baseline configuration slightly outperformed the tuned model (0.9384 vs. 0.9289 validation accuracy), indicating that the original regularization setting was already well-suited to the dataset.

### 🔹 MLP Classifier

Best parameters: {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (64,), 'solver': 'adam'}

<table>
<tr>
<td width="50%">

### Pre-tuned MLP Classifier

**Training Accuracy:** 1.0000  
**Validation Accuracy:** 0.9336  

**Classification Report (Validation Set):**
```
                 precision    recall  f1-score   support

         Dubai       0.94      0.86      0.90        56
 New York City       0.96      0.93      0.95        59
Rio de Janeiro       0.96      0.96      0.96        50
         Paris       0.87      1.00      0.93        46

      accuracy                           0.93       211
     macro avg       0.93      0.94      0.93       211
  weighted avg       0.94      0.93      0.93       211
```

**Confusion Matrix:**
```
[[48  1  2  5]
 [ 2 55  0  2]
 [ 1  1 48  0]
 [ 0  0  0 46]]
```

</td>
<td width="50%">

### Tuned MLP Classifier

**Training Accuracy:** 1.0000  
**Validation Accuracy:** 0.9431  

**Classification Report (Validation Set):**
```
                 precision    recall  f1-score   support

         Dubai       0.94      0.86      0.90        56
 New York City       0.97      0.95      0.96        59
Rio de Janeiro       0.96      0.98      0.97        50
         Paris       0.90      1.00      0.95        46

      accuracy                           0.94       211
     macro avg       0.94      0.95      0.94       211
  weighted avg       0.94      0.94      0.94       211
```

**Confusion Matrix:**
```
[[48  2  2  4]
 [ 2 56  0  1]
 [ 1  0 49  0]
 [ 0  0  0 46]]
```

</td>
</tr>
</table>

Tuning improved validation accuracy from 0.9336 to 0.9431 while maintaining perfect training accuracy. Although some overfitting remains, the tuned MLP demonstrates better class-level balance and improved generalization.

### 🔹 Random Forest

Best parameters: {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}


<table>
<tr>
<td width="50%">

### Pre-tuned Random Forest

**Training Accuracy:** 1.0000  
**Validation Accuracy:** 0.9147  

**Classification Report (Validation Set):**
```
                 precision    recall  f1-score   support

         Dubai       0.92      0.84      0.88        56
 New York City       0.93      0.90      0.91        59
Rio de Janeiro       0.96      0.98      0.97        50
         Paris       0.85      0.96      0.90        46

      accuracy                           0.91       211
     macro avg       0.91      0.92      0.92       211
  weighted avg       0.92      0.91      0.91       211
```

**Confusion Matrix:**
```
[[47  2  2  5]
 [ 3 53  0  3]
 [ 1  0 49  0]
 [ 0  2  0 44]]
```

</td>
<td width="50%">

### Tuned Random Forest

**Training Accuracy:** 0.9858  
**Validation Accuracy:** 0.9384  

**Classification Report (Validation Set):**
```
                precision    recall  f1-score   support

         Dubai       0.93      0.89      0.91        56
 New York City       0.98      0.92      0.95        59
Rio de Janeiro       0.94      0.98      0.96        50
         Paris       0.90      0.98      0.94        46

      accuracy                           0.94       211
     macro avg       0.94      0.94      0.94       211
  weighted avg       0.94      0.94      0.94       211
```

**Confusion Matrix:**
```
[[50  0  2  4]
 [ 3 54  1  1]
 [ 1  0 49  0]
 [ 0  1  0 45]]
```

</td>
</tr>
</table>

Tuning significantly improved validation accuracy (from 0.9147 to 0.9384) while slightly reducing training accuracy (from 1.0000 to 0.9858), indicating improved regularization. The tuned model exhibits stronger class-level performance and better generalization.

---

## 📊 Generalization Performance and Ensembling

Final evaluation was conducted on a completely held-out test set.

*   **Baseline Multinomial Logistic Regression:** 0.9577
*   **Tuned MLP Classifier:** 0.9437
*   **Tuned Random Forest Classifier:** 0.9437
   
<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/eb1105df-130b-4878-8cde-a0bc917380db" width="300"/>
<br>
</td>

<td align="center">
<img src="https://github.com/user-attachments/assets/cd45828e-6754-4756-b399-b95df3ccc176" width="300"/>
<br>
</td>

<td align="center">
<img src="https://github.com/user-attachments/assets/793603ef-13f7-47ea-b370-ff6f0b18cb4d" width="300"/>
<br>
</td>
</tr>
</table>

Two ensemble strategies were also evaluated:

*   **Soft Voting Classifier:** 0.9437
*   **Stacking Classifier:** 0.9437

<div align="center">
<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/f1ebcd99-9b91-4f34-b6c7-e011cf902736" width="400"/>
<br>
</td>

<td align="center">
<img src="https://github.com/user-attachments/assets/01dd33f8-2844-4c89-8f60-7eb0bb0101a0" width="400"/>
<br>
</td>
</tr>
</table>
</div>

---

## ✅ Final Model Selection

All evaluated models demonstrated strong generalization performance on the unseen test set, with accuracies exceeding 94%.

Among them, the baseline Multinomial Logistic Regression model achieved the highest test accuracy (0.9577) while maintaining robust and well-balanced class-level metrics. Despite the competitive performance of neural networks and ensemble methods, none provided a statistically meaningful improvement over MLR.

Given its superior test performance, balanced per-class behavior, interpretability, and lower computational complexity compared to neural networks and ensemble approaches, Multinomial Logistic Regression is selected as the final model.

