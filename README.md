# Machine Learning vs Deep Learning: Predicting FIFA 21 Player Tiers ⚽

This project explores how traditional **Machine Learning (ML)** models and **Deep Learning (DL)** models perform in predicting FIFA 21 player tiers (High, Mid, Low).  
The study compares decision tree–based classifiers, ensemble models, and artificial neural networks to evaluate accuracy, precision, recall, and F1-score.  

---

## 📂 Dataset
- Source: **FIFA 21 Player Dataset** (Kaggle)  
- Number of samples: ~3,789 players  
- Target variable: **Player Tier**
  - **High** (top-rated players)  
  - **Mid** (average-tier players)  
  - **Low** (lower-tier players)  
- Features: Player statistics such as:
  - Overall rating, potential, pace, shooting, passing, dribbling, defending, physical, wage, and more.

---

## 🧠 Models Used
We applied both **traditional ML algorithms** and **deep learning approaches**:

### Machine Learning
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Naïve Bayes Classifier**

### Deep Learning
- **Multilayer Perceptron (ANN)**

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Normalization/standardization of numerical features  
   - Splitting into Train/Test sets  

2. **Model Development**
   - Train ML and DL models on preprocessed data  
   - Perform hyperparameter tuning (where applicable)  

3. **Model Evaluation**
   - Metrics used:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Comparison of results across ML vs DL models  

---

## 📊 Results

### Decision Tree
- Accuracy: ~97.8%  
- Strength: Simple and interpretable  
- Weakness: Overfitting possible

### Random Forest
- Accuracy: ~98.2%  
- Strength: High precision & recall, robust performance  
- Weakness: Slower training compared to DT

### Naïve Bayes
- Accuracy: ~90–92%  
- Strength: Fast and efficient on high-dimensional data  
- Weakness: Assumes independence of features

### ANN (Deep Learning)
- Accuracy: ~96–97%  
- Strength: Can capture non-linear relationships  
- Weakness: Needs more tuning & longer training

📌 **Observation:** Traditional ensemble methods (Random Forest) slightly outperformed the Deep Learning model in this dataset due to structured/tabular data.  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<USERNAME>/Machine-Learning-vs-Deep-Learning-Predicting-FIFA-21-Player-Tiers.git
   cd Machine-Learning-vs-Deep-Learning-Predicting-FIFA-21-Player-Tiers
