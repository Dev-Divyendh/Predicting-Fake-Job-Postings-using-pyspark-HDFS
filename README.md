# 🔍 Predicting Fake Job Postings Using Pyspark 

This PySpark-based project implements a machine learning pipeline to identify fraudulent job postings using the **Fake Job Posting Prediction** dataset. It includes data cleaning, feature extraction, handling class imbalance, and evaluating four ML models with cross-validation. The project demonstrates scalable text classification in a distributed environment.

---

## 👥 Contributors

- **Dev Divyendh Dhinakaran** (G01450299)  
- **Tejaswi Samineni** (G01460925)

---

## ⚙️ Technologies & Tools

- **PySpark** (MLlib + DataFrame API)
- **HDFS** (Perseus cluster)
- **TF-IDF**, **Tokenizer**, **StopWordsRemover**
- **CrossValidator**, **ParamGridBuilder**
- **MLlib Models:**
  - LogisticRegression
  - LinearSVC
  - RandomForestClassifier
  - MultilayerPerceptronClassifier

---

## 📋 Features

### ✅ Data Cleaning and Preparation
- Removed invalid `fraudulent` values (kept only `0` or `1`)
- Dropped columns with >1% missing values
- Cleaned text: removed non-letter characters, multiple spaces, lowercase conversion

### 🧮 Handling Class Imbalance
- Used PySpark's `sampleBy` to **undersample** majority class (fraudulent=0)

### 📊 Feature Engineering
- Tokenization of `title` and `description`
- Stopword removal
- TF-IDF vectorization

### 🔀 Data Split
- 70/30 random train-test split

---

## 🤖 Models Trained and Evaluated

| Model                        | Accuracy | F1 Score | Notes                                     |
|-----------------------------|----------|----------|-------------------------------------------|
| Logistic Regression         | 94.68%   | 94.44%   | Best performance overall                  |
| Linear SVC                  | 93.75%   | 93.52%   | High margin classifier                    |
| Random Forest Classifier    | 87.30%   | 82.69%   | Strong on non-linear patterns             |
| Multilayer Perceptron (MLP) | 94.25%   | 94.14%   | Neural network with 3 hidden layers       |

> All models were tuned using **10-fold Cross-Validation** with `CrossValidator`

---

## 🧪 Metrics & Evaluation

- Accuracy
- F1 Score
- ROC Curves & Confusion Matrices (for Random Forest & MLP)

---

## 🔍 Best Hyperparameters (per model)

- **Logistic Regression**
  - `regParam = 0.01`, `maxIter = 20`, `elasticNetParam = 0.0`, `family = auto`

- **Linear SVC**
  - `regParam = 1.0`, `maxIter = 100`, `standardization = true`, `aggregationDepth = 2`

- **Random Forest**
  - `numTrees = 10`, `maxDepth = 10`, `impurity = gini`, `bootstrap = true`

- **Multilayer Perceptron**
  - `layers = [20003, 5, 4, 3]`, `maxIter = 100`, `stepSize = 0.03`, `solver = l-bfgs`

---

## 📌 Dataset

- **Fake Job Posting Prediction** dataset  
  📥 [Download on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- Total Samples: ~18,000  
- Fraudulent Jobs: ~800 (≈4.4%)  

---

## 🧠 Key Learnings

- Real-world class imbalance handling via PySpark's `sampleBy`
- Distributed model training and validation with MLlib
- Effective preprocessing and text vectorization in Spark
- Understanding and interpreting `lift`, F1, and ROC metrics in fraud detection
