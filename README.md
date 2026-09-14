# AutoClassify

### Intelligent Dataset-Aware Classification Algorithm Selection Framework

AutoClassify is an intelligent framework that analyzes the characteristics of a classification dataset and recommends a suitable classification algorithm using a **dataset-aware suitability engine**.

Instead of simply training all algorithms and selecting the model with the highest accuracy, AutoClassify first analyzes the dataset characteristics and generates an analytical recommendation.

## 🚀 Live Demo

[**Open AutoClassify Web App**](https://dataset-aware-classification-algorithm-selection-framework-cqr.streamlit.app/)

## 📌 Project Overview

AutoClassify analyzes a classification dataset based on characteristics such as:

* Number of instances
* Number of features
* Numerical features
* Categorical features
* Binary features
* Number of target classes
* Class distribution
* Class imbalance
* Missing values
* Duplicate records
* Feature-to-sample ratio
* Feature correlation

Based on these characteristics, the framework calculates algorithm suitability scores and produces an **analytical prediction**.

The analytical prediction is then compared with experimental model performance.

## 🤖 Algorithms Considered

AutoClassify considers the following classification algorithms:

1. Decision Tree
2. Naive Bayes
3. Support Vector Machine (SVM)

For SVM, both **Linear** and **RBF** configurations are considered during hyperparameter experimentation.

## 🔄 System Workflow

```text
Classification Dataset
        ↓
Dataset Profiling
        ↓
Automated Preprocessing
        ↓
Dataset Characteristics
        ↓
Suitability Engine
        ↓
Analytical Prediction
        ↓
Experimental Validation
        ↓
SVM Hyperparameter Experiment
        ↓
Analytical vs Experimental Comparison
        ↓
Final Recommendation
```

## 🔍 Main Features

### 1. Dataset Profiling

The framework automatically analyzes the uploaded dataset and provides:

* Number of instances
* Number of features
* Number of classes
* Numerical feature count
* Categorical feature count
* Binary feature count
* Missing percentage
* Duplicate records
* Class distribution
* Imbalance ratio
* Feature/sample ratio
* Statistical characteristics
* Feature correlation

### 2. Automated Preprocessing

AutoClassify performs preprocessing according to the dataset characteristics.

The preprocessing stage includes:

* Missing value handling
* Duplicate handling
* Categorical encoding
* Numerical feature processing
* Feature scaling where required
* Target encoding
* Train-test splitting

The preprocessing decisions are displayed in the application.

### 3. Dataset-Aware Suitability Engine

The main component of AutoClassify is the **Suitability Engine**.

The engine analyzes dataset characteristics before experimental model training and assigns suitability scores to the candidate algorithms.

The scoring considers factors such as:

* Dataset size
* Numerical and categorical features
* Number of classes
* Class imbalance
* Feature/sample ratio
* Missing values

The algorithm with the highest suitability score becomes the **Analytical Prediction**.

### 4. Experimental Validation

The candidate algorithms are experimentally evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* 5-fold Cross-Validation F1
* Training Time
* Prediction Time
* Confusion Matrix

The experimental results are compared with the analytical prediction.

### 5. SVM Hyperparameter Experiment

AutoClassify performs additional SVM experiments using different hyperparameter configurations.

The experiment includes:

* Linear SVM
* RBF SVM
* Different `C` values
* Different `gamma` values for RBF
* 5-fold cross-validation

The best SVM configuration is reported by the application.

### 6. Explainable Recommendation

AutoClassify provides an explanation for its analytical prediction.

The application displays:

```text
Analytical Prediction
        ↓
Experimental Winner
        ↓
Prediction Status
        ↓
Explainable Recommendation
```

If the analytical prediction and experimental winner are the same, the prediction is marked as:

```text
CONFIRMED
```

If they differ, the prediction is marked as:

```text
NOT CONFIRMED
```

and a failure analysis is provided.

## 📊 Dataset Used

The primary dataset used for development and demonstration is the **Bank Marketing Dataset**.

The target variable is:

```text
y
```

The target represents whether the client subscribed to a term deposit.

The dataset contains both numerical and categorical attributes, making it suitable for demonstrating automated preprocessing and dataset-aware algorithm selection.

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* GitHub

## 📁 Project Structure

```text
dataset-aware-classification-algorithm-selection-framework/
│
├── app.py
├── autoclassify.py
├── requirements.txt
└── README.md
```

### `app.py`

Contains the Streamlit web application and user interface.

### `autoclassify.py`

Contains the core AutoClassify implementation, including:

* Dataset profiling
* Automated preprocessing
* Suitability analysis
* Analytical prediction
* Model training
* Model evaluation
* SVM hyperparameter experimentation
* Final recommendation

### `requirements.txt`

Contains the Python libraries required to run the application.

## 💻 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd dataset-aware-classification-algorithm-selection-framework
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## ▶️ How to Use

1. Open the [**AutoClassify Web App**](https://dataset-aware-classification-algorithm-selection-framework-cqr.streamlit.app/).
2. Upload a classification CSV dataset.
3. AutoClassify profiles the dataset.
4. The target column `y` is automatically selected for the Bank Marketing dataset.
5. Click **Run AutoClassify**.
6. View the dataset profile.
7. View automated preprocessing decisions.
8. View algorithm suitability scores.
9. View the analytical prediction.
10. View experimental validation results.
11. View SVM hyperparameter results.
12. Compare the analytical prediction with the experimental winner.
13. View the final recommendation.

## 📈 Evaluation

AutoClassify evaluates the models using multiple performance measures rather than relying only on accuracy.

The main evaluation measures are:

```text
Accuracy
Precision
Recall
F1 Score
5-Fold Cross-Validation F1
Training Time
Prediction Time
```

This allows both predictive performance and computational performance to be considered.

## 🎯 Analytical Prediction vs Experimental Winner

AutoClassify compares:

**Analytical Prediction**

The algorithm selected by the suitability engine based on dataset characteristics.

**Experimental Winner**

The algorithm that achieves the best overall experimental evaluation.

The framework checks:

```text
Analytical Prediction == Experimental Winner
```

If they match:

```text
Prediction Status: CONFIRMED
```

If they do not match:

```text
Prediction Status: NOT CONFIRMED
```

This comparison helps evaluate how effectively the dataset-aware suitability engine predicts algorithm performance.

## 🌐 Deployment

The application is deployed using **Streamlit** and hosted through the project's GitHub repository.

### Live Application

[**AutoClassify — Streamlit App**](https://dataset-aware-classification-algorithm-selection-framework-cqr.streamlit.app/)

## 🔮 Future Improvements

Possible future improvements include:

* Testing on more development datasets
* Testing on unseen validation datasets
* Adding more classification algorithms
* Refining suitability criteria
* Learning suitability weights from multiple datasets
* Supporting more multiclass datasets
* Adding additional dataset complexity measures
* Improving computational efficiency
* Adding interactive visualizations
* Improving failure analysis
* Improving generalization to unseen datasets

## 🎓 Project Objective

The objective of AutoClassify is to develop a **dataset-aware framework for automatic classification algorithm selection**.

The framework first analyzes the characteristics of a dataset, predicts a suitable algorithm using a suitability engine, and then validates the prediction through experimental evaluation.

This approach provides both **algorithm selection and explainability**, rather than relying only on model accuracy.

## 👨‍💻 Project

**Project Name:** AutoClassify
**Domain:** Machine Learning / Data Mining / Automated Algorithm Selection
**Application:** Streamlit

### 🔗 Quick Links

* [**Live Streamlit Application**](https://dataset-aware-classification-algorithm-selection-framework-cqr.streamlit.app/)
* [**GitHub Repository**](https://github.com/](https://github.com/Sudipta-Mitra/Dataset-Aware-Classification-Algorithm-Selection-Framework)
