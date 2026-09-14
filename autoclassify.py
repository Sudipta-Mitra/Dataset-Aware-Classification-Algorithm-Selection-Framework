# ============================================================
# AUTOCLASSIFY.PY
# Dataset-Aware Classification Algorithm Selection Framework
# ============================================================

import os
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def AutoClassify(file_path, target_column):

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    try:
        data = pd.read_csv(file_path, sep=";")
    except Exception:
        data = pd.read_csv(file_path)

    data = data.copy()

    # Remove unnecessary index column if present
    unnamed_columns = [
        col for col in data.columns
        if str(col).lower().startswith("unnamed")
    ]

    if unnamed_columns:
        data.drop(columns=unnamed_columns, inplace=True)


    # ========================================================
    # 2. DATASET PROFILE
    # ========================================================

    instances = data.shape[0]
    total_columns = data.shape[1]
    features = total_columns - 1

    X_original = data.drop(columns=[target_column])
    y_original = data[target_column]

    numerical_columns = X_original.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X_original.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numerical_features = len(numerical_columns)
    categorical_features = len(categorical_columns)

    binary_features = 0

    for column in X_original.columns:

        if X_original[column].nunique(
            dropna=True
        ) == 2:

            binary_features += 1


    classes = y_original.nunique()

    class_distribution = (
        y_original.value_counts(
            normalize=True
        )
        .round(4)
        .to_dict()
    )

    class_counts = y_original.value_counts()

    if len(class_counts) > 1:

        imbalance_ratio = (
            class_counts.max()
            /
            class_counts.min()
        )

    else:

        imbalance_ratio = 1.0


    missing_percentage = (
        data.isnull().sum().sum()
        /
        (data.shape[0] * data.shape[1])
    ) * 100


    duplicate_records = data.duplicated().sum()

    feature_sample_ratio = (
        features / instances
    )


    # ========================================================
    # 3. STATISTICS
    # ========================================================

    statistics = X_original.describe(
        include="all"
    ).transpose()


    # ========================================================
    # 4. CORRELATION
    # ========================================================

    if numerical_features > 1:

        correlation = X_original[
            numerical_columns
        ].corr()

    else:

        correlation = pd.DataFrame()


    profile = {

        "instances": instances,

        "features": features,

        "classes": classes,

        "numerical_features":
            numerical_features,

        "categorical_features":
            categorical_features,

        "binary_features":
            binary_features,

        "class_distribution":
            class_distribution,

        "imbalance_ratio":
            imbalance_ratio,

        "missing_percentage":
            missing_percentage,

        "duplicate_records":
            duplicate_records,

        "feature_sample_ratio":
            feature_sample_ratio,

        "statistics":
            statistics,

        "correlation":
            correlation
    }


    # ========================================================
    # 5. DATA QUALITY ANALYSIS
    # ========================================================

    preprocessing_decisions = []

    if duplicate_records > 0:

        preprocessing_decisions.append(
            f"{duplicate_records} duplicate records detected "
            "and removed."
        )

        data = data.drop_duplicates()

    else:

        preprocessing_decisions.append(
            "No duplicate records detected."
        )


    missing_values = data.isnull().sum().sum()

    if missing_values > 0:

        preprocessing_decisions.append(
            "Missing values detected and automatically "
            "handled using imputation."
        )

    else:

        preprocessing_decisions.append(
            "No missing values detected."
        )


    # Remove rows where target is missing
    data = data.dropna(
        subset=[target_column]
    )


    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]


    # ========================================================
    # 6. TARGET ENCODING
    # ========================================================

    if y.dtype == "object":

        target_mapping = {
            value: index
            for index, value
            in enumerate(
                sorted(y.unique())
            )
        }

        y = y.map(target_mapping)

        preprocessing_decisions.append(
            "Categorical target values were encoded "
            "into numerical class labels."
        )

    else:

        target_mapping = None


    # ========================================================
    # 7. FEATURE IDENTIFICATION
    # ========================================================

    numerical_features_list = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features_list = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()


    # ========================================================
    # 8. TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # ========================================================
    # 9. PREPROCESSING PIPELINES
    # ========================================================

    numeric_scaled = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    numeric_unscaled = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


    # ========================================================
    # 10. DECISION TREE PREPROCESSOR
    # ========================================================

    tree_preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_unscaled,
                numerical_features_list
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features_list
            )
        ],

        remainder="drop"
    )


    # ========================================================
    # 11. SVM / NB PREPROCESSOR
    # ========================================================

    scaled_preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_scaled,
                numerical_features_list
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features_list
            )
        ],

        remainder="drop"
    )


    # ========================================================
    # 12. MODELS
    # ========================================================

    decision_tree_model = Pipeline(

        steps=[

            (
                "preprocessor",
                tree_preprocessor
            ),

            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]
    )


    naive_bayes_model = Pipeline(

        steps=[

            (
                "preprocessor",
                scaled_preprocessor
            ),

            (
                "classifier",
                GaussianNB()
            )
        ]
    )


    svm_model = Pipeline(

        steps=[

            (
                "preprocessor",
                scaled_preprocessor
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale"
                )
            )
        ]
    )


    models = {

        "Decision Tree":
            decision_tree_model,

        "Naive Bayes":
            naive_bayes_model,

        "SVM":
            svm_model
    }


    # ========================================================
    # 13. ALGORITHM SUITABILITY ENGINE
    # ========================================================

    scores = {

        "Decision Tree": 0.0,

        "Naive Bayes": 0.0,

        "SVM": 0.0
    }


    reasons = {

        "Decision Tree": [],

        "Naive Bayes": [],

        "SVM": []
    }


    # --------------------------------------------------------
    # Dataset Size
    # --------------------------------------------------------

    if instances < 5000:

        scores["Decision Tree"] += 1

        scores["Naive Bayes"] += 2

        scores["SVM"] += 1

    elif instances < 20000:

        scores["Decision Tree"] += 2

        scores["Naive Bayes"] += 2

        scores["SVM"] += 2

    else:

        scores["Decision Tree"] += 2

        scores["Naive Bayes"] += 2

        scores["SVM"] += 1

        reasons["Naive Bayes"].append(
            "Naive Bayes is computationally efficient "
            "for a relatively large dataset."
        )


    # --------------------------------------------------------
    # Categorical Features
    # --------------------------------------------------------

    if categorical_features > 0:

        scores["Decision Tree"] += 2

        scores["Naive Bayes"] += 1

        scores["SVM"] += 1

        reasons["Decision Tree"].append(
            "The dataset contains categorical features, "
            "which can be handled effectively after encoding."
        )


    # --------------------------------------------------------
    # Numerical Features
    # --------------------------------------------------------

    if numerical_features > categorical_features:

        scores["SVM"] += 2

        scores["Naive Bayes"] += 2

        reasons["SVM"].append(
            "The dataset contains a substantial number "
            "of numerical features, making feature scaling "
            "useful for SVM."
        )

        reasons["Naive Bayes"].append(
            "The numerical feature structure is compatible "
            "with Gaussian Naive Bayes."
        )


    # --------------------------------------------------------
    # Number of Classes
    # --------------------------------------------------------

    if classes <= 2:

        scores["SVM"] += 2

        scores["Decision Tree"] += 2

        scores["Naive Bayes"] += 2

        reasons["SVM"].append(
            "The problem is binary classification, which "
            "is well supported by SVM."
        )


    # --------------------------------------------------------
    # Imbalance
    # --------------------------------------------------------

    if imbalance_ratio > 3:

        scores["Decision Tree"] += 1

        scores["SVM"] += 1

        reasons["Decision Tree"].append(
            "The target distribution is imbalanced, so "
            "F1-score and class-sensitive evaluation are important."
        )

    else:

        scores["Naive Bayes"] += 1


    # --------------------------------------------------------
    # Feature / Sample Ratio
    # --------------------------------------------------------

    if feature_sample_ratio > 0.05:

        scores["SVM"] += 2

        scores["Naive Bayes"] += 1

        reasons["SVM"].append(
            "The feature-to-sample ratio is relatively high, "
            "which supports the use of a margin-based classifier."
        )

    else:

        scores["Decision Tree"] += 1


    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    if missing_percentage > 0:

        scores["Decision Tree"] += 1

        reasons["Decision Tree"].append(
            "Missing values are present and are handled "
            "through automated preprocessing."
        )


    # ========================================================
    # 14. NORMALIZE SUITABILITY SCORES
    # ========================================================

    max_score = max(scores.values())

    if max_score > 0:

        suitability_scores = {

            algorithm:
                round(
                    score / max_score * 100,
                    2
                )

            for algorithm, score
            in scores.items()
        }

    else:

        suitability_scores = {
            algorithm: 0
            for algorithm in scores
        }


    # Analytical prediction
    analytical_prediction = max(
        suitability_scores,
        key=suitability_scores.get
    )


    # Add default reasons if necessary

    for algorithm in reasons:

        if len(reasons[algorithm]) == 0:

            reasons[algorithm].append(
                "The dataset characteristics provide "
                "limited additional suitability evidence "
                "for this algorithm."
            )


    # ========================================================
    # 15. EXPERIMENTAL MODEL EVALUATION
    # ========================================================

    experimental_results = []

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    for algorithm, model in models.items():

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        start_time = time.time()

        y_pred = model.predict(
            X_test
        )

        prediction_time = (
            time.time() - start_time
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        # ----------------------------------------------------
        # 5-Fold CV
        # ----------------------------------------------------

        cv_scores = cross_val_score(

            model,

            X,

            y,

            cv=cv,

            scoring="f1_weighted"
        )

        cv_f1_mean = cv_scores.mean()


        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )


        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )


        # ----------------------------------------------------
        # Evaluation Score
        # ----------------------------------------------------

        evaluation_score = (

            0.30 * f1

            +

            0.20 * precision

            +

            0.20 * recall

            +

            0.30 * cv_f1_mean
        )


        experimental_results.append({

            "Algorithm":
                algorithm,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1 Score":
                f1,

            "CV F1 Mean":
                cv_f1_mean,

            "Training Time":
                training_time,

            "Prediction Time":
                prediction_time,

            "Evaluation Score":
                evaluation_score,

            "Confusion Matrix":
                cm.tolist(),

            "Classification Report":
                report
        })


    results_df = pd.DataFrame(
        experimental_results
    )


    # ========================================================
    # 16. EXPERIMENTAL WINNER
    # ========================================================

    experimental_winner = (

        results_df
        .sort_values(
            "Evaluation Score",
            ascending=False
        )
        .iloc[0]["Algorithm"]
    )


    # ========================================================
    # 17. PREDICTION STATUS
    # ========================================================

    prediction_confirmed = (

        analytical_prediction
        ==
        experimental_winner
    )


    # ========================================================
    # 18. SVM HYPERPARAMETER EXPERIMENT
    # ========================================================

    svm_experiments = []


    # Linear SVM

    for C in [0.1, 1, 10]:

        model = Pipeline(

            steps=[

                (
                    "preprocessor",
                    scaled_preprocessor
                ),

                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=C
                    )
                )
            ]
        )


        cv_score = cross_val_score(

            model,

            X,

            y,

            cv=cv,

            scoring="f1_weighted"
        ).mean()


        svm_experiments.append({

            "Kernel":
                "Linear",

            "C":
                C,

            "Gamma":
                "-",

            "CV F1":
                cv_score
        })


    # RBF SVM

    for C in [0.1, 1, 10]:

        for gamma in [
            "scale",
            0.01,
            0.1
        ]:

            model = Pipeline(

                steps=[

                    (
                        "preprocessor",
                        scaled_preprocessor
                    ),

                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=C,
                            gamma=gamma
                        )
                    )
                ]
            )


            cv_score = cross_val_score(

                model,

                X,

                y,

                cv=cv,

                scoring="f1_weighted"
            ).mean()


            svm_experiments.append({

                "Kernel":
                    "RBF",

                "C":
                    C,

                "Gamma":
                    gamma,

                "CV F1":
                    cv_score
            })


    svm_experiment_df = pd.DataFrame(
        svm_experiments
    )


    best_svm_row = (

        svm_experiment_df
        .sort_values(
            "CV F1",
            ascending=False
        )
        .iloc[0]
    )


    svm_best_parameters = {

        "kernel":
            best_svm_row["Kernel"],

        "C":
            best_svm_row["C"],

        "gamma":
            best_svm_row["Gamma"]
    }


    svm_best_cv_score = (
        best_svm_row["CV F1"]
    )


    # ========================================================
    # 19. SAVE MODELS AS PICKLE
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )


    # Save the already trained pipelines

    with open(
        "models/decision_tree.pkl",
        "wb"
    ) as file:

        pickle.dump(
            decision_tree_model,
            file
        )


    with open(
        "models/naive_bayes.pkl",
        "wb"
    ) as file:

        pickle.dump(
            naive_bayes_model,
            file
        )


    with open(
        "models/svm.pkl",
        "wb"
    ) as file:

        pickle.dump(
            svm_model,
            file
        )


    # ========================================================
    # 20. FINAL RESULT
    # ========================================================

    return {

        "profile":
            profile,

        "preprocessing_decisions":
            preprocessing_decisions,

        "suitability_scores":
            suitability_scores,

        "analytical_prediction":
            analytical_prediction,

        "suitability_reasons":
            reasons,

        "experimental_results":
            results_df,

        "svm_best_parameters":
            svm_best_parameters,

        "svm_best_cv_score":
            svm_best_cv_score,

        "experimental_winner":
            experimental_winner,

        "prediction_confirmed":
            prediction_confirmed
    }
