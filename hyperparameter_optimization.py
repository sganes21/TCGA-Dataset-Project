#
# Optimize Hyperparameters
#
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

def optimize_random_forest(trial, features_df, target):
    '''
    Function for Optuna library to optimize Random Forest hyperparameters.

    Inputs:
        trial(optuna.trial.Trial): A single trial object from Optuna.
        features_df(pd.DataFrame): DataFrame containing the features.
        target(pd.Series): Target variable.

    Returns:
        mean_score(float): Mean AUROC score across K-Fold cross-validation.
    '''
    # Hyperparameter ranges for optimization
    n_estimators = trial.suggest_int('n_estimators', 20, 300)
    max_depth = trial.suggest_int('max_depth', 1, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Identify numeric and non-numeric  columns
    numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features_df.select_dtypes(include=['object']).columns

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))  # Impute missing values with mean for numeric data
    ])

    # Preprocessing for non-numeric features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with mode for categorical data
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline([ ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf, random_state=0))
    ])

    # Perform cross-validation and calculate mean AUROC score
    scores = cross_val_score(pipeline, features_df, target, cv=kf, scoring='roc_auc')

    return np.mean(scores)
   

