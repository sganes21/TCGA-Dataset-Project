#
# Random Forest Model
#
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from hyperparameter_optimization import optimize_random_forest
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category= UndefinedMetricWarning)


def train_random_forest(features_df, target):
    '''
    Performs K-Fold cross-validation with a Random Forest Classifier

    Inputs:
        features_df(pd.DataFrame): DataFrame containing the features
        target(pd.Series): Target variable

    Returns:
        model: trained Random Forest model
        best_params: the best tuned parameters consisting of the following: 
            n_estimators: Optimized between 20 & 300 
            max_depth: Optimized between 1 & 30  
            min_samples_split: Optimized between 2 & 20 
            min_samples_leaf: Optimized between 5 & 20  
    '''
    # Initialize optimization libray and define optimum variable values
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_random_forest(trial, features_df, target), n_trials=100)
    best_params = study.best_params

    # Identify numeric and non-numeric or string columns
    numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features_df.select_dtypes(include=['object']).columns

    # Preprocessing number based features using median based strategy
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))  
    ])

    # Preprocessing for categorical features using most-frequent based strategy
    # One-hot encode categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    # Output pipeline
    pipeline = Pipeline([ ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],min_samples_split=best_params['min_samples_split'],min_samples_leaf=best_params['min_samples_leaf'],class_weight=best_params['class_weight'], random_state=0))
    ])

    # Fit pipeline on target labels
    pipeline.fit(features_df, target)

    return pipeline, best_params



   

