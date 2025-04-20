#
# Evaluation
#
import pandas as pd
from feature_generation import feature_extraction
from random_forest_model_train import train_random_forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category= UndefinedMetricWarning)

# Load data
clinical_df = pd.read_csv('Data/clinical.tsv', sep='\t')
follow_up_df= pd.read_csv('Data/follow_up.tsv', sep='\t')

# Merge Dataset

total_data_df = pd.merge(clinical_df, follow_up_df, on='cases.case_id', how='left')

# Extract features from data
features_df = feature_extraction(total_data_df)

# Add features from data into pandas df and visualize 
print(features_df.head(25))

# Split the data into features and target 
final_df=features_df.dropna(subset=['cases.disease_type'])
X = final_df.drop('cases.disease_type', axis=1)
y = final_df['cases.disease_type']

# Split into training and test datasets

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=0, stratify=y_temp
)

# Check for missing classes
print("\nValidation class distribution:")
print(y_val.value_counts())

# Train on the data and generate the model
#target = train_df['label']
model, best_params = train_random_forest(X_train, y_train)
print("Best Parameters:", best_params)


# Validation Evaluation
val_preds = model.predict(X_val)
print("Validation Results:")
print(f"Precision: {precision_score(y_val, val_preds, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_val, val_preds, average='weighted', zero_division=0):.4f}")
print(classification_report(y_val, val_preds, zero_division=0))

# Test Evaluation
test_preds = model.predict(X_test)
print("\nTest Results:")
print(f"Precision: {precision_score(y_test, test_preds, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, test_preds, average='weighted', zero_division=0):.4f}")
print(classification_report(y_test, test_preds, zero_division=0))


