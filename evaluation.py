#
# Evaluation
#
import pandas as pd
import numpy as np
from feature_generation import feature_extraction
from random_forest_model_train import train_random_forest
from data_preprocessor import data_processing
from data_loader import load_data
from deep_learning import create_clinical_neuralnet
from deep_learning_training import train_model, evaluate, extract_matrix_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
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

## Traditional Pipeline

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
report_random_forest_validation = classification_report(y_val, val_preds, zero_division=0)
print(report_random_forest_validation)
with open("Analysis/classification_report_val_ml.txt", "w") as f:
    f.write(report_random_forest_validation)

# Test Evaluation
test_preds = model.predict(X_test)
print("\nTest Results:")
print(f"Precision: {precision_score(y_test, test_preds, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, test_preds, average='weighted', zero_division=0):.4f}")
report_random_forest = classification_report(y_test, test_preds, zero_division=0)
print(report_random_forest)
with open("Analysis/classification_report_test_ml.txt", "w") as f:
    f.write(report_random_forest)


# Generate confusion matrix
cm = confusion_matrix(y_val, val_preds)
classes = y_val.unique()
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Plot confusion matrix and saving
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Validation Set - Traditional ML Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('Analysis/confusion_matrix_validation_ml.png', bbox_inches='tight', dpi=300)


## Neural Net Pipeline
print("\nCommencing Neural Net Pipeline:")

# Entering data in pre-processing pipeline
[X_train_res, y_train_res, train_dataset, val_dataset, test_dataset]=data_processing(X, y)

# Setting up dataloaders for the training, validation, and test data
[train_loader, val_loader, test_loader, class_weights]=load_data(y_train_res, train_dataset, val_dataset, test_dataset)
input_size = X_train_res.shape[1]
num_classes = len(np.unique(y))

# Defining Deep Learning Model
model = create_clinical_neuralnet(input_size, num_classes)
num_epochs=100

# Setting up class weights, scheduler, and optimizer.
class_weights = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


# Training the model with defined paramters.
trained_model= train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs)

# Load best model
trained_model.load_state_dict(torch.load('best_model.pth'))

# Print results
print("Validation Results:")
report_deep_learning_validation=evaluate(trained_model, val_loader)
print(report_deep_learning_validation)
with open("Analysis/classification_report_val_deep_learning.txt", "w") as f:
    f.write(report_deep_learning_validation)

# Compute confusion matrix
all_preds, all_labels = extract_matrix_labels(trained_model, val_loader)
cmval = confusion_matrix(all_preds, all_labels)
classes_dl = y_val.unique()
cmval_df = pd.DataFrame(cmval, index=classes_dl, columns=classes_dl)

# Plot confusion matrix and saving
plt.figure(figsize=(12, 10))
sns.heatmap(cmval_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Validation Set - Deep Learning Model Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('Analysis/confusion_matrix_validation_deep_learning.png', bbox_inches='tight', dpi=300)

      

print("\nTest Results:")
report_deep_learning=evaluate(trained_model, test_loader)
print(report_deep_learning)
with open("Analysis/classification_report_test_deep_learning.txt", "w") as f:
    f.write(report_deep_learning)


