#
# Evaluation
#
import pandas as pd
import numpy as np
from feature_generation import feature_extraction
from random_forest_model_train import train_random_forest

# Load data
clinical_df = pd.read_csv('clinical.tsv', sep='\t')
follow_up_df= pd.read_csv('follow_up.tsv', sep='\t')
train_df = pd.read_csv('train.csv')
tickets_df = pd.read_csv('tickets_all.csv')
subscriptions_df = pd.read_csv('subscriptions.csv')
account_df = pd.read_csv('account.csv', encoding='latin1')

# Extract features from data
features_list = []
for patient_id in train_df['cases.case_id']:
    features = feature_extraction(patient_id, clinical_df,follow_up_df)
    features_list.append(features)


# Add features from data into pandas df and visualize 
features_df = pd.DataFrame(features_list)
print(features_df.head(5))


# Split the data into features and target (drop account ids from feature extraction)
X = features_df.drop('account_id', axis=1)
y = train_df['label']



# Train on the data and generate the model
target = train_df['label']
model, best_params = train_random_forest(X, y)
print("Best Parameters:", best_params)

# Load Test
test_df = pd.read_csv('test.csv')

# Extract features from test data
features_list_test = []
for Id in test_df['ID']:
    features_test = feature_extraction(Id, subscriptions_df,tickets_df,account_df)
    features_list_test.append(features_test)

# Add features from test data into pandas df and visualize 
features_df_test = pd.DataFrame(features_list_test)
print(features_df_test.head(5))

# Split the test data into features and target (drop account ids from feature extraction)
X_test = features_df_test.drop('account_id', axis=1)


# Generate predictions on model applied to test data
predictions = model.predict_proba(X_test)
print("Printing predictions")
print(predictions)

# Create a submission DataFrame
kaggle_submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Predicted': predictions[:,1]
})

# Save to submission.csv
kaggle_submission_df.to_csv('submission.csv', index=False)
print("Predictions saved to kaggle submission.csv")
