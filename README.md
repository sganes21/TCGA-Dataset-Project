![Duke AIPI Logo](https://storage.googleapis.com/aipi_datasets/Duke-AIPI-Logo.png)

# TCGA Dataset Project

[TCGA Reference](https://portal.gdc.cancer.gov/)

Publicly available online information was included in the dataset from the National Cancer Institute Database. The information contains de-identified patient data from a population subset presenting with various forms of lung cancer. The dataset contains 5,334 total cases.

Problem Statement: This project aims to develop a model using TCGA data to predict patient diagnosis of various types of lung cancer, comparing the performance of a non-deep learning model with a deep learning model (e.g., Linear Neural Network). Accurate cancer prediction is integral in survival rates and may be useful in terms of allocating resources and advocating for aggressive versus passive treatment options.


## Description of Scripts:

### Traditional ML Pipeline (random forest)

This dataset contains the following scripts to support the traditional ml pipeline:

feature_generation.py: 
    A script containing functions to extract a broad feature sets for a given case id, cleaning 
    and thresholding that information.
     
random_forest_model_train.py:
    A function that performs K-Fold cross-validation with a Random Forest Classifier
    that returns a pipeline and an optimized set of parameters. 
   
hyperparameter_optimization.py:
    A function utlizing the Optuna library to optimize a Random Forest Classifier model.

### Deep Learning Pipeline (Linear NN)

deep_learning.py: 
    A function that constructs a neural network for clinical dataset multi-class classification with
    two hidden layers, ReLU activation function, batch normalization, and regularization
    
deep_learning_training.py:
    A script that contains functions to train a neural net model along with corresponding arguments 
    and extracts predictions and true labels from a model and dataloader that can be used to compute a confusion matrix for traditional algorithm.. 
   
data_preprocessor.py:
    A function to preprocess tabular data and labels for machine learning, including encoding, compensating for class imbalance, and conversion to tensors.Converts target label from string to numeric for SMOTE processing. 

data_loader.py:
    A function that creates dataloaders for training, validation, and test sets, and computes class weights
    for handling class imbalance during training. Training utlizes WeightedRandomSampler to ensure equalized distribution of classes based on training labels. 
### Main Script

evaluation.py
    Main script that loads data, executes scripts, outputs prediction and generates analysis. 

## Instructions to Run Script

To set up the development environment: 

For Windows:

1) Activate Virtual Environment
   ```
   python  -m venv venv
   .venv .venv/Scripts/activate 
   ```
    
2) Install the requirements.txt file
    ``` 
    python -m pip install -r requirements.txt 
    ```

3) Execute Model Pipeline
    ``` 
    python evaluation.py 
    ```

## Evaluation

    # Figure 1 Confusion Matrix - Validation Set - Traditional ML Model
![Figure 1](Analysis/confusion_matrix_validation_ml.png)

    # Figure 2 Classification Report - Validation Set - Traditional ML Model
![Figure 2](Analysis/classification_report_val_ml.txt)

    # Figure 3 Classification Report - Test Set - Traditional ML Model
![Figure 3](Analysis/classification_report_test_ml.txt)

    # Figure 4 Classification Report - Test Set - Deep Learning Model
![Figure 4](Analysis/confusion_matrix_validation_deep_learning.png)

    # Figure 5 Classification Report - Validation Set - Deep Learning Model
![Figure 5](Analysis/classification_report_val_deep_learning.txt)

    # Figure 6 Classification Report - Test Set - Deep Learning Model
![Figure 6](Analysis/classification_report_test_deep_learning.txt)