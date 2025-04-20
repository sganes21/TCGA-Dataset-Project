#
# Feature Extraction
#
import numpy as np
import pandas as pd


def clean_column(col):

    '''
    Function to clean columns of a clinical dataset.

    Inputs:
        col: input column from clinical dataset


    Returns:
        col: Column scrubbed of unknown values and changing columns to numeric type
    '''

    # Convert unknown values to NaN
    col = col.replace(['not reported', 'Not Reported', '--', ''], np.nan)
    
    # Convert numeric columns
    if col.dtype == 'object':
        try:
            col = pd.to_numeric(col, errors='ignore')
        except:
            pass
    return col

def feature_extraction(data_df):
     
    """
    Extracts features for specified deintified patient information.

    This function extracts a broad feature sets for a case id representing a deidentified study participant. 
    The extracted features are used to to develop a model to generate 
    predictions on which subjects will be diagnosed from a set of conditions.
    Please note: Not all provided data is fed into the model.

    Parameters:
        data_df: Raw clinical dataset, were rows are sorted by unique case id


    Returns:
        clean_df: A list containing the extracted features for a case id.
    """
        
    # Initialize features list
    
    features = [
    
    # Clinical features
    'cases.consent_type',
    'cases.days_to_consent',
    'diagnoses.age_at_diagnosis',
    'cases.primary_site',
    'demographic.country_of_residence_at_enrollment',
    'demographic.days_to_death',
    'demographic.gender',
    'demographic.race',
    'demographic.vital_status',
    'diagnoses.ajcc_clinical_m',
    'diagnoses.ajcc_clinical_n',
    'diagnoses.ajcc_clinical_stage',
    'diagnoses.ajcc_clinical_t',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.ajcc_staging_system_edition',
    'diagnoses.classification_of_tumor',
    'diagnoses.morphology',
    'diagnoses.site_of_resection_or_biopsy',
    'diagnoses.tissue_or_organ_of_origin',
    'diagnoses.tumor_grade',
    'cases.disease_type',
    
    # Follow-up features
    'follow_ups.bmi',
    'follow_ups.weight'
    ]

    # Filter and clean
    df = data_df[features].copy()
    clean_df = df.apply(clean_column)



    # Specific handling for key columns
    ## Days to death: alive patients get np.nan
    clean_df['demographic.days_to_death'] = clean_df['demographic.days_to_death'].where(
    clean_df['demographic.vital_status'] == 'Dead', np.nan)

    ## Convert vital status to binary
    clean_df['demographic.vital_status'] = (clean_df['demographic.vital_status'] == 'Dead').astype(float)

    ## Handle AJCC language / Removing wording in prefixes
    ajcc_columns = [c for c in clean_df.columns if 'ajcc' in c]
    for col in ajcc_columns:
        # Extract numbers 
        if 'stage' in col.lower():
            clean_df[col] = clean_df[col].str.extract('(\d+)').astype(float)
        
    ## Imputing missing values & adding median for missing values
    numerical_cols = clean_df.select_dtypes(include=np.number).columns
    categorical_cols = clean_df.select_dtypes(exclude=np.number).columns
    clean_df[numerical_cols] = clean_df[numerical_cols].fillna(clean_df[numerical_cols].median())

    ## Handling unknown values
    for col in categorical_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0] if not clean_df[col].mode().empty else 'Unknown')
    
    # Finalizing Dataframe
    clean_df = clean_df.dropna(axis=1, how='all')  # Remove columns with all missing values
    clean_df = df.drop_duplicates()

    # Removing rare class members (less than 21 samples)
    class_counts = clean_df['cases.disease_type'].value_counts()
    valid_classes = class_counts[class_counts >= 21].index
    clean_df = clean_df[clean_df['cases.disease_type'].isin(valid_classes)]
    print("Printing Cleaned Member Classes within the Function:")
    testing = clean_df['cases.disease_type']
    print(testing.value_counts())

    return clean_df