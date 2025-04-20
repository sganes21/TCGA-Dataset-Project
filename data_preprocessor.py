import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def data_processing(data, labels):

    # Identify categorical and numerical columns
    string_cols = data.select_dtypes(include=['object', 'category']).columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), string_cols)
    ])

    X_processed = preprocessor.fit_transform(data)

    # Convert string labels to numerical indices
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    X_temp, X_test, y_temp, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=0
)
    X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=0
)

    sm = SMOTE(random_state=0)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    X_train_res = X_train_res.toarray()  
    X_val = X_val.toarray()              
    X_test = X_test.toarray()            

    # 3. Convert to PyTorch Tensors
    train_dataset = TensorDataset(
    torch.FloatTensor(X_train_res), 
    torch.LongTensor(y_train_res)
    )
    val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
    )

    return X_train_res, y_train_res, train_dataset, val_dataset, test_dataset



  