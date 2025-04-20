import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler 

def load_data(y_train_res, train_dataset, val_dataset, test_dataset):
    """
    Function that creates dataloaders for training, validation, and test sets, and computes class weights
    for handling class imbalance during training. Training utlizes WeightedRandomSampler to ensure equalized distribution of classes 
    based on training labels. 

    Inputs:
        y_train_res: array of upsampled training labels.
        train_dataset: Tensor dataset containing the training data and labels.
        val_dataset: Tensor dataset containing the validation data and labels.
        test_dataset TEnsor dataset containing the test data and labels.

    Returns:
        train_loader: Data loader for the training set with weighted sampling.
        val_loader: Data loader for the validation set.
        test_loader: Data loader for the test set.
        class_weights: array of class weights.
    """
    class_counts = np.bincount(y_train_res)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train_res]

    sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
    )

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, class_weights