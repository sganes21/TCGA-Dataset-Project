import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler 

def load_data(y_train_res, train_dataset, val_dataset, test_dataset):
    """
    

    Inputs:
        y_train_res: Upsampled labelled dataset
        train_dataset:
        val_dataset:
        test_dataset: 

    Returns:
        train_loader: 
        val_loader:
        test_loader:
        class_weights 
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