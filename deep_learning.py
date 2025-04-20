import torch.nn as nn


def create_clinical_neuralnet(input_size, num_classes):

    '''
    Function that constructs a neural network for clinical dataset multi-class classification with:
    two hidden layers (128 → 64 units), ReLU activation function, batch normalization, and regularization

    Inputs:
        input_size: int - # of input features 
        num_classes: int - # of output classes for classification


    Returns:
        model: linear neural net from pytorch 
    '''

    model= nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        
        nn.Linear(64, num_classes)
    )

    return model

