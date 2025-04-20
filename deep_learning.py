import torch.nn as nn


def create_clinical_neuralnet(input_size, num_classes):
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

