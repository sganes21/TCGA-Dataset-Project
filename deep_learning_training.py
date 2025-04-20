import torch
from sklearn.metrics import classification_report


def extract_matrix_labels(model, data_loader, device=None):
    """
    Extracts predictions and true labels from a model and dataloader that can be used to compute a confusion matrix.
        
    Inputs:
        model: neural net model 
        data loader: associated batch information from dataset
    
    Returns:
        all_preds: list of predicted classes
        all_labels: list of true classes
    """
    model.eval()
    all_preds = []
    all_labels = []
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


def evaluate(model, loader):
    """
    Extracts predictions and true labels from a model and dataloader that can be used to compute a confusion matrix for traditional algorithm.
        
    Inputs:
        model: ml model 
        data loader: associated batch information from dataset
    
    Returns:
        all_preds: list of predicted classes
        all_labels: list of true classes
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return classification_report(all_labels, all_preds, zero_division=0)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs):
    """
    Extracts predictions and true labels from a model and dataloader that can be used to compute a confusion matrix for traditional algorithm.
        
    Inputs:
        model: The neural network model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler
        criterion: Loss function to optimize
        num_epochs (int): Number of epochs to train the model
    
    Returns:
        model: trained model with incorporated parameters.
      
    """
        
    best_loss = float('inf')
    patience = 5
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Same forward pass
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping logic remains identical
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model

