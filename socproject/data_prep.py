import numpy as np
import torch
def prepare_data_with_delay(CDR, male_death, female_death, infant_death, sequence_length, delay=3):
    """
    Prepare data with consideration for time delay effects
    """
    data = np.column_stack((CDR, male_death, female_death, infant_death))
    X, y = [], []
    
    for i in range(len(data) - sequence_length - delay):
        # Input sequence
        seq = data[i:i + sequence_length]
        # Target is 'delay' steps ahead
        target = data[i + sequence_length + delay]
        X.append(seq)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Split into train and validation sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val