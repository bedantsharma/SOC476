import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('some_data.csv',index_col=0)
data = data.T
print(data.head())

data.fillna(method='ffill',inplace=True)


years = np.array(data.index)
years = [int(i) for i in years]
years = np.array(years)
CDR = np.array(data["CDR"])
male_death = np.array(data["male_mortality"])
female_death = np.array(data["female_mortality"])
infant_death = np.array(data["infant_mortality"])

## normalize the data 
CDR = (CDR - np.min(CDR))/(np.max(CDR) - np.min(CDR))
male_death = (male_death - np.min(male_death))/(np.max(male_death) - np.min(male_death))
female_death = (female_death - np.min(female_death))/(np.max(female_death) - np.min(female_death))
infant_death = (infant_death - np.min(infant_death))/(np.max(infant_death) - np.min(infant_death))

# plt.plot(CDR,label='CDR')
# plt.plot(male_death,label='male_mortality')
# plt.plot(female_death,label='female_mortality')
# plt.plot(infant_death,label='infant_mortality')
# plt.grid(visible=True)
# plt.legend()
# plt.show()



class MortalityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MortalityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Parameters
sequence_length = 10
hidden_size = 64
num_layers = 2
learning_rate = 0.01
num_epochs = 100
input_size = 4  # CDR, male_mortality, female_mortality, infant_mortality
output_size = 4

# Prepare the data
def prepare_data(CDR, male_death, female_death, infant_death, sequence_length):
    # Combine all features
    data = np.column_stack((CDR, male_death, female_death, infant_death))
    
    # Create sequences
    X, y = create_sequences(data, sequence_length)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Split into train and validation sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val

def train_model(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

def forecast_future(model, last_sequence, num_steps=5):
    model.eval()
    future_predictions = []
    current_sequence = last_sequence.clone()
    
    for _ in range(num_steps):
        with torch.no_grad():
            prediction = model(current_sequence.unsqueeze(0))
            future_predictions.append(prediction.numpy())
            
            # Update sequence for next prediction
            current_sequence = torch.cat((current_sequence[1:], prediction), 0)
    
    return np.array(future_predictions)

# Usage example:

# Prepare the data
X_train, y_train, X_val, y_val = prepare_data(CDR, male_death, female_death, infant_death, sequence_length)

# Initialize and train the model
model = MortalityLSTM(input_size, hidden_size, num_layers, output_size)
train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate)

# Make future predictions
last_sequence = X_val[-1]  # Use the last sequence from validation set
future_predictions = forecast_future(model, last_sequence, num_steps=5)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions
metrics = ['CDR', 'Male Mortality', 'Female Mortality', 'Infant Mortality']
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(range(len(X_val)), X_val[:, -1, i], label='Actual')
    plt.plot(range(len(X_val), len(X_val) + 5), future_predictions[:, i], label='Predicted')
    plt.title(f'{metrics[i]} Forecast')
    plt.legend()
plt.tight_layout()
plt.show()
