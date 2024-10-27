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



class EnhancedMortalityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(EnhancedMortalityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Additional layers for better feature extraction
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size // 2, output_size)
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm1(x, (h0, c0))
        
        # Apply attention
        attn_output = self.attention_net(lstm_out)
        
        # Feature extraction
        features = self.feature_layers(attn_output)
        
        # Final output
        output = self.fc(features)
        return output

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

def train_enhanced_model(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

# Usage example:
sequence_length = 10
hidden_size = 128  # Increased hidden size
num_layers = 3     # Increased number of layers
learning_rate = 0.001
num_epochs = 1000  # Increased number of epochs
input_size = 4
output_size = 4
delay = 3  # Time delay parameter

# Prepare the data with delay consideration
X_train, y_train, X_val, y_val = prepare_data_with_delay(
    CDR, male_death, female_death, infant_death, 
    sequence_length, delay
)

# Initialize and train the enhanced model
model = EnhancedMortalityLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout_rate=0.2
)

train_losses, val_losses = train_enhanced_model(
    model, X_train, y_train, X_val, y_val, 
    num_epochs, learning_rate
)


# Denormalization functions
def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Retrieve original min and max values for denormalization
CDR_min, CDR_max = np.min(data["CDR"]), np.max(data["CDR"])
male_death_min, male_death_max = np.min(data["male_mortality"]), np.max(data["male_mortality"])
female_death_min, female_death_max = np.min(data["female_mortality"]), np.max(data["female_mortality"])
infant_death_min, infant_death_max = np.min(data["infant_mortality"]), np.max(data["infant_mortality"])

# Prediction and plotting
model.eval()
with torch.no_grad():
    # Prepare last sequence for prediction
    last_seq = torch.FloatTensor(data.iloc[-sequence_length:].values).unsqueeze(0)
    predictions = []

    # Predict for the next 5 years (or any desired number of years)
    num_years_to_predict = 5
    for _ in range(num_years_to_predict):
        pred = model(last_seq).cpu().numpy()
        predictions.append(pred)
        last_seq = torch.cat((last_seq[:, 1:], torch.FloatTensor(pred).unsqueeze(0)), dim=1)

    # Convert predictions list to array
    predictions = np.array(predictions).squeeze()

# Denormalize data for original and predictions
CDR_denorm = denormalize(CDR, CDR_min, CDR_max)
male_death_denorm = denormalize(male_death, male_death_min, male_death_max)
female_death_denorm = denormalize(female_death, female_death_min, female_death_max)
infant_death_denorm = denormalize(infant_death, infant_death_min, infant_death_max)

pred_CDR = denormalize(predictions[:, 0], CDR_min, CDR_max)
pred_male_death = denormalize(predictions[:, 1], male_death_min, male_death_max)
pred_female_death = denormalize(predictions[:, 2], female_death_min, female_death_max)
pred_infant_death = denormalize(predictions[:, 3], infant_death_min, infant_death_max)

# Generate years for predictions
future_years = np.arange(years[-1] + 1, years[-1] + 1 + num_years_to_predict)

# Import necessary libraries for plotting

# Create the plot with the modifications
plt.figure(figsize=(14, 8))

# Define colors for consistency
colors = {'CDR': 'blue', 'Male Mortality': 'orange', 'Female Mortality': 'green', 'Infant Mortality': 'red'}

# Plot historical and forecast data with shaded future
plt.plot(years, CDR_denorm, label='CDR (Actual)', color=colors['CDR'])
plt.plot(np.concatenate([years[-1:], future_years]), np.concatenate([CDR_denorm[-1:], pred_CDR]),
         linestyle='--', label='CDR (Forecast)', color=colors['CDR'])
plt.fill_between(future_years, pred_CDR, color=colors['CDR'], alpha=0.1)

plt.plot(years, male_death_denorm, label='Male Mortality (Actual)', color=colors['Male Mortality'])
plt.plot(np.concatenate([years[-1:], future_years]), np.concatenate([male_death_denorm[-1:], pred_male_death]),
         linestyle='--', label='Male Mortality (Forecast)', color=colors['Male Mortality'])
plt.fill_between(future_years, pred_male_death, color=colors['Male Mortality'], alpha=0.1)

plt.plot(years, female_death_denorm, label='Female Mortality (Actual)', color=colors['Female Mortality'])
plt.plot(np.concatenate([years[-1:], future_years]), np.concatenate([female_death_denorm[-1:], pred_female_death]),
         linestyle='--', label='Female Mortality (Forecast)', color=colors['Female Mortality'])
plt.fill_between(future_years, pred_female_death, color=colors['Female Mortality'], alpha=0.1)

plt.plot(years, infant_death_denorm, label='Infant Mortality (Actual)', color=colors['Infant Mortality'])
plt.plot(np.concatenate([years[-1:], future_years]), np.concatenate([infant_death_denorm[-1:], pred_infant_death]),
         linestyle='--', label='Infant Mortality (Forecast)', color=colors['Infant Mortality'])
plt.fill_between(future_years, pred_infant_death, color=colors['Infant Mortality'], alpha=0.1)

# Customize the legend and labels
plt.xlabel('Year')
plt.ylabel('Mortality')
plt.title('Mortality Forecast')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Moves the legend outside the plot
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate the legend

# Show plot
plt.show()
