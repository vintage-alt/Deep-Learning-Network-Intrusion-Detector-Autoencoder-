import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1. Load and Preprocess Data ---
print("Loading data...")
# The NSL-KDD dataset doesn't have headers, so we define them
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
             "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
             "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
             "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

train_df = pd.read_csv("KDDTrain+.txt", names=col_names)
test_df = pd.read_csv("KDDTest+.txt", names=col_names)

# Drop the 'difficulty' column, we don't need it
train_df = train_df.drop(columns=['difficulty'])
test_df = test_df.drop(columns=['difficulty'])

# The 'label' column tells us if it's 'normal' or an attack (e.g., 'neptune', 'smurf')
# We want to train ONLY on normal traffic.
normal_train = train_df[train_df['label'] == 'normal'].copy()

# Drop the label column for training the autoencoder
X_train_normal = normal_train.drop(columns=['label'])

# For testing, we want a mix of normal and attacks
X_test = test_df.drop(columns=['label'])
y_test = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1) # 0 = normal, 1 = attack

# We have categorical text columns (protocol_type, service, flag). Neural nets need numbers.
# Let's use LabelEncoder to turn them into numbers.
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on the combined data so we have the same mapping
    le.fit(pd.concat([X_train_normal[col], X_test[col]]))
    X_train_normal[col] = le.transform(X_train_normal[col])
    X_test[col] = le.transform(X_test[col])

# Scale the data (Neural Nets hate unnormalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# --- 2. Define the Autoencoder Architecture ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder: Compresses the input down to a bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4) # The Bottleneck
        )
        # Decoder: Reconstructs the input from the bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid() # Sigmoid forces output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
input_dim = X_train_tensor.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss() # Mean Squared Error (Reconstruction Error)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Training Loop ---
print("Training Autoencoder...")
epochs = 10
batch_size = 256

for epoch in range(epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        batch = X_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch) # Compare output to the original input!
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --- 4. Evaluation (Anomaly Detection) ---
print("\nEvaluating on Test Data...")
model.eval() # Set to evaluation mode

with torch.no_grad():
    # Get reconstruction errors for the test set
    test_outputs = model(X_test_tensor)
    # Calculate MSE for each row
    mse_loss = torch.mean((test_outputs - X_test_tensor)**2, dim=1).numpy()

# The logic: If the error is high, it's an attack.
# We need to set a threshold. Let's say anything above the 95th percentile of NORMAL training error is an attack.
# (In a real project, you would tune this threshold).
threshold = pd.Series(mse_loss).quantile(0.95) 

# Predict: 1 if error > threshold, else 0
predictions = [1 if loss > threshold else 0 for loss in mse_loss]

# Compare to actual labels
from sklearn.metrics import classification_report
print(f"Anomaly Threshold (Reconstruction Error): {threshold:.4f}")
print(classification_report(y_test, predictions, target_names=['Normal', 'Attack']))

import joblib
joblib.dump(model, 'autoencoder_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved.")