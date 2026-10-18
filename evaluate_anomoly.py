import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.metrics import classification_report

# --- 1. Define the Model Architecture (Must match training exactly) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. Load the saved Model and Data ---
model = joblib.load('autoencoder_model.pkl')
scaler = joblib.load('scaler.pkl')
model.eval()

print("Loading test data...")
# (Assuming you still have KDDTest+.txt)
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
             "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
             "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
             "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

test_df = pd.read_csv("KDDTest+.txt", names=col_names)
X_test = test_df.drop(columns=['label', 'difficulty'])
y_test = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical variables (using the same logic as training)
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col]) # Simplified for evaluation

# Scale the data using the SAVED scaler
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# --- 3. Calculate Reconstruction Errors ---
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # Calculate MSE for each row (this is our anomaly score)
    mse_loss = torch.mean((test_outputs - X_test_tensor)**2, dim=1).numpy()

# --- 4. The Fix: Dynamic Thresholding ---
# Let's test a few thresholds to see which one catches the most attacks
for threshold in [0.01, 0.02, 0.05, 0.1]:
    predictions = [1 if loss > threshold else 0 for loss in mse_loss]
    
    print(f"\n--- Results for Threshold: {threshold} ---")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Attack'], zero_division=0))