import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve,
                             precision_score, recall_score, f1_score)
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
import psutil

# =============================================================================
# 1. Load and Prepare Data

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load training and test datasets
training_data = pd.read_csv("Filtered Training Dataset.csv")
test_data = pd.read_csv("Labelled Testing Dataset.csv")

print("\nData Loaded")

# Remove "sha256" if present and align test columns to training
training_data = training_data.drop(columns=["sha256"], errors="ignore")
test_data = test_data.drop(columns=["sha256"], errors="ignore")
test_data = test_data.reindex(columns=training_data.columns, fill_value=0)

# Separate out labels from the training data and remove them from features
if 'label' in training_data.columns:
    y_all = training_data['label'].values
    training_data = training_data.drop(columns=['label'])
if 'label' in test_data.columns:
    y_test = test_data['label'].values
    test_data = test_data.drop(columns=['label'])

# Replace missing values
training_data = training_data.fillna(0)
test_data = test_data.fillna(0)

# Split the original training data into a new training set and a validation set
train_data, validation_set, y_train, y_validation = train_test_split(
    training_data, y_all, test_size=0.15, random_state=RANDOM_SEED
)

# Perform feature selection on the training set and apply the same transformation to validation and test sets
variance_threshold = 1e-3  # Adjust threshold if needed
selector = VarianceThreshold(threshold=variance_threshold)
train_data_selected = pd.DataFrame(selector.fit_transform(train_data),
                                   columns=train_data.columns[selector.get_support()])
validation_set_selected = pd.DataFrame(selector.transform(validation_set),
                                       columns=train_data.columns[selector.get_support()])
test_data_selected = pd.DataFrame(selector.transform(test_data),
                                  columns=train_data.columns[selector.get_support()])

# Standardize the data (zero mean, unit variance)
scaler = StandardScaler()
train_array = scaler.fit_transform(train_data_selected.values)
validation_array = scaler.transform(validation_set_selected.values)
test_array = scaler.transform(test_data_selected.values)

print(f"\nData Standardised using {scaler}")
# Convert arrays to PyTorch tensors

training_tensor = torch.tensor(train_array, dtype=torch.float32)
validation_tensor = torch.tensor(validation_array, dtype=torch.float32)
test_tensor = torch.tensor(test_array, dtype=torch.float32)

# =============================================================================
# 2. Define the Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(negative_slope=0.1),  # Activation Input
            nn.Linear(64, encoding_dim),
            nn.LeakyReLU(negative_slope=0.1)   # Activation Output
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = training_tensor.shape[1]
encoding_dim = 32
autoencoder = Autoencoder(input_dim, encoding_dim)

print("\nAutoencoder Defined")
# =============================================================================
# 3. Baseline Evaluation: Initial SVC (Before Autoencoder Training)

print("\nSVC Baseline Obtained")

autoencoder.eval()
initial_latent_start = time.time()
with torch.no_grad():
    latent_training_initial = autoencoder.encoder(training_tensor).numpy()
    latent_test_initial = autoencoder.encoder(test_tensor).numpy()
initial_latent_time = time.time() - initial_latent_start

# Standardize the initial latent features
latent_scaler_initial = StandardScaler()
latent_training_initial_norm = latent_scaler_initial.fit_transform(latent_training_initial)
latent_test_initial_norm = latent_scaler_initial.transform(latent_test_initial)

# Train an SVC on the initial latent features with timing
svc_initial = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
initial_svc_train_start = time.time()
svc_initial.fit(latent_training_initial_norm, y_train)
initial_svc_train_time = time.time() - initial_svc_train_start

initial_svc_test_start = time.time()
y_pred_initial_test = svc_initial.predict(latent_test_initial_norm)
initial_svc_test_time = time.time() - initial_svc_test_start

# Compute baseline predictions and accuracies (Test)
acc_initial_train = accuracy_score(y_train, svc_initial.predict(latent_training_initial_norm))
acc_initial_test = accuracy_score(y_test, y_pred_initial_test)
initial_precision = precision_score(y_test, y_pred_initial_test)
initial_recall = recall_score(y_test, y_pred_initial_test)
initial_f1 = f1_score(y_test, y_pred_initial_test)
initial_auc = roc_auc_score(y_test, svc_initial.predict_proba(latent_test_initial_norm)[:, 1])
initial_cm = confusion_matrix(y_test, y_pred_initial_test)
TN_i, FP_i, FN_i, TP_i = initial_cm.ravel()
initial_FPR = FP_i / (FP_i + TN_i) if (FP_i + TN_i) > 0 else 0
initial_FNR = FN_i / (FN_i + TP_i) if (FN_i + TP_i) > 0 else 0

# --- Prepare plots for Initial SVC Metrics on Test Data ---
# Confusion Matrix
fig_initial_cm = plt.figure(figsize=(6, 5))
plt.imshow(initial_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Initial Confusion Matrix")
# plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign (0)', 'Malware (1)'])
plt.yticks(tick_marks, ['Benign (0)', 'Malware (1)'])
thresh = initial_cm.max() / 2.
for i in range(initial_cm.shape[0]):
    for j in range(initial_cm.shape[1]):
        plt.text(j, i, format(initial_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if initial_cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# ROC Curve
fig_initial_roc = plt.figure(figsize=(8, 6))
fpr_initial, tpr_initial, thresholds_initial = roc_curve(y_test, svc_initial.predict_proba(latent_test_initial_norm)[:, 1])
plt.plot(fpr_initial, tpr_initial, label=f'Initial AUC-ROC ({initial_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Initial Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)

# =============================================================================
# 4. Train the Autoencoder and Evaluate SVC Accuracy per Epoch
#    (Calculating both loss and SVC accuracy on training and validation sets)

criterion = nn.MSELoss()
learning_rate = 1e-5
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
num_epochs = 20
batch_size = 32
train_loader = torch.utils.data.DataLoader(training_tensor, batch_size=batch_size, shuffle=True)

print(f"\nAutoencoder Training with {num_epochs} Epochs\n")

train_loss_history = []
val_loss_history = []       # using validation loss instead of test loss
training_accuracy_epoch_history = []
validation_accuracy_epoch_history = []  # using validation accuracy

# --- Measure Autoencoder Training Time ---
ae_train_start = time.time()

for epoch in range(num_epochs):
    autoencoder.train()
    for batch in train_loader:
        optimizer.zero_grad()
        _, decoded = autoencoder(batch)
        loss = criterion(decoded, batch)
        loss.backward()
        optimizer.step()
        
    # Evaluate reconstruction loss on full training and validation sets
    autoencoder.eval()
    with torch.no_grad():
        _, full_train_decoded = autoencoder(training_tensor)
        full_train_loss = criterion(full_train_decoded, training_tensor).item()
        _, full_val_decoded = autoencoder(validation_tensor)
        full_val_loss = criterion(full_val_decoded, validation_tensor).item()
    
    train_loss_history.append(full_train_loss)
    val_loss_history.append(full_val_loss)
    
    # Extract latent features for the current epoch using training and validation sets
    with torch.no_grad():
        latent_training = autoencoder.encoder(training_tensor).numpy()
        latent_validation = autoencoder.encoder(validation_tensor).numpy()
    
    # Standardize latent features
    latent_scaler = StandardScaler()
    latent_training_norm = latent_scaler.fit_transform(latent_training)
    latent_validation_norm = latent_scaler.transform(latent_validation)
    
    # Train an SVC on the current latent features
    svc_epoch = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    svc_epoch.fit(latent_training_norm, y_train)
    
    # Compute predictions and accuracies for training and validation sets
    y_pred_train_epoch = svc_epoch.predict(latent_training_norm)
    acc_train_epoch = accuracy_score(y_train, y_pred_train_epoch)
    y_pred_val_epoch = svc_epoch.predict(latent_validation_norm)
    acc_val_epoch = accuracy_score(y_validation, y_pred_val_epoch)
    
    training_accuracy_epoch_history.append(acc_train_epoch)
    validation_accuracy_epoch_history.append(acc_val_epoch)
    
    # Live print per epoch
    print(f"[{epoch+1}] Training Loss: {full_train_loss:.4f}, Validation Loss: {full_val_loss:.4f}"
          # f", SVC Training Acc: {acc_train_epoch:.2f}, SVC Validation Acc: {acc_val_epoch:.2f}"
          )

ae_train_time = time.time() - ae_train_start

# Include the baseline from initial SVC as epoch 0
epochs_range = list(range(0, num_epochs + 1))
baseline_train_acc = acc_initial_train
baseline_val_acc = 0  # no baseline for validation in Section 3
training_accuracy_history_with_baseline = [baseline_train_acc] + training_accuracy_epoch_history
validation_accuracy_history_with_baseline = validation_accuracy_epoch_history  # only epochs

# --- Prepare plot: SVC Accuracy vs. Epochs (Validation used for loss & accuracy) ---
epochs_range = list(range(1, num_epochs + 1))
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, training_accuracy_epoch_history, label='Training Accuracy', color='blue')
plt.plot(epochs_range, validation_accuracy_epoch_history, label='Validation Accuracy', color='orange')
plt.title("SVC Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend(loc="center right")
plt.grid(False)


# --- Prepare plot: Autoencoder Loss vs. Epochs ---
fig_loss = plt.figure(figsize=(8, 6))
epochs_range_train = range(1, num_epochs + 1)
plt.plot(epochs_range_train, train_loss_history, label='Training Loss', color='blue')
plt.plot(epochs_range_train, val_loss_history, label='Validation Loss', color='orange')
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend(loc="center right")
plt.grid(False)

print("\nPlotting Graphs")
# =============================================================================
# 5. Final Evaluation of the SVC Classifier on the Latent Features (Test Set)

print("\nRunning Final SVC")

autoencoder.eval()
ae_test_start = time.time()
with torch.no_grad():
    final_latent_training = autoencoder.encoder(training_tensor).numpy()
    final_latent_test = autoencoder.encoder(test_tensor).numpy()
ae_test_time = time.time() - ae_test_start

# Standardize the final latent features
latent_scaler_final = StandardScaler()
final_latent_training_norm = latent_scaler_final.fit_transform(final_latent_training)
final_latent_test_norm = latent_scaler_final.transform(final_latent_test)

# --- Measure training time for final SVC ---
start_train_time = time.time()
svc_final = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
svc_final.fit(final_latent_training_norm, y_train)
svc_train_time = time.time() - start_train_time

# --- Measure testing time for final SVC ---
start_test_time = time.time()
y_pred_final = svc_final.predict(final_latent_test_norm)
y_prob_final = svc_final.predict_proba(final_latent_test_norm)[:, 1]
svc_test_time = time.time() - start_test_time

# Compute final SVC metrics (Testing)
final_test_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_auc = roc_auc_score(y_test, y_prob_final)
final_cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = final_cm.ravel()
final_FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
final_FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

# Compute final SVC training accuracy (on training set)
y_pred_final_train = svc_final.predict(final_latent_training_norm)
final_train_accuracy = accuracy_score(y_train, y_pred_final_train)

print("\nPlotting Final Results")

# --- Prepare plots for Final SVC Metrics ---

# Final Confusion Matrix
fig_final_cm = plt.figure(figsize=(6, 5))
plt.imshow(final_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Hybrid Model Confusion Matrix")
# plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign (0)', 'Malware (1)'])
plt.yticks(tick_marks, ['Benign (0)', 'Malware (1)'])
thresh = final_cm.max() / 2.
for i in range(final_cm.shape[0]):
    for j in range(final_cm.shape[1]):
        plt.text(j, i, format(final_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if final_cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Final ROC Curve
fig_final_roc = plt.figure(figsize=(8, 6))
fpr_final, tpr_final, thresholds_final = roc_curve(y_test, y_prob_final)
plt.plot(fpr_final, tpr_final, label=f'AUC-ROC ({final_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hybrid Model Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)

# =============================================================================
# 6. Compute Process Memory Usage and Total Times

process = psutil.Process()
process_memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB

total_train_time = ae_train_time + svc_train_time
total_test_time = ae_test_time + svc_test_time

# =============================================================================
# 7. Final Print Summary (All prints occur here)

print("\n=== Initial SVC Metrics ===")
print(f"Training Accuracy: {acc_initial_train}")
print(f"Testing Accuracy: {acc_initial_test}")
print(f"Precision: {initial_precision}")
print(f"Recall: {initial_recall}")
print(f"F1 Score: {initial_f1}")
print(f"AUC: {initial_auc}")
print(f"FPR: {initial_FPR}")
print(f"FNR: {initial_FNR}")
print(f"SVC Training Time (Initial): {initial_svc_train_time} seconds")
print(f"SVC Testing Time (Initial): {initial_svc_test_time} seconds")
# print(f"Latent Feature Extraction Time (Initial): {initial_latent_time:.2f} seconds")

print("\n=== Final Model Metrics ===")
print(f"Training Accuracy: {final_train_accuracy}")
print(f"Testing Accuracy: {final_test_accuracy}")
print(f"Precision: {final_precision}")
print(f"Recall: {final_recall}")
print(f"F1 Score: {final_f1}")
print(f"AUC: {final_auc}")
print(f"FPR: {final_FPR}")
print(f"FNR: {final_FNR}")
print(f"SVC Training Time (Final): {svc_train_time} seconds")
print(f"SVC Testing Time (Final): {svc_test_time} seconds")
print(f"Autoencoder Training Time: {ae_train_time} seconds")
print(f"Total Training Time: {total_train_time} seconds")
print(f"Total Testing Time: {total_test_time} seconds")
# print(f"Autoencoder Inference Time (Final): {ae_test_time:.2f} seconds")

print("\nModel and Training Configuration:")
print(f"Input_dim: {input_dim}")
print(f"Output_dim: {input_dim}")  # Autoencoder output dimension equals input dimension
print(f"Input_length: {training_tensor.shape[0]}")
print(f"Learning rate: {learning_rate}")
print(f"Dense Unit: {64}")  # Number of units in the dense layers
print("Activation Input: LeakyReLU (negative_slope=0.1)")
print("Activation Output: LeakyReLU (negative_slope=0.1)")
print("Optimizer: Adam")
print(f"Process Memory Usage: {process_memory_usage} MB")

# =============================================================================
# 8. Show All Plots at the End
plt.show()
