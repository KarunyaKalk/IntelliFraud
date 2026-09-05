import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping

from src.autoencoder import build_autoencoder

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.csv")
TEST_INDEX_PATH = os.path.join(MODEL_DIR, "test_indices.csv")

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
print("\n===== LOADING DATASET =====")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Normal transactions:", (df["Class"] == 0).sum())
print("Fraud transactions:", (df["Class"] == 1).sum())

# ---------------------------------------------------------
# 2. Create formal train/test split
# ---------------------------------------------------------
print("\n===== CREATING TRAIN/TEST SPLIT =====")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Class"]
)

print("Training transactions:", len(train_df))
print("Held-out test transactions:", len(test_df))
print("\nTraining class distribution:")
print(train_df["Class"].value_counts())
print("\nHeld-out test class distribution:")
print(test_df["Class"].value_counts())

# ---------------------------------------------------------
# 3. Use ONLY normal transactions for training
# ---------------------------------------------------------
train_normal = train_df[train_df["Class"] == 0].copy()

X_train = train_normal.drop(columns=["Class"])

# ---------------------------------------------------------
# 4. Scale using training-normal data only
# ---------------------------------------------------------
print("\n===== SCALING DATA =====")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

print("Training features:", X_train_scaled.shape[1])
print("Normal transactions used for Autoencoder training:", len(X_train_scaled))

# ---------------------------------------------------------
# 5. Build Autoencoder
# ---------------------------------------------------------
print("\n===== BUILDING AUTOENCODER =====")

input_dimension = X_train_scaled.shape[1]

model = build_autoencoder(input_dimension)

model.summary()

# ---------------------------------------------------------
# 6. Train Autoencoder
# ---------------------------------------------------------
print("\n===== TRAINING AUTOENCODER =====")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    X_train_scaled,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)

# ---------------------------------------------------------
# 7. Calculate reconstruction errors
#    ONLY on training-normal data
# ---------------------------------------------------------
print("\n===== CALCULATING TRAINING RECONSTRUCTION ERRORS =====")

reconstructed = model.predict(
    X_train_scaled,
    batch_size=512,
    verbose=0
)

reconstruction_errors = np.mean(
    np.square(X_train_scaled - reconstructed),
    axis=1
)

# ---------------------------------------------------------
# 8. Select anomaly threshold
#    ONLY from training-normal errors
# ---------------------------------------------------------
threshold = np.percentile(
    reconstruction_errors,
    95
)

print("95th percentile threshold:", threshold)

# ---------------------------------------------------------
# 9. Save model components
# ---------------------------------------------------------
print("\n===== SAVING MODEL =====")

os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)

joblib.dump(
    scaler,
    SCALER_PATH
)

with open(THRESHOLD_PATH, "w") as file:
    file.write(str(threshold))

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

# Save held-out test indices so evaluation uses
# exactly the same test transactions.
test_df.index.to_series().to_csv(
    TEST_INDEX_PATH,
    index=False,
    header=["index"]
)

print("\n===== TRAINING COMPLETED =====")

print("Model saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
print("Threshold saved:", THRESHOLD_PATH)
print("Training history saved:", HISTORY_PATH)
print("Held-out test indices saved:", TEST_INDEX_PATH)
print("\nFinal threshold:", threshold)