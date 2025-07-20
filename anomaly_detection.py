import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping

# 1. Load Data
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ambient_temperature_system_failure.csv')

# 2. Clean and Prepare Data
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
data = data.dropna()
data = data.sort_values('timestamp')

# 3. Normalize the 'value' column
scaler = MinMaxScaler()
data['scaled_value'] = scaler.fit_transform(data[['value']])

# 4. Train-Test Split
split_index = int(len(data) * 0.7)
train_data = data['scaled_value'].values[:split_index]
test_data = data['scaled_value'].values[split_index:]

train_tensor = tf.convert_to_tensor(train_data.reshape(-1, 1), dtype=tf.float32)

# 5. Build Autoencoder
input_dim = train_tensor.shape[1]
encoding_dim = 8

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 6. Train Autoencoder
autoencoder.fit(train_tensor, train_tensor,
                epochs=50, batch_size=32,
                shuffle=True, callbacks=[early_stop], verbose=1)

# 7. Set Threshold using Training MSE
train_recon = autoencoder.predict(train_tensor)
train_mse = tf.reduce_mean(tf.square(train_tensor - train_recon), axis=1)
threshold = np.quantile(train_mse, 0.99)

# 8. Inject Synthetic Anomalies in Test Data
test_data_with_anomalies = test_data.copy()
np.random.seed(42)
anomaly_indices = np.random.choice(len(test_data_with_anomalies), size=10, replace=False)
test_data_with_anomalies[anomaly_indices] += np.random.uniform(0.5, 1.0, size=10)
test_data_with_anomalies = np.clip(test_data_with_anomalies, 0, 1)  # keep in [0, 1]

test_tensor = tf.convert_to_tensor(test_data_with_anomalies.reshape(-1, 1), dtype=tf.float32)

# 9. Predict and Score
reconstructions = autoencoder.predict(test_tensor)
mse = tf.reduce_mean(tf.square(test_tensor - reconstructions), axis=1)
anomaly_scores = mse.numpy()
anomalous = anomaly_scores > threshold
binary_preds = anomalous.astype(int)

# 10. Create True Labels
true_labels = np.zeros_like(binary_preds)
true_labels[anomaly_indices] = 1

# 11. Evaluate
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, binary_preds, average='binary')

print(f"\n--- Evaluation ---")
print(f"Threshold       : {threshold:.6f}")
print(f"Precision       : {precision:.3f}")
print(f"Recall          : {recall:.3f}")
print(f"F1 Score        : {f1:.3f}")

# # 12. Plot MSE Distribution
# plt.figure(figsize=(10, 4))
# plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Test MSE')
# plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
# plt.title("Reconstruction Error Distribution")
# plt.xlabel("MSE")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 13. Plot Anomalies on Time Series
timestamps = data['timestamp'].values[split_index:]
original = data['value'].values[split_index:]

plt.figure(figsize=(14, 6))
plt.plot(timestamps, original, label='Temperature')
plt.scatter(timestamps[anomalous], original[anomalous], color='red', label='Detected Anomalies', s=25)
# plt.scatter(timestamps[anomaly_indices], original[anomaly_indices], marker='x', color='black', label='True Anomalies', s=60)
plt.xlabel("Timestamp")
plt.ylabel("Temperature (°C)")
plt.title("Anomaly Detection in Ambient Temperature Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
