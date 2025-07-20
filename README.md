# Ambient Temperature Anomaly Detection using Autoencoder

This project applies a deep learning-based autoencoder to detect anomalies in ambient temperature time-series data. It uses a synthetic anomaly injection approach for demonstration purposes.

---

## Dataset

- **File:** `ambient_temperature_system_failure.csv`
- **Columns:**
  - `timestamp`: Time at which the temperature was recorded.
  - `value`: The ambient temperature reading.

---

## Objective

To build an unsupervised anomaly detection system using an autoencoder that learns to reconstruct normal temperature data. Anomalies are identified when reconstruction error exceeds a certain threshold.

---

## Dependencies

Install the required packages before running the project:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras
