# Human Activity Recognition Using LSTM Networks 🧠📱

## 📌 Objective

The goal of this project is to build a deep learning model using **Long Short-Term Memory (LSTM)** networks to classify human physical activities such as walking, standing, and laying down. The classification is based on time-series sensor data (accelerometer and gyroscope) collected from smartphones. This project highlights the challenges of sequence modeling and the application of RNNs in wearable tech scenarios.

---

## 📂 Dataset: UCI HAR Dataset

The **UCI Human Activity Recognition (HAR) dataset** was collected from 30 subjects performing daily activities while wearing a smartphone on the waist.

- **Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **Sensors:** 3-axial accelerometer + 3-axial gyroscope
- **Sampling Rate:** 50 Hz
- **Train/Test Split:** Provided by the dataset authors

The dataset consists of pre-segmented and pre-processed time windows for each activity class.

---

## 🧠 Implementation Steps

### 1. Data Preparation
- Downloaded and unzipped the UCI HAR dataset.
- Loaded raw sensor data from the `/Inertial Signals/` directory for both training and testing.
- Transposed and reshaped the data to match LSTM input shape: `(samples, timesteps, features)`.
- Applied one-hot encoding to activity labels.

### 2. Model Architecture

A sequential LSTM model was constructed using the following architecture:
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(6, activation='softmax')
])
