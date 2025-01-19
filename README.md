# IoT-DDoS-Attack-Detection-using-Network-Traffic-Analysis
This project aims to detect Distributed Denial-of-Service (DDoS) attacks on IoT networks by analyzing network traffic patterns. The system uses machine learning to identify abnormal traffic behaviors and flag potential DDoS attacks targeting IoT devices.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import random

# Simulate network traffic data for IoT devices (e.g., packet size, connection duration)
def generate_network_traffic_data():
    """
    Simulate normal and DDoS attack network traffic for IoT devices.
    """
    normal_traffic = np.random.normal(loc=100, scale=15, size=(100, 3))  # Normal traffic behavior
    ddos_attack_traffic = np.random.normal(loc=1000, scale=150, size=(10, 3))  # DDoS attack-like behavior
    
    data = np.vstack([normal_traffic, ddos_attack_traffic])
    labels = np.array([0] * 100 + [1] * 10)  # 0 for normal traffic, 1 for DDoS attack
    return pd.DataFrame(data, columns=["packet_size", "connection_duration", "request_rate"]), labels

# Train a DDoS detection model using Random Forest Classifier
def train_ddos_detection_model(data):
    """
    Train a RandomForestClassifier model for DDoS attack detection.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data_scaled, labels)
    
    return model, scaler

# Monitor IoT network traffic in real-time and classify as normal or DDoS attack
def monitor_network_traffic(model, scaler):
    """
    Monitor IoT network traffic and classify as normal or DDoS attack in real-time.
    """
    while True:
        # Simulate new network traffic data
        new_data = np.random.normal(loc=100, scale=15, size=(1, 3))  # Simulate normal traffic
        
        # Occasionally simulate DDoS attack traffic
        if random.random() > 0.95:
            new_data = np.random.normal(loc=1000, scale=150, size=(1, 3))  # DDoS attack-like behavior
        
        # Standardize and classify the new data
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        
        if prediction == 1:
            print(f"[ALERT] DDoS attack detected! Traffic: {new_data}")
        else:
            print(f"[INFO] Normal traffic: {new_data}")
        
        time.sleep(2)  # Simulate real-time monitoring delay

if __name__ == "__main__":
    # Step 1: Generate IoT network traffic data
    print("Generating IoT network traffic data...")
    traffic_data, labels = generate_network_traffic_data()

    # Step 2: Train the DDoS detection model
    print("Training DDoS detection model...")
    model, scaler = train_ddos_detection_model(traffic_data)

    # Step 3: Monitor network traffic in real-time
    print("Monitoring IoT network traffic for DDoS attacks...")
    monitor_network_traffic(model, scaler)
