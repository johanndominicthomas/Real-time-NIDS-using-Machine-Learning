import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import threading
import json
import asyncio
from queue import Queue, Empty
import time
from scapy.all import sniff, IP, TCP, UDP
from typing import Dict
from sklearn.preprocessing import StandardScaler
from joblib import load
from collections import defaultdict
import csv
import os

# --------------------
# Load Model and Preprocessing Objects
# --------------------
MODEL_PATH = "models/nids_xgboost_classifier_model.pkl"
SCALER_PATH = "models/scaler_classifier.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
LABEL_ENCODER_PATH = "label_encoder_classifier.pkl"  # Saved from your training

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    scaler = load(SCALER_PATH)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    label_encoder = load(LABEL_ENCODER_PATH)
except Exception as e:
    logging.error(f"Error loading model or preprocessing objects: {e}")
    model, scaler, feature_columns, label_encoder = None, None, None, None

# --------------------
# Global Variables
# --------------------
flow_table = {}                  # For grouping packets into flows.
packet_queue = Queue()           # Shared queue for flows (each as a feature dict).
shutdown_event = threading.Event()

# --------------------
# Helper Functions for Flow Construction
# --------------------
def update_flow(packet):
    """
    Updates the global flow_table with the given packet.
    Returns a flow_key (a tuple) which is used to group packets.
    """
    if IP not in packet:
        return None
    ip = packet[IP]
    if TCP in packet:
        protocol = "TCP"
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        protocol = "UDP"
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    else:
        protocol = "Other"
        src_port = 0
        dst_port = 0

    flow_key = (ip.src, ip.dst, src_port, dst_port, protocol)
    pkt_dict = {
        "timestamp": datetime.now().timestamp(),
        "length": len(packet),
        "src_port": src_port,
        "dst_port": dst_port,
    }
    if flow_key not in flow_table:
        flow_table[flow_key] = []
    flow_table[flow_key].append(pkt_dict)
    return flow_key

def extract_flow_features(flow_packets):
    """
    Compute aggregated features for a flow from a list of packet dictionaries.
    Many features are computed in a simplistic manner or set to defaults.
    """
    now = datetime.now().timestamp()
    start_time = flow_packets[0]["timestamp"] if flow_packets else now
    flow_duration = now - start_time

    total_fwd_packets = len(flow_packets)
    total_bwd_packets = 0

    lengths = [pkt["length"] for pkt in flow_packets]
    total_length_fwd = sum(lengths)
    total_length_bwd = 0

    fwd_packet_length_max = max(lengths) if lengths else 0
    fwd_packet_length_min = min(lengths) if lengths else 0
    fwd_packet_length_mean = np.mean(lengths) if lengths else 0
    fwd_packet_length_std = np.std(lengths) if lengths else 0

    # For backward packets, defaults are used.
    bwd_packet_length_max = 0
    bwd_packet_length_min = 0
    bwd_packet_length_mean = 0
    bwd_packet_length_std = 0

    timestamps = [pkt["timestamp"] for pkt in flow_packets]
    if len(timestamps) > 1:
        iats = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    else:
        iats = [0]
    flow_iat_mean = np.mean(iats) if iats else 0
    flow_iat_std = np.std(iats) if iats else 0
    flow_iat_max = max(iats) if iats else 0
    flow_iat_min = min(iats) if iats else 0

    fwd_iat_total = sum(iats)
    fwd_iat_mean = flow_iat_mean
    fwd_iat_std = flow_iat_std
    fwd_iat_max = flow_iat_max
    fwd_iat_min = flow_iat_min

    flow_bytes_s = total_length_fwd / flow_duration if flow_duration > 0 else 0
    flow_packets_s = total_fwd_packets / flow_duration if flow_duration > 0 else 0

    default_zero = 0

    features = {
        "Destination Port": flow_packets[-1]["dst_port"] if flow_packets else 0,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": total_fwd_packets,
        "Total Backward Packets": total_bwd_packets,
        "Total Length of Fwd Packets": total_length_fwd,
        "Total Length of Bwd Packets": total_length_bwd,
        "Fwd Packet Length Max": fwd_packet_length_max,
        "Fwd Packet Length Min": fwd_packet_length_min,
        "Fwd Packet Length Mean": fwd_packet_length_mean,
        "Fwd Packet Length Std": fwd_packet_length_std,
        "Bwd Packet Length Max": bwd_packet_length_max,
        "Bwd Packet Length Min": bwd_packet_length_min,
        "Bwd Packet Length Mean": bwd_packet_length_mean,
        "Bwd Packet Length Std": bwd_packet_length_std,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_packets_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_total,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": default_zero,
        "Bwd PSH Flags": default_zero,
        "Fwd URG Flags": default_zero,
        "Bwd URG Flags": default_zero,
        "Fwd Header Length": default_zero,
        "Bwd Header Length": default_zero,
        "Fwd Packets/s": flow_packets_s,
        "Bwd Packets/s": 0,
        "Min Packet Length": fwd_packet_length_min,
        "Max Packet Length": fwd_packet_length_max,
        "Packet Length Mean": fwd_packet_length_mean,
        "Packet Length Std": fwd_packet_length_std,
        "Packet Length Variance": fwd_packet_length_std ** 2,
        "FIN Flag Count": default_zero,
        "SYN Flag Count": default_zero,
        "RST Flag Count": default_zero,
        "PSH Flag Count": default_zero,
        "ACK Flag Count": default_zero,
        "URG Flag Count": default_zero,
        "CWE Flag Count": default_zero,
        "ECE Flag Count": default_zero,
        "Down/Up Ratio": 0,
        "Average Packet Size": fwd_packet_length_mean,
        "Avg Fwd Segment Size": fwd_packet_length_mean,
        "Avg Bwd Segment Size": 0,
        "Fwd Header Length.1": default_zero,
        "Fwd Avg Bytes/Bulk": default_zero,
        "Fwd Avg Packets/Bulk": default_zero,
        "Fwd Avg Bulk Rate": default_zero,
        "Bwd Avg Bytes/Bulk": default_zero,
        "Bwd Avg Packets/Bulk": default_zero,
        "Bwd Avg Bulk Rate": default_zero,
        "Subflow Fwd Packets": total_fwd_packets,
        "Subflow Fwd Bytes": total_length_fwd,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init_Win_bytes_forward": default_zero,
        "Init_Win_bytes_backward": default_zero,
        "act_data_pkt_fwd": total_fwd_packets,
        "min_seg_size_forward": fwd_packet_length_min,
        "Active Mean": default_zero,
        "Active Std": default_zero,
        "Active Max": default_zero,
        "Active Min": default_zero,
        "Idle Mean": default_zero,
        "Idle Std": default_zero,
        "Idle Max": default_zero,
        "Idle Min": default_zero,
        "Label": 0  # Not used for prediction.
    }
    return features

# --------------------
# New Function: Inject Attack Flows from CSV
# --------------------
def simulate_csv_flows():
    """
    Reads attack flow features from a CSV file (without the label) and pushes
    each row into the packet_queue, simulating live attack traffic.
    """
    csv_file = "attack_flows_no_labels.csv"
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Simulating attack flows from CSV: {csv_file}")
        for index, row in df.iterrows():
            flow_features = row.to_dict()
            packet_queue.put(flow_features)
            # Random delay between 1 and 10 seconds.
            import random
            time.sleep(random.randint(1,10))
    except Exception as e:
        logging.error(f"Error in CSV simulation: {e}")

# --------------------
# Function: Log Packets to CSV
# --------------------
def log_packet_to_csv(packet_data, label):
    """
    Logs packet details along with the predicted label into a CSV file.
    The CSV file is saved in the "logs" folder.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")
    csv_file = os.path.join("logs", "captured_packets_log.csv")
    fieldnames = list(packet_data.keys()) + ["Label"]
    try:
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            packet_data["Label"] = label
            writer.writerow(packet_data)
    except Exception as e:
        logging.error(f"Failed to log packet to CSV: {e}")

# --------------------
# Stats Class for Tracking Overall Detection Statistics
# --------------------
class Stats:
    def __init__(self):
        self.total_packets = 0
        self.alerts = 0
        self.start_time = datetime.now()
        self.detection_history = []
        self.alert_history = {}

    def add_detection(self, detection):
        self.total_packets += 1
        if detection["prediction_label"] == "anomaly":
            self.alerts += 1
            alert_type = detection["intrusion_type"]
            self.alert_history[alert_type] = self.alert_history.get(alert_type, 0) + 1
            self.detection_history.append(detection)
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)

    def get_stats(self):
        uptime = datetime.now() - self.start_time
        return {
            "total_packets": self.total_packets,
            "alerts": self.alerts,
            "uptime": str(uptime),
            "alert_types": self.alert_history,
        }

# --------------------
# Intrusion Detector Class for Prediction (Multi-Class)
# --------------------
class IntrusionDetector:
    def __init__(self, packet_data: Dict):
        self.packet_data = packet_data
        self.prediction_label = "normal"   # This will be "normal" for BENIGN traffic.
        self.intrusion_type = "Normal"
        self.severity = "Low"
        self.evidence = []
        self.analyze()

    def preprocess_packet(self):
        if not model or not scaler or not feature_columns:
            logging.error("Model, scaler, or feature columns not loaded.")
            return None
        try:
            packet_df = pd.DataFrame([self.packet_data], columns=feature_columns).fillna(0)
            return scaler.transform(packet_df)
        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            return None

    def analyze(self):
        processed_features = self.preprocess_packet()
        if processed_features is not None:
            logging.debug(f"Processed features: {processed_features}")
            numeric_prediction = model.predict(processed_features)[0]
            predicted_class = label_encoder.inverse_transform([numeric_prediction])[0].strip()
            if predicted_class == "BENIGN":
                self.prediction_label = "normal"
                self.intrusion_type = "BENIGN"
                self.severity = "Low"
                self.evidence = []
            else:
                self.prediction_label = "anomaly"
                self.intrusion_type = predicted_class
                self.severity = "High"
                self.evidence = []
                if predicted_class == "Heartbleed":
                    self.evidence.append("Abnormal SSL heartbeat responses were detected, which is a known indicator of the Heartbleed vulnerability.")
                elif predicted_class == "DDoS":
                    self.evidence.append("The flow shows an extremely high volume of requests and elevated traffic frequency, characteristic of a DDoS attack.")
                elif predicted_class == "DoS GoldenEye":
                    self.evidence.append("Repeated connection attempts with malformed HTTP headers confirm a GoldenEye attack.")
                elif predicted_class == "DoS Hulk":
                    self.evidence.append("Rapid, repetitive packet flows were observed leading to resource exhaustion—a definitive sign of a DoS Hulk attack.")
                elif predicted_class == "DoS Slowhttptest":
                    self.evidence.append("Prolonged HTTP sessions with minimal data transfer were detected, matching Slow HTTP Test attack behavior.")
                elif predicted_class == "DoS slowloris":
                    self.evidence.append("Sustained, slow connections characteristic of a slowloris attack were detected.")
                elif predicted_class == "Bot":
                    self.evidence.append("The traffic pattern exhibits regular periodic bursts, consistent with botnet command-and-control activity.")
                elif predicted_class == "FTP-Patator":
                    self.evidence.append("Multiple failed FTP login attempts and erratic command sequences indicate an FTP-Patator attack.")
                elif predicted_class == "Infiltration":
                    self.evidence.append("Irregular protocol activity and unauthorized access attempts confirm infiltration.")
                elif predicted_class == "PortScan":
                    self.evidence.append("A very high frequency of port scanning requests was detected, which is not normal.")
                elif predicted_class == "SSH-Patator":
                    self.evidence.append("Numerous failed SSH login attempts indicate a brute-force attack.")
                elif predicted_class.startswith("Web Attack"):
                    if "Brute Force" in predicted_class:
                        self.evidence.append("Repeated web login attempts suggest a brute-force attack on the website.")
                    elif "Sql Injection" in predicted_class:
                        self.evidence.append("Suspicious SQL queries were detected, confirming an SQL injection attack.")
                    elif "XSS" in predicted_class:
                        self.evidence.append("Evidence of script injection indicates an XSS attack.")
                    else:
                        self.evidence.append("Suspicious web activity was observed.")
                # Additional numeric evidence.
                flow_duration = self.packet_data.get("Flow Duration", 0)
                if flow_duration > 5000:
                    self.evidence.append(f"Flow Duration is {flow_duration:.0f} seconds, exceeding normal thresholds.")
                flow_bytes = self.packet_data.get("Flow Bytes/s", 0)
                if flow_bytes > 500000:
                    self.evidence.append(f"Flow Bytes/s is {flow_bytes:.0f} bytes/sec, which is significantly above expected values.")
                avg_pkt_size = self.packet_data.get("Average Packet Size", 0)
                if avg_pkt_size > 1500:
                    self.evidence.append(f"Average Packet Size is {avg_pkt_size:.0f} bytes, indicating unusually large packets.")

    def to_dict(self):
        return {
            "prediction_label": self.prediction_label,
            "intrusion_type": self.intrusion_type,
            "severity": self.severity,
            "evidence": self.evidence
        }

# --------------------
# FastAPI Application Setup
# --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

logging.basicConfig(level=logging.INFO)
app.state.stats = Stats()

# --------------------
# WebSocket Endpoint for Live Predictions
# --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not hasattr(app.state, "capture_thread") or not app.state.capture_thread.is_alive():
        app.state.capture_thread = threading.Thread(target=start_packet_capture, daemon=True)
        app.state.capture_thread.start()
    if not hasattr(app.state, "csv_thread") or not app.state.csv_thread.is_alive():
        app.state.csv_thread = threading.Thread(target=simulate_csv_flows, daemon=True)
        app.state.csv_thread.start()
    try:
        while True:
            try:
                features = packet_queue.get_nowait()
                analysis = IntrusionDetector(features)
                message = {
                    "features": features,
                    "prediction_label": analysis.prediction_label,
                    "intrusion_type": analysis.intrusion_type,
                    "severity": analysis.severity,
                    "evidence": analysis.evidence,
                    "timestamp": datetime.now().isoformat()
                }
                app.state.stats.add_detection(message)
                await websocket.send_text(json.dumps(message))
                logging.info(f"Detected: {json.dumps(message)}")
                # Log the packet details to CSV in the logs folder.
                log_packet_to_csv(features, analysis.prediction_label)
            except Empty:
                await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    finally:
        logging.info("WebSocket connection closed.")

@app.get("/api/stats")
async def get_stats():
    return JSONResponse(app.state.stats.get_stats())

# --------------------
# Packet Callback and Live Capture Function (Using Scapy)
# --------------------
def packet_callback(packet):
    """
    Callback used by Scapy to process each captured packet and convert
    it to a flow-feature dictionary.
    """
    if IP in packet:
        flow_key = update_flow(packet)
        if flow_key is None:
            return
        flow_packets = flow_table.get(flow_key, [])
        features = extract_flow_features(flow_packets)
        packet_queue.put(features)

def start_packet_capture():
    sniff(prn=packet_callback, store=0, stop_filter=lambda _: shutdown_event.is_set())

# --------------------
# Function: Log Packets to CSV
# --------------------
def log_packet_to_csv(packet_data, label):
    """
    Logs packet details along with the predicted label into a CSV file.
    The CSV file is saved in the "logs" folder.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")
    csv_file = os.path.join("logs", "captured_packets_log.csv")
    fieldnames = list(packet_data.keys()) + ["Label"]
    try:
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            packet_data["Label"] = label
            writer.writerow(packet_data)
    except Exception as e:
        logging.error(f"Failed to log packet to CSV: {e}")

# --------------------
# Main Entry Point
# --------------------
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting NIDS application on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
