import serial
import numpy as np
from scipy import signal
import pandas as pd
import time
import pickle
import pyautogui
import logging
import os
from dotenv import load_dotenv
from collections import deque
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bci_prediction.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

class BCIPredictor:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.sampling_rate = self.config['sampling_rate']
        self.buffer_size = self.config['buffer_size']
        self.serial_port = self.config['serial_port']
        self.baud_rate = self.config['baud_rate']
        self.buffer = deque(maxlen=self.buffer_size)
        self.setup_filters()
        self.load_model_and_scaler()
        self.setup_logging()
        
    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'sampling_rate': 512,
            'buffer_size': 512,
            'serial_port': 'COM11',
            'baud_rate': 115200,
            'key_mappings': {
                '0': {'key': 'space', 'duration': 2},
                '1': {'key': 'w', 'duration': 2}
            }
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bci_prediction.log'),
                logging.StreamHandler()
            ]
        )

    def setup_filters(self):
        """Setup filters as per notebook implementation"""
        # Notch filter
        nyquist = 0.5 * self.sampling_rate
        notch_freq = 50.0
        notch_freq_normalized = notch_freq / nyquist
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq_normalized, Q=0.05, fs=self.sampling_rate)
        
        # Bandpass filter
        lowcut, highcut = 0.5, 30.0
        lowcut_normalized = lowcut / nyquist
        highcut_normalized = highcut / nyquist
        self.b_bandpass, self.a_bandpass = signal.butter(4, [lowcut_normalized, highcut_normalized], btype='band')
        
    def calculate_psd_features(self, segment):
        """Calculate PSD features as per notebook"""
        f, psd_values = signal.welch(segment, fs=self.sampling_rate, nperseg=len(segment))
        
        # Define frequency bands
        alpha_indices = np.where((f >= 8) & (f <= 13))
        beta_indices = np.where((f >= 14) & (f <= 30))
        theta_indices = np.where((f >= 4) & (f <= 7))
        delta_indices = np.where((f >= 0.5) & (f <= 3))
        
        # Calculate band energies
        energy_alpha = np.sum(psd_values[alpha_indices])
        energy_beta = np.sum(psd_values[beta_indices])
        energy_theta = np.sum(psd_values[theta_indices])
        energy_delta = np.sum(psd_values[delta_indices])
        
        # Calculate alpha-beta ratio
        alpha_beta_ratio = energy_alpha / energy_beta if energy_beta > 0 else 0
        
        return {
            'E_alpha': energy_alpha,
            'E_beta': energy_beta,
            'E_theta': energy_theta,
            'E_delta': energy_delta,
            'alpha_beta_ratio': alpha_beta_ratio
        }
        
    def calculate_additional_features(self, segment):
        """Calculate additional features as per notebook"""
        f, psd = signal.welch(segment, fs=self.sampling_rate, nperseg=len(segment))
        
        # Peak frequency
        peak_frequency = f[np.argmax(psd)]
        
        # Spectral centroid
        spectral_centroid = np.sum(f * psd) / np.sum(psd)
        
        # Spectral slope
        log_f = np.log(f[1:])
        log_psd = np.log(psd[1:])
        spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
        
        return {
            'peak_frequency': peak_frequency,
            'spectral_centroid': spectral_centroid,
            'spectral_slope': spectral_slope
        }
        
    def process_segment(self, segment):
        """Process a segment of EEG data"""
        # Apply filters
        segment = signal.filtfilt(self.b_notch, self.a_notch, segment)
        segment = signal.filtfilt(self.b_bandpass, self.a_bandpass, segment)
        
        # Calculate features
        psd_features = self.calculate_psd_features(segment)
        additional_features = self.calculate_additional_features(segment)
        
        # Combine features
        features = {**psd_features, **additional_features}
        return features
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler"""
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logging.info("Model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model and scaler: {e}")
            raise
            
    def execute_action(self, prediction):
        """Execute the corresponding action based on prediction"""
        try:
            action = self.config['key_mappings'].get(str(prediction))
            if action:
                pyautogui.keyDown(action['key'])
                time.sleep(action['duration'])
                pyautogui.keyUp(action['key'])
                logging.info(f"Executed action for prediction {prediction}")
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            
    def run(self):
        """Main prediction loop"""
        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as ser:
                logging.info(f"Connected to {self.serial_port}")
                while True:
                    try:
                        raw_data = ser.readline().decode('latin-1').strip()
                        if raw_data:
                            eeg_value = float(raw_data)
                            self.buffer.append(eeg_value)
                            
                            if len(self.buffer) == self.buffer_size:
                                # Process the buffer
                                buffer_array = np.array(self.buffer)
                                features = self.process_segment(buffer_array)
                                
                                # Convert features to DataFrame
                                df = pd.DataFrame([features])
                                
                                # Scale features
                                X_scaled = self.scaler.transform(df)
                                
                                # Make prediction
                                prediction = self.model.predict(X_scaled)[0]
                                logging.info(f"Predicted Class: {prediction}")
                                
                                # Execute action
                                self.execute_action(prediction)
                                
                                # Clear buffer
                                self.buffer.clear()
                                
                    except ValueError as ve:
                        logging.warning(f"Invalid data received: {ve}")
                        continue
                    except Exception as e:
                        logging.error(f"Error in main loop: {e}")
                        continue
                        
        except serial.SerialException as se:
            logging.error(f"Serial port error: {se}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            
def main():
    try:
        predictor = BCIPredictor()
        predictor.run()
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.error(f"Program terminated due to error: {e}")
        
if __name__ == '__main__':
    main()
