import numpy as np
import pandas as pd
from scipy import signal
import time
import json
import os
from datetime import datetime
import logging

class BCICalibrator:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.sampling_rate = self.config['sampling_rate']
        self.buffer_size = self.config['buffer_size']
        self.calibration_data = {
            'attention': [],
            'relaxation': [],
            'baseline': []
        }
        self.setup_logging()
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('calibration.log'),
                logging.StreamHandler()
            ]
        )
        
    def collect_calibration_data(self, state, duration=30):
        """Collect calibration data for a specific state"""
        logging.info(f"Starting calibration for {state} state")
        print(f"Please maintain {state} state for {duration} seconds...")
        
        data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Here you would collect data from your EEG device
            # This is a placeholder - replace with actual data collection
            sample = np.random.randn(self.buffer_size)  # Simulated data
            data.append(sample)
            time.sleep(1)
            
        self.calibration_data[state] = data
        logging.info(f"Completed calibration for {state} state")
        
    def calculate_baseline(self):
        """Calculate baseline parameters from calibration data"""
        baseline_params = {
            'mean': {},
            'std': {},
            'frequency_bands': {}
        }
        
        for state, data in self.calibration_data.items():
            if data:
                # Calculate time domain statistics
                baseline_params['mean'][state] = np.mean(data)
                baseline_params['std'][state] = np.std(data)
                
                # Calculate frequency domain statistics
                for sample in data:
                    f, psd = signal.welch(sample, fs=self.sampling_rate)
                    for band, (low, high) in self.config['feature_settings']['bands'].items():
                        idx = np.where((f >= low) & (f <= high))
                        if band not in baseline_params['frequency_bands']:
                            baseline_params['frequency_bands'][band] = {}
                        if state not in baseline_params['frequency_bands'][band]:
                            baseline_params['frequency_bands'][band][state] = []
                        baseline_params['frequency_bands'][band][state].append(np.mean(psd[idx]))
        
        return baseline_params
        
    def save_calibration(self, baseline_params):
        """Save calibration results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_dir = 'calibration_results'
        os.makedirs(calibration_dir, exist_ok=True)
        
        # Save baseline parameters
        with open(f'{calibration_dir}/baseline_{timestamp}.json', 'w') as f:
            json.dump(baseline_params, f, indent=4)
            
        # Save raw calibration data
        np.save(f'{calibration_dir}/calibration_data_{timestamp}.npy', self.calibration_data)
        
        logging.info(f"Calibration results saved to {calibration_dir}")
        
    def run_calibration(self):
        """Run the complete calibration process"""
        try:
            # Collect data for each state
            self.collect_calibration_data('baseline', duration=30)
            self.collect_calibration_data('attention', duration=30)
            self.collect_calibration_data('relaxation', duration=30)
            
            # Calculate baseline parameters
            baseline_params = self.calculate_baseline()
            
            # Save calibration results
            self.save_calibration(baseline_params)
            
            logging.info("Calibration completed successfully")
            return baseline_params
            
        except Exception as e:
            logging.error(f"Error during calibration: {e}")
            raise
            
    def validate_calibration(self, baseline_params):
        """Validate calibration results"""
        validation_results = {
            'state_separation': {},
            'signal_quality': {}
        }
        
        # Check state separation
        for band in self.config['feature_settings']['bands']:
            attention_mean = np.mean(baseline_params['frequency_bands'][band]['attention'])
            relaxation_mean = np.mean(baseline_params['frequency_bands'][band]['relaxation'])
            separation = abs(attention_mean - relaxation_mean)
            validation_results['state_separation'][band] = separation
            
        # Check signal quality
        for state in ['attention', 'relaxation']:
            snr = baseline_params['mean'][state] / baseline_params['std'][state]
            validation_results['signal_quality'][state] = snr
            
        return validation_results
        
    def update_config(self, baseline_params):
        """Update configuration with calibration results"""
        self.config['calibration'] = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_params
        }
        
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
            
        logging.info("Configuration updated with calibration results") 