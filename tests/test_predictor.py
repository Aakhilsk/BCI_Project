import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction import BCIPredictor

class TestBCIPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = BCIPredictor()
        
    def test_filter_setup(self):
        """Test if filters are properly initialized"""
        self.assertIsNotNone(self.predictor.b_notch)
        self.assertIsNotNone(self.predictor.a_notch)
        self.assertIsNotNone(self.predictor.b_bandpass)
        self.assertIsNotNone(self.predictor.a_bandpass)
        
    def test_process_eeg_data(self):
        """Test EEG data processing"""
        test_data = np.random.rand(512)
        processed_data = self.predictor.process_eeg_data(test_data)
        self.assertIsNotNone(processed_data)
        self.assertEqual(len(processed_data), 512)
        
    def test_calculate_features(self):
        """Test feature calculation"""
        test_data = np.random.rand(512)
        features = self.predictor.calculate_features(test_data)
        self.assertIsNotNone(features)
        self.assertIn('E_alpha', features)
        self.assertIn('E_beta', features)
        self.assertIn('E_theta', features)
        self.assertIn('E_delta', features)
        
    @patch('serial.Serial')
    def test_serial_connection(self, mock_serial):
        """Test serial connection handling"""
        mock_serial.return_value.readline.return_value = b'1.0\n'
        with self.assertRaises(Exception):
            self.predictor.run()
            
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(self.predictor.config)
        self.assertIn('sampling_rate', self.predictor.config)
        self.assertIn('buffer_size', self.predictor.config)
        self.assertIn('key_mappings', self.predictor.config)

if __name__ == '__main__':
    unittest.main() 