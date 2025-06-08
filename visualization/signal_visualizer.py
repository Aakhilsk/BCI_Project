import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import signal
import pandas as pd
from collections import deque
import threading
import queue

class SignalVisualizer:
    def __init__(self, buffer_size=512, sampling_rate=512):
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.data_queue = queue.Queue()
        self.is_running = False
        self.fig = None
        self.ani = None
        
    def setup_plot(self):
        """Setup the plotting environment"""
        plt.style.use('seaborn')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('EEG Signal Visualization', fontsize=16)
        
        # Time domain plot
        self.ax1.set_title('Raw Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        self.line1, = self.ax1.plot([], [], 'b-', label='Raw Signal')
        self.line2, = self.ax1.plot([], [], 'r-', label='Filtered Signal')
        self.ax1.legend()
        
        # Frequency domain plot
        self.ax2.set_title('Power Spectral Density')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Power')
        self.line3, = self.ax2.plot([], [], 'g-', label='PSD')
        self.ax2.legend()
        
        plt.tight_layout()
        
    def update_plot(self, frame):
        """Update the plot with new data"""
        try:
            raw_data = self.data_queue.get_nowait()
            if raw_data is not None:
                # Time domain plot
                time = np.arange(len(raw_data)) / self.sampling_rate
                self.line1.set_data(time, raw_data)
                self.ax1.set_xlim(0, len(raw_data) / self.sampling_rate)
                self.ax1.set_ylim(min(raw_data) * 1.1, max(raw_data) * 1.1)
                
                # Calculate and plot PSD
                f, psd = signal.welch(raw_data, fs=self.sampling_rate)
                self.line3.set_data(f, psd)
                self.ax2.set_xlim(0, 50)  # Show up to 50 Hz
                self.ax2.set_ylim(0, max(psd) * 1.1)
                
        except queue.Empty:
            pass
        return self.line1, self.line2, self.line3
    
    def start_visualization(self):
        """Start the real-time visualization"""
        self.is_running = True
        self.setup_plot()
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        plt.show()
        
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        if self.ani is not None:
            self.ani.event_source.stop()
        plt.close(self.fig)
        
    def add_data(self, data):
        """Add new data to the visualization queue"""
        if self.is_running:
            self.data_queue.put(data)
            
    def plot_offline_data(self, data, title="EEG Signal Analysis"):
        """Plot offline data analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)
        
        # Time domain plot
        time = np.arange(len(data)) / self.sampling_rate
        ax1.plot(time, data, 'b-', label='Raw Signal')
        ax1.set_title('Time Domain')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        
        # Frequency domain plot
        f, psd = signal.welch(data, fs=self.sampling_rate)
        ax2.semilogy(f, psd)
        ax2.set_title('Power Spectral Density')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power')
        ax2.set_xlim(0, 50)
        
        # Spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_rate)
        ax3.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax3.set_title('Spectrogram')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_ylim(0, 50)
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, features, importance):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, cm, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show() 