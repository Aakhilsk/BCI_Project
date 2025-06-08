import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy import signal

class BCIConfigGUI:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_gui()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_config()
            
    def create_default_config(self):
        """Create default configuration"""
        return {
            'sampling_rate': 512,
            'buffer_size': 512,
            'serial_port': 'COM11',
            'baud_rate': 115200,
            'key_mappings': {
                '0': {'key': 'space', 'duration': 2},
                '1': {'key': 'w', 'duration': 2}
            },
            'filter_settings': {
                'notch_freq': 50.0,
                'notch_quality': 30.0,
                'bandpass_low': 0.5,
                'bandpass_high': 30.0,
                'bandpass_order': 4
            }
        }
        
    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("BCI Configuration")
        self.root.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create tabs
        self.create_general_tab()
        self.create_filter_tab()
        self.create_key_mapping_tab()
        self.create_visualization_tab()
        
        # Create save button
        self.save_button = ttk.Button(self.root, text="Save Configuration", command=self.save_config)
        self.save_button.pack(pady=10)
        
    def create_general_tab(self):
        """Create general settings tab"""
        general_frame = ttk.Frame(self.notebook)
        self.notebook.add(general_frame, text="General")
        
        # Sampling rate
        ttk.Label(general_frame, text="Sampling Rate (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.sampling_rate = ttk.Entry(general_frame)
        self.sampling_rate.insert(0, str(self.config['sampling_rate']))
        self.sampling_rate.grid(row=0, column=1, padx=5, pady=5)
        
        # Buffer size
        ttk.Label(general_frame, text="Buffer Size:").grid(row=1, column=0, padx=5, pady=5)
        self.buffer_size = ttk.Entry(general_frame)
        self.buffer_size.insert(0, str(self.config['buffer_size']))
        self.buffer_size.grid(row=1, column=1, padx=5, pady=5)
        
        # Serial port
        ttk.Label(general_frame, text="Serial Port:").grid(row=2, column=0, padx=5, pady=5)
        self.serial_port = ttk.Entry(general_frame)
        self.serial_port.insert(0, self.config['serial_port'])
        self.serial_port.grid(row=2, column=1, padx=5, pady=5)
        
        # Baud rate
        ttk.Label(general_frame, text="Baud Rate:").grid(row=3, column=0, padx=5, pady=5)
        self.baud_rate = ttk.Entry(general_frame)
        self.baud_rate.insert(0, str(self.config['baud_rate']))
        self.baud_rate.grid(row=3, column=1, padx=5, pady=5)
        
    def create_filter_tab(self):
        """Create filter settings tab"""
        filter_frame = ttk.Frame(self.notebook)
        self.notebook.add(filter_frame, text="Filters")
        
        # Notch filter
        ttk.Label(filter_frame, text="Notch Frequency (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.notch_freq = ttk.Entry(filter_frame)
        self.notch_freq.insert(0, str(self.config['filter_settings']['notch_freq']))
        self.notch_freq.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Notch Quality:").grid(row=1, column=0, padx=5, pady=5)
        self.notch_quality = ttk.Entry(filter_frame)
        self.notch_quality.insert(0, str(self.config['filter_settings']['notch_quality']))
        self.notch_quality.grid(row=1, column=1, padx=5, pady=5)
        
        # Bandpass filter
        ttk.Label(filter_frame, text="Bandpass Low (Hz):").grid(row=2, column=0, padx=5, pady=5)
        self.bandpass_low = ttk.Entry(filter_frame)
        self.bandpass_low.insert(0, str(self.config['filter_settings']['bandpass_low']))
        self.bandpass_low.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Bandpass High (Hz):").grid(row=3, column=0, padx=5, pady=5)
        self.bandpass_high = ttk.Entry(filter_frame)
        self.bandpass_high.insert(0, str(self.config['filter_settings']['bandpass_high']))
        self.bandpass_high.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Bandpass Order:").grid(row=4, column=0, padx=5, pady=5)
        self.bandpass_order = ttk.Entry(filter_frame)
        self.bandpass_order.insert(0, str(self.config['filter_settings']['bandpass_order']))
        self.bandpass_order.grid(row=4, column=1, padx=5, pady=5)
        
    def create_key_mapping_tab(self):
        """Create key mapping tab"""
        key_frame = ttk.Frame(self.notebook)
        self.notebook.add(key_frame, text="Key Mappings")
        
        # Create key mapping entries
        self.key_entries = {}
        for i, (state, mapping) in enumerate(self.config['key_mappings'].items()):
            ttk.Label(key_frame, text=f"State {state}:").grid(row=i, column=0, padx=5, pady=5)
            
            # Key
            ttk.Label(key_frame, text="Key:").grid(row=i, column=1, padx=5, pady=5)
            key_entry = ttk.Entry(key_frame)
            key_entry.insert(0, mapping['key'])
            key_entry.grid(row=i, column=2, padx=5, pady=5)
            
            # Duration
            ttk.Label(key_frame, text="Duration (s):").grid(row=i, column=3, padx=5, pady=5)
            duration_entry = ttk.Entry(key_frame)
            duration_entry.insert(0, str(mapping['duration']))
            duration_entry.grid(row=i, column=4, padx=5, pady=5)
            
            self.key_entries[state] = {'key': key_entry, 'duration': duration_entry}
            
    def create_visualization_tab(self):
        """Create visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add test signal button
        ttk.Button(viz_frame, text="Test Filter", command=self.test_filter).pack(pady=10)
        
    def test_filter(self):
        """Test filter settings with a sample signal"""
        try:
            # Generate test signal
            t = np.linspace(0, 1, 1000)
            signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
            
            # Apply filters
            notch_freq = float(self.notch_freq.get())
            notch_quality = float(self.notch_quality.get())
            bandpass_low = float(self.bandpass_low.get())
            bandpass_high = float(self.bandpass_high.get())
            bandpass_order = int(self.bandpass_order.get())
            
            b_notch, a_notch = signal.iirnotch(notch_freq / (0.5 * 1000), notch_quality)
            b_bandpass, a_bandpass = signal.butter(bandpass_order, 
                                                 [bandpass_low / (0.5 * 1000), 
                                                  bandpass_high / (0.5 * 1000)], 
                                                 'band')
            
            filtered_signal = signal.filtfilt(b_notch, a_notch, signal)
            filtered_signal = signal.filtfilt(b_bandpass, a_bandpass, filtered_signal)
            
            # Plot results
            self.ax1.clear()
            self.ax1.plot(t, signal, label='Original')
            self.ax1.plot(t, filtered_signal, label='Filtered')
            self.ax1.set_title('Time Domain')
            self.ax1.legend()
            
            # Plot frequency domain
            f_orig, psd_orig = signal.welch(signal, fs=1000)
            f_filt, psd_filt = signal.welch(filtered_signal, fs=1000)
            
            self.ax2.clear()
            self.ax2.semilogy(f_orig, psd_orig, label='Original')
            self.ax2.semilogy(f_filt, psd_filt, label='Filtered')
            self.ax2.set_title('Frequency Domain')
            self.ax2.legend()
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error testing filter: {str(e)}")
            
    def save_config(self):
        """Save configuration to file"""
        try:
            # Update configuration
            self.config['sampling_rate'] = int(self.sampling_rate.get())
            self.config['buffer_size'] = int(self.buffer_size.get())
            self.config['serial_port'] = self.serial_port.get()
            self.config['baud_rate'] = int(self.baud_rate.get())
            
            # Update filter settings
            self.config['filter_settings'] = {
                'notch_freq': float(self.notch_freq.get()),
                'notch_quality': float(self.notch_quality.get()),
                'bandpass_low': float(self.bandpass_low.get()),
                'bandpass_high': float(self.bandpass_high.get()),
                'bandpass_order': int(self.bandpass_order.get())
            }
            
            # Update key mappings
            self.config['key_mappings'] = {}
            for state, entries in self.key_entries.items():
                self.config['key_mappings'][state] = {
                    'key': entries['key'].get(),
                    'duration': float(entries['duration'].get())
                }
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            messagebox.showinfo("Success", "Configuration saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration: {str(e)}")
            
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == '__main__':
    gui = BCIConfigGUI()
    gui.run() 