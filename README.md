# Brain-Computer Interface (BCI) Project

A Python-based Brain-Computer Interface system that processes EEG signals for real-time control and analysis.

## Features

- Real-time EEG signal processing and analysis
- Machine learning-based state classification
- Interactive visualization of EEG signals
- Calibration system for personalized settings
- Configuration GUI for easy parameter adjustment
- Comprehensive training pipeline with visualization

## Project Structure

```
BCI/
├── calibration/          # Calibration system
├── data/                # Training and test data
├── gui/                 # Configuration GUI
├── logs/                # Log files
├── models/              # Trained models and results
├── training/            # Training pipeline
├── visualization/       # Signal visualization
├── config.json          # Configuration file
├── prediction.py        # Real-time prediction
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BCI.git
cd BCI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Prepare your training data in CSV format
2. Run the training pipeline:
```bash
python training/train.py
```

### Real-time Prediction

1. Configure your settings in `config.json`
2. Run the prediction script:
```bash
python prediction.py
```

### Calibration

1. Run the calibration system:
```bash
python calibration/calibrator.py
```

### Configuration GUI

1. Launch the configuration GUI:
```bash
python gui/config_gui.py
```

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyAutoGUI
- PySerial
- SciPy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for their valuable tools and libraries


