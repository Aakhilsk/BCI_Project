import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import json
import os
from datetime import datetime
import logging
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BCITrainer:
    def __init__(self, config_path='config.json'):
        """Initialize the BCI trainer with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            raise
            
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
    def load_data(self, data_path):
        """Load and preprocess training data."""
        try:
            df = pd.read_csv(data_path)
            logging.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df):
        """Preprocess the data and extract features."""
        # Extract features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        
        return X_scaled, y, scaler
        
    def train_model(self, X, y):
        """Train the model using GridSearchCV."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        # Create and train model
        model = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Log results
        logging.info(f"Best parameters: {model.best_params_}")
        logging.info(f"Best cross-validation score: {model.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        logging.info("\nClassification Report:\n" + 
                    classification_report(y_test, y_pred))
        
        return model, X_test, y_test
        
    def save_model(self, model, scaler, results_dir='models'):
        """Save the trained model and results."""
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Save model
        model_path = os.path.join(results_dir, 'model.pkl')
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save results
        results = {
            'best_params': model.best_params_,
            'best_score': model.best_score_,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = os.path.join(results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {results_path}")
        
    def plot_feature_importance(self, X, feature_names):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        importances = np.abs(np.corrcoef(X.T, y)[:-1, -1])
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('models/roc_curve.png')
        plt.close()
        
    def plot_learning_curve(self, model, X, y):
        """Plot learning curve."""
        plt.figure(figsize=(10, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, 
                        test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('models/learning_curve.png')
        plt.close()
        
    def run_training(self, data_path):
        """Run the complete training pipeline."""
        try:
            # Load and preprocess data
            df = self.load_data(data_path)
            X, y, scaler = self.preprocess_data(df)
            
            # Train model
            model, X_test, y_test = self.train_model(X, y)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Save model and results
            self.save_model(model, scaler)
            
            # Create plots
            self.plot_feature_importance(X, df.columns[:-1])
            self.plot_confusion_matrix(y_test, y_pred)
            self.plot_roc_curve(y_test, y_pred_proba)
            self.plot_learning_curve(model, X, y)
            
            logging.info("Training completed successfully")
            return model, scaler
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    trainer = BCITrainer()
    model, scaler = trainer.run_training('data/training_data.csv') 