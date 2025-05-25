# Geometric Action-Based Authentication Neural Network (GABAN)

> A neural network-based authentication system that uses geometric action patterns from accelerometer data to verify user identity through movement analysis.

**Advanced biometric authentication using deep learning and motion patterns**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)

## 🚀 Overview

This project implements an innovative authentication system that leverages geometric action patterns captured through accelerometer data. By analyzing user movement patterns in both time and frequency domains, the system can accurately identify and authenticate individuals based on their unique motion signatures.

## ✨ Key Features

- **🔄 Dual-Input Architecture**: Processes both time-domain and frequency-domain features simultaneously
- **📊 Advanced Data Preprocessing**: Implements Global Average Pooling for feature optimization
- **🧠 Custom Neural Network**: Multi-layer architecture with sophisticated regularization
- **⚡ Real-time Classification**: Binary classification for authentic user verification
- **📈 High Accuracy**: Achieves 90% test accuracy with robust validation
- **🛡️ Regularization**: Comprehensive dropout, batch normalization, and L1/L2 regularization

## 🏗️ Model Architecture

### Dual-Branch Neural Network

#### 🌊 Frequency Domain Branch (43 features)
```
Input (43) → Dense(64) → BatchNorm → Dropout → Dense(32) → Dropout
```

#### ⏰ Time Domain Branch (88 features)
```
Input (88) → Dense(64) → BatchNorm → Dropout → Dense(32) → Dropout
```

#### 🔗 Merged Architecture
```
Concatenate → Dense(64) → BatchNorm → Dropout → Dense(32) → Dropout → Dense(16) → Output(2, softmax)
```

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 100% |
| **Validation Accuracy** | 88.89% |
| **Test Accuracy** | 90% |

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- [Python](https://www.python.org/) 3.x
- [pip](https://pip.pypa.io/en/stable/) package manager

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/GABAN.git
cd GABAN
```

### 2. Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Required packages

```txt
tensorflow>=2.10.0
keras>=2.10.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

## 📁 Data Structure

### Dataset Organization

Your dataset should follow this naming convention:

```
data/
├── U01_Acc_TimeD_FDay.csv    # User 1 Time Domain Training
├── U01_Acc_FreqD_FDay.csv    # User 1 Frequency Domain Training
├── U01_Acc_TimeD_MDay.csv    # User 1 Time Domain Testing
├── U01_Acc_FreqD_MDay.csv    # User 1 Frequency Domain Testing
├── U02_Acc_TimeD_FDay.csv    # User 2 Time Domain Training
└── ...
```

### File Naming Convention

```
U[UserID]_Acc_[TimeD/FreqD]_[FDay/MDay].csv
```

- **UserID**: Unique identifier for each user
- **TimeD/FreqD**: Time Domain or Frequency Domain features
- **FDay/MDay**: Training (FDay) or Testing (MDay) data

## 🎯 Usage

### 1. Prepare your dataset

Ensure your CSV files contain accelerometer data with proper labeling and follow the naming convention above.

### 2. Run the authentication system

```bash
# Launch Jupyter notebook
jupyter notebook

# Or run the Python script directly
python train_model.py
```

### 3. Training Process

The system will automatically:
- 📥 Load and preprocess data
- 🔧 Build the dual-input neural network
- 🏋️ Train the model with early stopping
- 📊 Evaluate performance metrics
- 💾 Save the trained model

### 4. Authentication

```python
# Load trained model
model = load_model('gaban_model.h5')

# Authenticate user with new accelerometer data
prediction = model.predict([freq_features, time_features])
is_authentic = prediction[0][1] > 0.5
```

## 🔧 Configuration

### Training Parameters

```python
# Model configuration
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Regularization
DROPOUT_RATE = 0.3
L1_REG = 0.01
L2_REG = 0.01
```

## 📈 Model Training Details

### Optimizer & Loss Function
- **Optimizer**: Adam (learning_rate=0.01)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Implemented to prevent overfitting

### Regularization Techniques
- **Dropout**: 30% dropout rate for regularization
- **Batch Normalization**: Applied after dense layers
- **L1/L2 Regularization**: Weight decay for better generalization

## 🚀 Future Improvements

- [ ] **📱 Real-time Authentication**: Live accelerometer data processing
- [ ] **👥 Multi-user Scaling**: Support for larger user databases
- [ ] **🔄 Data Augmentation**: Synthetic data generation techniques
- [ ] **🏗️ Architecture Exploration**: CNN, LSTM, and Transformer models
- [ ] **📊 Feature Engineering**: Advanced signal processing techniques
- [ ] **🌐 Web Interface**: User-friendly authentication portal
- [ ] **📱 Mobile Integration**: Smartphone app development
- [ ] **🔒 Security Enhancement**: Adversarial attack resistance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research Applications

This implementation demonstrates:

- **Biometric Authentication**: Novel approach using motion patterns
- **Deep Learning**: Dual-input neural network architecture
- **Signal Processing**: Time and frequency domain analysis
- **Security Systems**: Advanced user verification methods

## 📚 References

- Deep Learning for Biometric Authentication
- Accelerometer-based Human Activity Recognition
- Neural Network Architectures for Time Series Analysis
- Motion Pattern Recognition in Security Systems

## 🙏 Acknowledgments

- Research community working on biometric authentication
- Contributors to TensorFlow and Keras frameworks
- Developers of accelerometer data analysis techniques
- Open-source machine learning community

---

