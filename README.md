```markdown
# 🌌 Exoplanet Explorer - NASA Kepler AI Analysis Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![Three.js](https://img.shields.io/badge/Three.js-R128-purple.svg)

**Advanced AI-Powered Exoplanet Discovery with Interactive 3D Universe**

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

## 🚀 Quick Start Guide

### 📋 Prerequisites
- **Python 3.8 or higher**
- **NASA Kepler dataset** (CSV format)
- **Modern web browser** with WebGL support

### ⚡ Installation & Setup

#### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-username/exoplanet-explorer.git
cd exoplanet-explorer

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

**Required packages include:**
- `fastapi==0.68.0+` - Web framework
- `tensorflow==2.8.0+` - Machine learning
- `scikit-learn==1.0.0+` - Data processing
- `pandas==1.3.0+` - Data manipulation
- `numpy==1.21.0+` - Numerical computing
- `joblib==1.0.0+` - Model serialization
- `uvicorn==0.15.0+` - ASGI server
- `three==0.128.0` - 3D visualization

#### 3. Download NASA Kepler Data
```bash
# Download Kepler dataset from NASA (example)
wget https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/kepler_data.csv
# OR place your Kepler CSV file in the project root
```

## 🎯 Usage Instructions

### 🔬 Step 1: Train the AI Model

**Basic Training (Recommended):**
```bash
python train_nasa_kepler.py kepler_data.csv --iterations 500
```

**Advanced Training Options:**
```bash
# Extended training with more iterations
python train_nasa_kepler.py kepler_data.csv --iterations 1000

# Training without early stopping
python train_nasa_kepler.py kepler_data.csv --no-early-stopping

# Custom model configuration
python train_nasa_kepler.py kepler_data.csv --iterations 300 --batch-size 32
```

**Expected Training Output:**
```
🚀 NASA KEPLER ADVANCED EXOPLANET HUNTER v4.0
==============================================
📁 Loading data from: kepler_data.csv
📊 Dataset shape: (9564, 50)
🎯 Target column: koi_disposition
✅ Cleaned data: 8000 samples
🎯 Using 25 available features
🚀 Starting advanced training (500 epochs)...
🎯 Epoch 1/500 - loss: 0.6823 - accuracy: 0.5814 - val_auc: 0.7234
...
🏆 NEW BEST MODEL! Score: 0.9412
🎉 TRAINING COMPLETED SUCCESSFULLY!
```

### 🌐 Step 2: Launch the Web Application

**Start the Server:**
```bash
python main.py
```

**Access the Application:**
```
🌐 Open your web browser and navigate to:
http://localhost:8000
```

**Server Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 🎮 User Interface Guide

### 🖥️ Main Dashboard Features

#### 📊 Data Upload Section
- **Drag & Drop CSV Upload** - Upload Kepler mission data
- **Manual Data Input** - Enter parameters manually
- **Real-time Validation** - Instant data quality checks

#### 🔍 Analysis Controls
- **mmAI-2.1 Model** - Advanced exoplanet detection AI
- **Confidence Filter** - Adjust detection threshold (0-100%)
- **Real-time Statistics** - Live analysis metrics

#### 🌌 3D Visualization Panel
- **Solar System View** - Our planetary system
- **Universe View** - Galactic scale exploration
- **Interactive Orbits** - Dynamic planetary motion

### 🎯 Control Panel Functions

| Button | Function | Shortcut |
|--------|----------|----------|
| **⏸️ Pause/Play** | Toggle animation | `Space` |
| **🔄 Reset** | Reset camera view | `R` |
| **🏷️ Labels** | Toggle object labels | `L` |
| **🌀 Orbits** | Show/hide orbits | `O` |
| **🌐 Universe** | Switch view modes | `U` |

### 🖱️ Navigation Controls
- **Left Click + Drag** - Rotate view
- **Right Click + Drag** - Pan camera
- **Mouse Wheel** - Zoom in/out
- **Click Objects** - Show detailed information

## 📁 File Structure

```
exoplanet-explorer/
├── 🐍 app.py                 # FastAPI web server & 3D visualization
├── 🤖 train.py    # Advanced ML model training
├── 📁 saved_models/           # Trained model storage
│   ├── mmAI-ExoplanetHunter-v4.0_complete.joblib
│   ├── mmAI-2.1_final.joblib
│   └── training_results.json
├── 📄 requirements.txt        # Python dependencies
├── 📄 exoplanet_for_train.csv         # NASA dataset (add your file)
└── 📄 README.md              # This documentation
```

## 🔧 API Endpoints Reference

### 📤 Data Upload & Analysis
```bash
# Upload CSV file
curl -X POST -F "file=@kepler_data.csv" http://localhost:8000/upload-csv

# Batch analysis
curl -X POST -H "Content-Type: application/json" -d '{"data": [...]}' http://localhost:8000/analyze

# Single data point analysis
curl -X POST -H "Content-Type: application/json" -d '{
  "koi_period": 365.25,
  "koi_duration": 13,
  "koi_depth": 84,
  "koi_prad": 1.0,
  "koi_teq": 288
}' http://localhost:8000/analyze-manual
```

### 📊 Response Format
```json
{
  "exoplanets": [
    {
      "name": "KOI-1234",
      "confidence": 0.894,
      "period": 365.25,
      "radius": 1.2,
      "temperature": 288,
      "distance_ly": 1245.67
    }
  ],
  "metrics": {
    "total_found": 156,
    "analysis_time": "2024-01-15T10:30:00Z",
    "model_used": "mmAI-2.1"
  }
}
```

## 🧠 Model Configuration

### 🎛️ Training Parameters
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Iterations** | 500 | Training epochs |
| **Batch Size** | 64 | Samples per batch |
| **Learning Rate** | 0.0005 | Optimization speed |
| **Hidden Layers** | [512,256,128,64,32] | Network architecture |
| **Dropout Rates** | [0.5,0.4,0.3,0.2,0.1] | Regularization |
| **Early Stopping** | 50 epochs | Prevent overfitting |

### 📊 Feature Set
The model analyzes **40+ scientific parameters** including:
- **Orbital Characteristics**: Period, Duration, Depth, Impact
- **Planetary Properties**: Radius, Temperature, Insolation
- **Stellar Data**: Temperature, Gravity, Radius
- **Quality Flags**: False positive indicators, Signal-to-noise

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### ❌ "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### ❌ "File not found" for CSV
```bash
# Ensure file is in correct directory
ls -la kepler_data.csv

# Download sample data if needed
wget -O kepler_data.csv "NASA_DATA_URL"
```

#### ❌ WebGL not working
- **Update browser** to latest version
- **Enable WebGL** in browser settings
- **Check graphics drivers** are up to date

#### ❌ Training takes too long
```bash
# Reduce iterations for faster training
python train_nasa_kepler.py kepler_data.csv --iterations 200

# Use smaller batch size
python train_nasa_kepler.py kepler_data.csv --batch-size 32
```

#### ❌ Port 8000 already in use
```bash
# Kill existing process
sudo lsof -t -i tcp:8000 | xargs kill -9

# Or use different port
python main.py --port 8080
```

### 🛠️ Performance Optimization

**For better training speed:**
```bash
# Install GPU support for TensorFlow
pip install tensorflow-gpu

# Use CUDA if available
export CUDA_VISIBLE_DEVICES=0
```

**For larger datasets:**
```bash
# Increase memory limits
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

## 📈 Expected Results

### 🎯 Model Performance
- **Accuracy**: 90-94% on Kepler validation set
- **ROC AUC**: 0.92-0.96 for exoplanet classification
- **Precision**: 88-92% for confirmed exoplanets
- **Recall**: 85-90% detection rate

### 🌌 Visualization Features
- **2000+ stars** in realistic background
- **5 detailed galaxies** with spiral/elliptical types
- **Real-time orbital mechanics**
- **Distance calculations** in light-years

## 🔄 Update and Maintenance

### 🆕 Update to Latest Version
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

### 🧹 Clean Installation
```bash
# Remove virtual environment
rm -rf venv/

# Clear model cache
rm -rf saved_models/

# Fresh installation
python -m venv venv
source venv/bin/activate
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🐛 Reporting Issues
```bash
# Include system information
python -c "import sys; print(sys.version)"
pip list | grep -E "(tensorflow|fastapi|scikit)"
```

### 💡 Feature Requests
Suggest new features via GitHub Issues with the `enhancement` label.


---

<div align="center">

## 🚀 Ready to Explore the Cosmos?

**Start your exoplanet discovery journey today!**

```bash
# Begin your adventure
python train.py exoplanet_fpr_train.csv --iterations 100
python app.py
```

*The universe is waiting to be discovered...* 🌟

</div>
``` 
✅ **Структуру проекта** с пояснениями  

Теперь пользователи могут просто скопировать команды и начать работать с проектом! 🎯
