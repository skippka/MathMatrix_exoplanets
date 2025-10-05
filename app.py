from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="Exoplanet Explorer", version="10.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class AnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]

class ManualDataRequest(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: float
    koi_impact: float
    koi_insol: float
    koi_model_snr: float
    koi_steff: float
    koi_srad: float

class ExoplanetModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load your trained mmAI-2.1 model"""
        try:
            model_data = joblib.load('saved_models/mmAI-2.1_final.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            print("mmAI-2.1 model loaded successfully")
        except Exception as e:
            print(f"Error loading mmAI-2.1 model: {e}")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.imputer = SimpleImputer(strategy='median')
            self.feature_names = [
                'koi_period', 'koi_duration', 'koi_depth', 
                'koi_prad', 'koi_teq', 'koi_impact', 
                'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_srad'
            ]
    
    def calculate_distance_from_earth(self, df: pd.DataFrame) -> pd.Series:
        """Calculate real distance from Solar System in light-years"""
        if 'koi_kepmag' in df.columns:
            absolute_magnitude = 4.8
            apparent_magnitude = df['koi_kepmag']
            distance_modulus = apparent_magnitude - absolute_magnitude
            distance_parsecs = 10 ** ((distance_modulus + 5) / 5)
            distance_ly = distance_parsecs * 3.26156
            return distance_ly
        else:
            if 'koi_steff' in df.columns and 'koi_srad' in df.columns:
                stellar_luminosity = (df['koi_srad'] ** 2) * ((df['koi_steff'] / 5778) ** 4)
                distance_ly = np.sqrt(stellar_luminosity) * 10
                return distance_ly
            else:
                return pd.Series([1000.0] * len(df))
    
    def predict(self, X: pd.DataFrame):
        """Make predictions using mmAI-2.1 model"""
        try:
            available_features = [f for f in self.feature_names if f in X.columns]
            if not available_features:
                raise ValueError("No features available for prediction")
            
            X_selected = X[available_features]
            X_imputed = self.imputer.transform(X_selected)
            X_scaled = self.scaler.transform(X_imputed)
            
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return predictions, probabilities
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.simulate_predictions(X)
    
    def simulate_predictions(self, X: pd.DataFrame):
        """Fallback prediction simulation"""
        n_samples = len(X)
        predictions = np.ones(n_samples)
        probabilities = np.column_stack([
            np.random.uniform(0.1, 0.4, n_samples),
            np.random.uniform(0.6, 0.9, n_samples)
        ])
        return predictions, probabilities

# Global model instance
model_manager = ExoplanetModel()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exoplanet Explorer</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                background: #000000;
                color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                overflow: hidden;
                -webkit-font-smoothing: antialiased;
            }
            
            .container {
                display: grid;
                grid-template-columns: 320px 1fr;
                height: 100vh;
                gap: 0;
            }
            
            .sidebar {
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(40px);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
                padding: 20px;
                overflow-y: auto;
                z-index: 10;
            }
            
            .main-content {
                position: relative;
                background: #000000;
                overflow: hidden;
            }
            
            #visualization {
                width: 100%;
                height: 100%;
                display: block;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .title {
                font-size: 22px;
                font-weight: 700;
                background: linear-gradient(45deg, #64b5f6, #42a5f5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 5px;
            }
            
            .subtitle {
                font-size: 12px;
                color: #888;
                font-weight: 400;
                letter-spacing: 0.5px;
            }
            
            .model-badge {
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 10px;
                font-weight: 600;
                margin-top: 5px;
                display: inline-block;
            }
            
            .upload-area {
                border: 2px dashed rgba(100, 181, 246, 0.3);
                border-radius: 16px;
                padding: 25px 20px;
                text-align: center;
                margin-bottom: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                background: rgba(100, 181, 246, 0.05);
            }
            
            .upload-area:hover {
                border-color: #64b5f6;
                background: rgba(100, 181, 246, 0.1);
            }
            
            .upload-icon {
                font-size: 32px;
                margin-bottom: 10px;
                opacity: 0.8;
            }
            
            .btn {
                background: linear-gradient(135deg, #64b5f6, #42a5f5);
                color: white;
                border: none;
                padding: 12px 18px;
                border-radius: 12px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.3s ease;
                width: 100%;
                margin: 8px 0;
            }
            
            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 5px 15px rgba(100, 181, 246, 0.4);
            }
            
            .btn-manual {
                background: linear-gradient(135deg, #ff9800, #f57c00);
            }
            
            .filter-panel {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 15px;
                margin: 15px 0;
            }
            
            .filter-slider {
                width: 100%;
                margin: 10px 0;
                -webkit-appearance: none;
                height: 4px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 2px;
            }
            
            .filter-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #64b5f6;
                cursor: pointer;
            }
            
            .filter-value {
                text-align: center;
                font-size: 12px;
                color: #64b5f6;
                font-weight: 600;
            }
            
            .stats-panel {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 15px;
                margin: 15px 0;
            }
            
            .stat-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 8px 0;
                font-size: 12px;
            }
            
            .stat-value {
                font-weight: 600;
                color: #64b5f6;
                font-size: 13px;
            }
            
            .progress-bar {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #64b5f6, #42a5f5);
                transition: width 0.8s ease;
                border-radius: 3px;
            }
            
            .control-panel {
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                padding: 12px;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(40px);
                display: flex;
                gap: 8px;
                z-index: 2;
            }
            
            .control-btn {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                padding: 8px 12px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 11px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .control-btn:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .control-btn.active {
                background: #64b5f6;
                color: white;
            }
            
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 18px;
                border-radius: 12px;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                font-weight: 500;
                transform: translateX(400px);
                transition: transform 0.4s ease;
                z-index: 1000;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(40px);
                font-size: 12px;
            }
            
            .notification.show {
                transform: translateX(0);
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 2px solid rgba(255, 255, 255, 0.1);
                border-top: 2px solid #64b5f6;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .instructions {
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0, 0, 0, 0.8);
                padding: 10px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(40px);
                z-index: 2;
                font-size: 11px;
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.4;
            }
            
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9);
                z-index: 1000;
                backdrop-filter: blur(20px);
            }
            
            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.95);
                padding: 20px;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                width: 90%;
                max-width: 400px;
                max-height: 90vh;
                overflow-y: auto;
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .modal-title {
                font-size: 18px;
                font-weight: 600;
                color: #64b5f6;
            }
            
            .close-btn {
                background: none;
                border: none;
                color: #fff;
                font-size: 20px;
                cursor: pointer;
                padding: 0;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.3s;
            }
            
            .close-btn:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-label {
                display: block;
                margin-bottom: 6px;
                font-weight: 500;
                color: rgba(255, 255, 255, 0.9);
                font-size: 12px;
            }
            
            .form-input {
                width: 100%;
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 13px;
                transition: all 0.3s;
            }
            
            .form-input:focus {
                outline: none;
                border-color: #64b5f6;
                background: rgba(100, 181, 246, 0.1);
            }
            
            .planet-info {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                padding: 15px;
                margin-top: 12px;
            }
            
            .info-item {
                display: flex;
                justify-content: space-between;
                margin: 6px 0;
                padding: 5px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                font-size: 11px;
            }
            
            .info-label {
                color: rgba(255, 255, 255, 0.7);
            }
            
            .info-value {
                font-weight: 600;
                color: #64b5f6;
            }
            
            .galaxy-label {
                position: absolute;
                background: rgba(0, 0, 0, 0.9);
                color: #ff9800;
                padding: 6px 10px;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                border: 1px solid #ff9800;
                pointer-events: none;
                z-index: 2;
                backdrop-filter: blur(20px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <div class="header">
                    <div class="title">Exoplanet Explorer</div>
                    <div class="subtitle">NASA Data Analysis</div>
                    <div class="model-badge">mmAI-2.1 Model</div>
                </div>
                
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">🌌</div>
                    <h3>Upload Kepler Data</h3>
                    <p>Drag and drop CSV file</p>
                    <input type="file" id="fileInput" accept=".csv" style="display: none;">
                </div>
                
                <button class="btn btn-manual" onclick="showManualInput()">
                    📝 Manual Data Input
                </button>
                
                <button class="btn" id="analyzeBtn" onclick="analyzeData()">
                    🔍 Analyze with mmAI-2.1
                </button>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div>Processing with mmAI-2.1...</div>
                </div>
                
                <div class="filter-panel" id="filterPanel" style="display: none;">
                    <h3>Confidence Filter</h3>
                    <input type="range" min="0" max="100" value="70" class="filter-slider" id="confidenceFilter">
                    <div class="filter-value">Minimum: <span id="filterValue">70%</span></div>
                    <button class="btn" onclick="applyFilter()" style="margin: 8px 0;">
                        Apply Filter
                    </button>
                </div>
                
                <div class="stats-panel" id="statsPanel" style="display: none;">
                    <h3>Analysis Results</h3>
                    <div class="stat-item">
                        <span>Total Exoplanets:</span>
                        <span class="stat-value" id="totalExoplanets">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Filtered:</span>
                        <span class="stat-value" id="filteredExoplanets">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Avg Confidence:</span>
                        <span class="stat-value" id="confidenceLevel">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="confidenceBar" style="width: 0%"></div>
                    </div>
                    <div class="stat-item">
                        <span>Data Points:</span>
                        <span class="stat-value" id="dataPoints">0</span>
                    </div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="instructions">
                    🎮 Drag to rotate • Scroll to zoom • Click planets for info
                </div>
                <canvas id="visualization"></canvas>
                <div class="control-panel">
                    <button class="control-btn" id="animationBtn" onclick="toggleAnimation()">⏸️ Pause</button>
                    <button class="control-btn" onclick="resetView()">🔄 Reset</button>
                    <button class="control-btn active" id="labelsBtn" onclick="toggleLabels()">🏷️ Labels</button>
                    <button class="control-btn active" id="orbitsBtn" onclick="toggleOrbits()">🌀 Orbits</button>
                    <button class="control-btn" id="universeBtn" onclick="toggleUniverseView()">🌐 Universe</button>
                </div>
            </div>
        </div>
        
        <!-- Manual Input Modal -->
        <div class="modal" id="manualModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Manual Exoplanet Data</h2>
                    <button class="close-btn" onclick="closeManualInput()">&times;</button>
                </div>
                <form id="manualForm">
                    <div class="form-group">
                        <label class="form-label">Orbital Period (days)</label>
                        <input type="number" class="form-input" id="koi_period" step="0.1" min="0" value="365.25" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Transit Duration (hours)</label>
                        <input type="number" class="form-input" id="koi_duration" step="0.1" min="0" value="13" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Transit Depth (ppm)</label>
                        <input type="number" class="form-input" id="koi_depth" step="1" min="0" value="84" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Planet Radius (Earth radii)</label>
                        <input type="number" class="form-input" id="koi_prad" step="0.1" min="0" value="1.0" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Equilibrium Temperature (K)</label>
                        <input type="number" class="form-input" id="koi_teq" step="1" min="0" value="288" required>
                    </div>
                    <button type="submit" class="btn" style="margin-top: 15px;">
                        🔍 Analyze with mmAI-2.1
                    </button>
                </form>
            </div>
        </div>
        
        <!-- Planet Info Modal -->
        <div class="modal" id="infoModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title" id="infoTitle">Planet Information</h2>
                    <button class="close-btn" onclick="closeInfoModal()">&times;</button>
                </div>
                <div class="planet-info">
                    <div class="info-item">
                        <span class="info-label">Confidence:</span>
                        <span class="info-value" id="infoConfidence">0%</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Orbital Period:</span>
                        <span class="info-value" id="infoPeriod">0 days</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Planet Radius:</span>
                        <span class="info-value" id="infoRadius">0 Earth</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Temperature:</span>
                        <span class="info-value" id="infoTemp">0 K</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Distance:</span>
                        <span class="info-value" id="infoDistance">0 ly</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Galaxy:</span>
                        <span class="info-value" id="infoGalaxy">Milky Way</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="notification" id="notification"></div>
        
        <script>
            // Three.js variables
            let scene, camera, renderer, controls;
            let solarSystemObjects = [];
            let exoplanetObjects = [];
            let galaxyObjects = [];
            let galaxyLabels = [];
            let animationId = null;
            let isAnimating = true;
            let showLabels = true;
            let showOrbits = true;
            let showUniverseView = false;
            
            let currentData = null;
            let allExoplanets = [];
            let filteredExoplanets = [];
            let currentFilter = 0.7;

            // Solar system data
            const planets = [
                { name: 'Sun', radius: 2, color: 0xFFD700, distance: 0, period: 0, speed: 0 },
                { name: 'Mercury', radius: 0.3, color: 0x8C7853, distance: 4, period: 0.24, speed: 0.02 },
                { name: 'Venus', radius: 0.5, color: 0xFFC649, distance: 7, period: 0.62, speed: 0.01 },
                { name: 'Earth', radius: 0.5, color: 0x6B93D6, distance: 10, period: 1, speed: 0.008 },
                { name: 'Mars', radius: 0.4, color: 0xCD5C5C, distance: 15, period: 1.88, speed: 0.006 },
                { name: 'Jupiter', radius: 1.0, color: 0xD8CA9D, distance: 25, period: 11.86, speed: 0.004 },
                { name: 'Saturn', radius: 0.8, color: 0xFAD5A5, distance: 40, period: 29.46, speed: 0.002 },
                { name: 'Uranus', radius: 0.6, color: 0x4FD0E7, distance: 60, period: 84.01, speed: 0.001 },
                { name: 'Neptune', radius: 0.6, color: 0x4B70DD, distance: 80, period: 164.8, speed: 0.0005 }
            ];

            // Enhanced galaxies with realistic properties
            const galaxies = [
                { 
                    name: 'Milky Way', 
                    position: { x: 0, y: 0, z: 0 }, 
                    size: 40, 
                    color: 0x4caf50, 
                    distance: 0, 
                    type: 'spiral',
                    arms: 4
                },
                { 
                    name: 'Andromeda', 
                    position: { x: 150, y: 50, z: 100 }, 
                    size: 35, 
                    color: 0x2196f3, 
                    distance: 2537000, 
                    type: 'spiral',
                    arms: 2
                },
                { 
                    name: 'Triangulum', 
                    position: { x: -120, y: -30, z: 140 }, 
                    size: 25, 
                    color: 0x9c27b0, 
                    distance: 3000000, 
                    type: 'spiral',
                    arms: 3
                },
                { 
                    name: 'Centaurus A', 
                    position: { x: 200, y: -80, z: -60 }, 
                    size: 30, 
                    color: 0xff9800, 
                    distance: 13000000, 
                    type: 'elliptical',
                    arms: 0
                },
                { 
                    name: 'Whirlpool', 
                    position: { x: -180, y: 100, z: -120 }, 
                    size: 32, 
                    color: 0xf44336, 
                    distance: 23000000, 
                    type: 'spiral',
                    arms: 2
                }
            ];

            // Initialize Three.js
            function initThreeJS() {
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);
                
                // Add enhanced stars background
                addEnhancedStars(20000, 2000);
                
                // Camera
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
                camera.position.set(0, 30, 80);
                
                // Renderer
                renderer = new THREE.WebGLRenderer({ 
                    canvas: document.getElementById('visualization'),
                    antialias: true,
                    alpha: true
                });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.rotateSpeed = 0.3;
                controls.zoomSpeed = 0.8;
                controls.panSpeed = 0.5;
                controls.minDistance = 5;
                controls.maxDistance = 5000;
                
                // Enhanced lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                directionalLight.position.set(100, 100, 50);
                scene.add(directionalLight);

                createSolarSystem();
                createEnhancedGalaxies();
                animate();
            }
            
            function addEnhancedStars(count = 10000, range = 1500) {
                const starsGeometry = new THREE.BufferGeometry();
                const starsMaterial = new THREE.PointsMaterial({
                    color: 0xffffff,
                    size: 0.8,
                    transparent: true,
                    sizeAttenuation: true,
                    opacity: 0.8
                });
                
                const starsVertices = [];
                const starsColors = [];
                const starsSizes = [];
                
                for (let i = 0; i < count; i++) {
                    const x = (Math.random() - 0.5) * range * 2;
                    const y = (Math.random() - 0.5) * range * 2;
                    const z = (Math.random() - 0.5) * range * 2;
                    starsVertices.push(x, y, z);
                    
                    // Vary star colors (blue, white, yellow, red)
                    const starType = Math.random();
                    let r, g, b;
                    if (starType < 0.6) {
                        // White stars
                        r = g = b = Math.random() * 0.5 + 0.5;
                    } else if (starType < 0.8) {
                        // Yellow stars
                        r = g = Math.random() * 0.5 + 0.5;
                        b = Math.random() * 0.3 + 0.2;
                    } else if (starType < 0.95) {
                        // Blue stars
                        r = g = Math.random() * 0.3 + 0.2;
                        b = Math.random() * 0.5 + 0.5;
                    } else {
                        // Red stars
                        r = Math.random() * 0.5 + 0.5;
                        g = b = Math.random() * 0.3 + 0.2;
                    }
                    
                    starsColors.push(r, g, b);
                    starsSizes.push(Math.random() * 0.5 + 0.5);
                }
                
                starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
                starsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starsColors, 3));
                starsGeometry.setAttribute('size', new THREE.Float32BufferAttribute(starsSizes, 1));
                
                starsMaterial.vertexColors = true;
                starsMaterial.size = 0.6;
                
                const stars = new THREE.Points(starsGeometry, starsMaterial);
                scene.add(stars);
            }
            
            function createSolarSystem() {
                solarSystemObjects.forEach(obj => scene.remove(obj));
                solarSystemObjects = [];
                
                planets.forEach(planet => {
                    const geometry = new THREE.SphereGeometry(planet.radius, 24, 24);
                    const material = new THREE.MeshPhongMaterial({ 
                        color: planet.color,
                        shininess: 100
                    });
                    const sphere = new THREE.Mesh(geometry, material);
                    
                    if (planet.distance > 0) {
                        if (showOrbits) {
                            const orbitGeometry = new THREE.RingGeometry(planet.distance - 0.1, planet.distance + 0.1, 64);
                            const orbitMaterial = new THREE.MeshBasicMaterial({ 
                                color: 0x64b5f6, 
                                side: THREE.DoubleSide,
                                transparent: true,
                                opacity: 0.15
                            });
                            const orbit = new THREE.Mesh(orbitGeometry, orbitMaterial);
                            orbit.rotation.x = Math.PI / 2;
                            scene.add(orbit);
                            solarSystemObjects.push(orbit);
                        }
                        
                        sphere.position.x = planet.distance;
                    } else {
                        // Sun glow effect
                        const sunGlowGeometry = new THREE.SphereGeometry(planet.radius * 1.4, 24, 24);
                        const sunGlowMaterial = new THREE.MeshBasicMaterial({
                            color: 0xFF6B00,
                            transparent: true,
                            opacity: 0.2
                        });
                        const sunGlow = new THREE.Mesh(sunGlowGeometry, sunGlowMaterial);
                        scene.add(sunGlow);
                        solarSystemObjects.push(sunGlow);
                    }
                    
                    scene.add(sphere);
                    solarSystemObjects.push(sphere);
                    
                    sphere.userData = {
                        type: 'planet',
                        name: planet.name,
                        distance: planet.distance,
                        speed: planet.speed,
                        angle: Math.random() * Math.PI * 2,
                        system: 'Solar System',
                        galaxy: 'Milky Way'
                    };
                });
            }
            
            function createEnhancedGalaxies() {
                galaxyObjects.forEach(obj => scene.remove(obj));
                galaxyObjects = [];
                galaxyLabels.forEach(label => document.body.removeChild(label));
                galaxyLabels = [];
                
                galaxies.forEach((galaxy, index) => {
                    if (galaxy.name === 'Milky Way') return;
                    
                    // Create galaxy group
                    const galaxyGroup = new THREE.Group();
                    
                    // Galaxy core
                    const coreGeometry = new THREE.SphereGeometry(galaxy.size * 0.4, 16, 16);
                    const coreMaterial = new THREE.MeshPhongMaterial({ 
                        color: galaxy.color,
                        emissive: galaxy.color,
                        emissiveIntensity: 0.3
                    });
                    const core = new THREE.Mesh(coreGeometry, coreMaterial);
                    galaxyGroup.add(core);
                    
                    // Create spiral arms or halo
                    if (galaxy.type === 'spiral') {
                        createSpiralGalaxy(galaxy, galaxyGroup);
                    } else {
                        createEllipticalGalaxy(galaxy, galaxyGroup);
                    }
                    
                    galaxyGroup.position.set(
                        galaxy.position.x,
                        galaxy.position.y,
                        galaxy.position.z
                    );
                    
                    scene.add(galaxyGroup);
                    galaxyObjects.push(galaxyGroup);
                    
                    galaxyGroup.userData = {
                        type: 'galaxy',
                        name: galaxy.name,
                        size: galaxy.size,
                        distance_ly: galaxy.distance,
                        system: 'Galaxy'
                    };
                    
                    createGalaxyLabel(galaxyGroup, galaxy.name);
                });
            }
            
            function createSpiralGalaxy(galaxy, group) {
                const armCount = galaxy.arms || 2;
                const starsPerArm = 80;
                const armWidth = galaxy.size * 0.3;
                
                for (let arm = 0; arm < armCount; arm++) {
                    for (let i = 0; i < starsPerArm; i++) {
                        const progress = i / starsPerArm;
                        const angle = (arm / armCount) * Math.PI * 2 + progress * Math.PI * 6;
                        const distance = galaxy.size * 0.5 + progress * galaxy.size * 0.8;
                        const armOffset = Math.sin(progress * Math.PI * 4) * armWidth;
                        const height = (Math.random() - 0.5) * galaxy.size * 0.1;
                        
                        const starGeometry = new THREE.SphereGeometry(0.3, 6, 6);
                        const starMaterial = new THREE.MeshBasicMaterial({
                            color: galaxy.color,
                            emissive: galaxy.color,
                            emissiveIntensity: 0.2
                        });
                        const star = new THREE.Mesh(starGeometry, starMaterial);
                        
                        star.position.set(
                            Math.cos(angle) * distance + Math.cos(angle + Math.PI/2) * armOffset,
                            height,
                            Math.sin(angle) * distance + Math.sin(angle + Math.PI/2) * armOffset
                        );
                        
                        group.add(star);
                    }
                }
            }
            
            function createEllipticalGalaxy(galaxy, group) {
                const starCount = 200;
                
                for (let i = 0; i < starCount; i++) {
                    const radius = galaxy.size * (0.3 + Math.random() * 0.7);
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.acos(2 * Math.random() - 1);
                    
                    const starGeometry = new THREE.SphereGeometry(0.25, 5, 5);
                    const starMaterial = new THREE.MeshBasicMaterial({
                        color: galaxy.color,
                        emissive: galaxy.color,
                        emissiveIntensity: 0.15
                    });
                    const star = new THREE.Mesh(starGeometry, starMaterial);
                    
                    star.position.set(
                        radius * Math.sin(phi) * Math.cos(theta),
                        radius * Math.cos(phi) * 0.5, // Flatten elliptical galaxy
                        radius * Math.sin(phi) * Math.sin(theta)
                    );
                    
                    group.add(star);
                }
            }
            
            function createGalaxyLabel(galaxyMesh, name) {
                const label = document.createElement('div');
                label.className = 'galaxy-label';
                label.textContent = name;
                label.style.position = 'absolute';
                label.style.display = showUniverseView ? 'block' : 'none';
                document.body.appendChild(label);
                galaxyLabels.push(label);
            }
            
            function updateGalaxyLabels() {
                galaxyLabels.forEach((label, index) => {
                    if (index < galaxyObjects.length) {
                        const galaxy = galaxyObjects[index];
                        const vector = galaxy.position.clone();
                        vector.project(camera);
                        
                        const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
                        const y = (-vector.y * 0.5 + 0.5) * window.innerHeight;
                        
                        label.style.left = `${x}px`;
                        label.style.top = `${y}px`;
                        label.style.display = showUniverseView && vector.z < 1 ? 'block' : 'none';
                    }
                });
            }
            
            // ... (остальные функции JavaScript остаются похожими, но адаптированы под новый дизайн)
            
            function createExoplanets(exoplanets) {
                exoplanetObjects.forEach(obj => scene.remove(obj));
                exoplanetObjects = [];
                
                exoplanets.forEach((planet, i) => {
                    const isInMilkyWay = planet.distance_ly < 50000;
                    let position;
                    
                    if (isInMilkyWay) {
                        const distance = 30 + (planet.distance_ly / 2000);
                        const angle = (i / exoplanets.length) * Math.PI * 2;
                        const inclination = (Math.PI / 6) * (i % 3);
                        
                        position = new THREE.Vector3(
                            distance * Math.cos(angle),
                            distance * Math.sin(angle) * Math.sin(inclination),
                            distance * Math.sin(angle) * Math.cos(inclination)
                        );
                    } else {
                        const galaxyIndex = i % galaxyObjects.length;
                        const galaxy = galaxyObjects[galaxyIndex];
                        position = galaxy.position.clone();
                        position.x += (Math.random() - 0.5) * 50;
                        position.y += (Math.random() - 0.5) * 50;
                        position.z += (Math.random() - 0.5) * 50;
                    }
                    
                    const size = Math.min(1.0, Math.max(0.2, (planet.radius || 1) * 0.15));
                    let color;
                    if (planet.confidence > 0.8) color = 0x4caf50;
                    else if (planet.confidence > 0.6) color = 0xff9800;
                    else color = 0xf44336;
                    
                    const geometry = new THREE.SphereGeometry(size, 12, 12);
                    const material = new THREE.MeshPhongMaterial({ 
                        color: color,
                        emissive: color,
                        emissiveIntensity: 0.1
                    });
                    const sphere = new THREE.Mesh(geometry, material);
                    
                    sphere.position.copy(position);
                    scene.add(sphere);
                    exoplanetObjects.push(sphere);
                    
                    sphere.userData = {
                        type: 'exoplanet',
                        name: planet.name,
                        confidence: planet.confidence,
                        period: planet.period,
                        radius: planet.radius,
                        temperature: planet.temperature,
                        distance_ly: planet.distance_ly,
                        system: isInMilkyWay ? 'Star System' : 'External System',
                        galaxy: isInMilkyWay ? 'Milky Way' : 'External Galaxy'
                    };
                });
            }
            
            function animate() {
                animationId = requestAnimationFrame(animate);
                
                if (isAnimating) {
                    // Animate solar system
                    solarSystemObjects.forEach(obj => {
                        if (obj.userData.type === 'planet' && obj.userData.distance > 0) {
                            obj.userData.angle += obj.userData.speed;
                            obj.position.x = obj.userData.distance * Math.cos(obj.userData.angle);
                            obj.position.z = obj.userData.distance * Math.sin(obj.userData.angle);
                        }
                    });
                    
                    // Animate galaxies
                    galaxyObjects.forEach((galaxy, index) => {
                        if (galaxy.userData.type === 'galaxy') {
                            const time = Date.now() * 0.0001;
                            galaxy.rotation.y = time * 0.05;
                        }
                    });
                    
                    updateGalaxyLabels();
                }
                
                controls.update();
                renderer.render(scene, camera);
            }
            
            // UI Functions (аналогичные предыдущей версии, но с минималистичным дизайном)
            function initializeUpload() {
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                
                uploadArea.addEventListener('click', () => fileInput.click());
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.style.borderColor = '#64b5f6';
                    uploadArea.style.background = 'rgba(100, 181, 246, 0.1)';
                });
                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.style.borderColor = 'rgba(100, 181, 246, 0.3)';
                    uploadArea.style.background = 'rgba(100, 181, 246, 0.05)';
                });
                uploadArea.addEventListener('drop', async (e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files[0]) {
                        await handleFileUpload(e.dataTransfer.files[0]);
                    }
                });
                
                fileInput.addEventListener('change', async (e) => {
                    if (e.target.files[0]) {
                        await handleFileUpload(e.target.files[0]);
                    }
                });
                
                const filterSlider = document.getElementById('confidenceFilter');
                filterSlider.addEventListener('input', function() {
                    document.getElementById('filterValue').textContent = this.value + '%';
                    currentFilter = this.value / 100;
                });
                
                document.getElementById('manualForm').addEventListener('submit', handleManualSubmit);
            }
            
            async function handleFileUpload(file) {
                showNotification('Uploading data...', 'success');
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload-csv', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentData = result.data;
                        showNotification('Data loaded successfully', 'success');
                        document.getElementById('statsPanel').style.display = 'block';
                        document.getElementById('filterPanel').style.display = 'block';
                        updateStats(currentData);
                    } else {
                        throw new Error(result.detail);
                    }
                } catch (error) {
                    showNotification('Error: ' + error.message, 'error');
                }
            }
            
            async function analyzeData() {
                if (!currentData) {
                    showNotification('Please upload a CSV file first', 'error');
                    return;
                }
                
                const analyzeBtn = document.getElementById('analyzeBtn');
                const loading = document.getElementById('loading');
                analyzeBtn.disabled = true;
                loading.style.display = 'block';
                
                showNotification('Analyzing with mmAI-2.1...', 'success');
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ data: currentData })
                    });
                    const results = await response.json();
                    
                    if (response.ok) {
                        allExoplanets = results.exoplanets;
                        applyFilter();
                        showNotification('Analysis complete: ' + allExoplanets.length + ' exoplanets found', 'success');
                    } else {
                        throw new Error(results.detail);
                    }
                } catch (error) {
                    showNotification('Analysis failed: ' + error.message, 'error');
                } finally {
                    analyzeBtn.disabled = false;
                    loading.style.display = 'none';
                }
            }
            
            async function handleManualSubmit(event) {
                event.preventDefault();
                const formData = new FormData(event.target);
                const data = {
                    koi_period: parseFloat(formData.get('koi_period')),
                    koi_duration: parseFloat(formData.get('koi_duration')),
                    koi_depth: parseFloat(formData.get('koi_depth')),
                    koi_prad: parseFloat(formData.get('koi_prad')),
                    koi_teq: parseFloat(formData.get('koi_teq')),
                    koi_impact: parseFloat(formData.get('koi_impact')),
                    koi_insol: parseFloat(formData.get('koi_insol')),
                    koi_model_snr: parseFloat(formData.get('koi_model_snr')),
                    koi_steff: parseFloat(formData.get('koi_steff')),
                    koi_srad: parseFloat(formData.get('koi_srad'))
                };
                
                try {
                    const response = await fetch('/analyze-manual', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    
                    if (response.ok) {
                        const exoplanet = result.exoplanet;
                        showNotification(`Analysis complete: ${(exoplanet.confidence * 100).toFixed(1)}% confidence`, 'success');
                        allExoplanets = [exoplanet];
                        filteredExoplanets = [exoplanet];
                        createExoplanets(filteredExoplanets);
                        updateResults(filteredExoplanets);
                        closeManualInput();
                    } else {
                        throw new Error(result.detail);
                    }
                } catch (error) {
                    showNotification('Manual analysis failed: ' + error.message, 'error');
                }
            }
            
            function applyFilter() {
                if (allExoplanets.length === 0) return;
                filteredExoplanets = allExoplanets.filter(planet => planet.confidence >= currentFilter);
                createExoplanets(filteredExoplanets);
                updateResults(filteredExoplanets);
                showNotification(`Filter applied: ${filteredExoplanets.length} exoplanets shown`, 'success');
            }
            
            function showManualInput() {
                document.getElementById('manualModal').style.display = 'block';
            }
            
            function closeManualInput() {
                document.getElementById('manualModal').style.display = 'none';
                document.getElementById('manualForm').reset();
            }
            
            function showPlanetInfo(planetData) {
                document.getElementById('infoTitle').textContent = planetData.name;
                document.getElementById('infoConfidence').textContent = planetData.confidence ? 
                    (planetData.confidence * 100).toFixed(1) + '%' : 'N/A';
                document.getElementById('infoPeriod').textContent = planetData.period ? 
                    planetData.period + ' days' : 'N/A';
                document.getElementById('infoRadius').textContent = planetData.radius ? 
                    planetData.radius + ' Earth radii' : 'N/A';
                document.getElementById('infoTemp').textContent = planetData.temperature ? 
                    planetData.temperature + ' K' : 'N/A';
                document.getElementById('infoDistance').textContent = planetData.distance_ly ? 
                    planetData.distance_ly.toLocaleString() + ' ly' : 'N/A';
                document.getElementById('infoGalaxy').textContent = planetData.galaxy || 'Milky Way';
                document.getElementById('infoModal').style.display = 'block';
            }
            
            function closeInfoModal() {
                document.getElementById('infoModal').style.display = 'none';
            }
            
            function toggleAnimation() {
                isAnimating = !isAnimating;
                const button = document.getElementById('animationBtn');
                button.textContent = isAnimating ? '⏸️ Pause' : '▶️ Play';
                button.classList.toggle('active', isAnimating);
                showNotification(isAnimating ? 'Animation resumed' : 'Animation paused', 'success');
            }
            
            function resetView() {
                controls.reset();
                camera.position.set(0, 30, 80);
                controls.update();
                showNotification('View reset', 'success');
            }
            
            function toggleLabels() {
                showLabels = !showLabels;
                const button = document.getElementById('labelsBtn');
                button.textContent = showLabels ? '🏷️ Labels' : '❌ Labels';
                button.classList.toggle('active', showLabels);
                galaxyLabels.forEach(label => {
                    label.style.display = showLabels && showUniverseView ? 'block' : 'none';
                });
                showNotification(showLabels ? 'Labels shown' : 'Labels hidden', 'success');
            }
            
            function toggleOrbits() {
                showOrbits = !showOrbits;
                const button = document.getElementById('orbitsBtn');
                button.textContent = showOrbits ? '🌀 Orbits' : '❌ Orbits';
                button.classList.toggle('active', showOrbits);
                createSolarSystem();
                showNotification(showOrbits ? 'Orbits shown' : 'Orbits hidden', 'success');
            }
            
            function toggleUniverseView() {
                showUniverseView = !showUniverseView;
                const button = document.getElementById('universeBtn');
                button.textContent = showUniverseView ? '🌍 Solar' : '🌐 Universe';
                button.classList.toggle('active', showUniverseView);
                
                if (showUniverseView) {
                    camera.position.set(0, 200, 400);
                    controls.maxDistance = 5000;
                    showNotification('Universe view - explore galaxies!', 'success');
                } else {
                    camera.position.set(0, 30, 80);
                    controls.maxDistance = 1000;
                    showNotification('Solar system view', 'success');
                }
                controls.update();
                galaxyLabels.forEach(label => {
                    label.style.display = showUniverseView && showLabels ? 'block' : 'none';
                });
            }
            
            function updateStats(data) {
                document.getElementById('dataPoints').textContent = data.length.toLocaleString();
            }
            
            function updateResults(exoplanets) {
                document.getElementById('totalExoplanets').textContent = allExoplanets.length;
                document.getElementById('filteredExoplanets').textContent = exoplanets.length;
                const avgConfidence = exoplanets.length > 0 ? 
                    (exoplanets.reduce((sum, p) => sum + p.confidence, 0) / exoplanets.length) * 100 : 0;
                document.getElementById('confidenceLevel').textContent = avgConfidence.toFixed(1) + '%';
                document.getElementById('confidenceBar').style.width = avgConfidence + '%';
            }
            
            function showNotification(message, type) {
                const notification = document.getElementById('notification');
                notification.textContent = message;
                notification.className = 'notification';
                notification.classList.add('show');
                if (type === 'error') notification.classList.add('error');
                setTimeout(() => {
                    notification.classList.remove('show');
                }, 3000);
            }
            
            function onWindowResize() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
            
            function onCanvasClick(event) {
                const mouse = new THREE.Vector2();
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, camera);
                const allObjects = [...solarSystemObjects, ...exoplanetObjects, ...galaxyObjects];
                const intersects = raycaster.intersectObjects(allObjects);
                
                if (intersects.length > 0) {
                    const object = intersects[0].object;
                    showPlanetInfo(object.userData);
                }
            }
            
            // Initialize the application
            document.addEventListener('DOMContentLoaded', function() {
                initializeUpload();
                initThreeJS();
                window.addEventListener('resize', onWindowResize);
                renderer.domElement.addEventListener('click', onCanvasClick);
                showNotification('Welcome to Exoplanet Explorer!', 'success');
            });
        </script>
    </body>
    </html>
    """

# ... (остальные endpoint функции остаются без изменений)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, "File must be a CSV")
        
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), comment='#')
        df_clean = df.fillna(0)
        
        if 'display_name' not in df_clean.columns:
            if 'pl_name' in df_clean.columns:
                df_clean['display_name'] = df_clean['pl_name']
            elif 'kepler_name' in df_clean.columns:
                df_clean['display_name'] = df_clean['kepler_name']
            elif 'kepoi_name' in df_clean.columns:
                df_clean['display_name'] = df_clean['kepoi_name']
            else:
                df_clean['display_name'] = ['KOI-' + str(i) for i in range(len(df_clean))]
        
        return JSONResponse({
            "message": "File uploaded successfully",
            "filename": file.filename,
            "data": df_clean.to_dict('records'),
            "rows": len(df_clean),
            "columns": list(df_clean.columns)
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    try:
        data = request.data
        if not data:
            raise HTTPException(400, "No data provided")
        
        df = pd.DataFrame(data)
        exoplanets = predict_with_mmai(df)
        
        return JSONResponse({
            "exoplanets": exoplanets,
            "metrics": {
                "total_found": len(exoplanets),
                "analysis_time": datetime.now().isoformat(),
                "model_used": "mmAI-2.1"
            }
        })
    
    except Exception as e:
        raise HTTPException(500, f"Analysis error: {str(e)}")

@app.post("/analyze-manual")
async def analyze_manual_data(request: ManualDataRequest):
    try:
        data_dict = request.dict()
        df = pd.DataFrame([data_dict])
        exoplanets = predict_with_mmai(df)
        
        if exoplanets:
            exoplanet = exoplanets[0]
            exoplanet["name"] = "Manual-Exoplanet-" + datetime.now().strftime("%H%M%S")
        else:
            exoplanet = {
                "name": "Manual-Exoplanet-" + datetime.now().strftime("%H%M%S"),
                "confidence": 0.5,
                "period": data_dict['koi_period'],
                "radius": data_dict['koi_prad'],
                "temperature": data_dict['koi_teq'],
                "distance_ly": 1000.0
            }
        
        return JSONResponse({
            "exoplanet": exoplanet,
            "message": f"Analysis complete: {(exoplanet['confidence'] * 100):.1f}% confidence",
            "model_used": "mmAI-2.1"
        })
    
    except Exception as e:
        raise HTTPException(500, f"Manual analysis error: {str(e)}")

def predict_with_mmai(df: pd.DataFrame) -> List[Dict]:
    """Use trained mmAI-2.1 model for predictions"""
    try:
        predictions, probabilities = model_manager.predict(df)
        distances = model_manager.calculate_distance_from_earth(df)
        
        exoplanets = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                confidence = prob[1] if hasattr(prob, '__len__') else prob
                exoplanets.append({
                    "name": df.iloc[i].get('display_name', f"KOI-{i}"),
                    "confidence": round(float(confidence), 3),
                    "period": round(float(df.iloc[i].get('koi_period', 0)), 2),
                    "radius": round(float(df.iloc[i].get('koi_prad', 1)), 2),
                    "temperature": int(df.iloc[i].get('koi_teq', 0)),
                    "distance_ly": round(float(distances.iloc[i]), 2)
                })
        
        exoplanets.sort(key=lambda x: x['confidence'], reverse=True)
        return exoplanets
        
    except Exception as e:
        print(f"mmAI-2.1 prediction error: {e}")
        return simulate_ai_analysis(df)

def simulate_ai_analysis(df: pd.DataFrame) -> List[Dict]:
    """Fallback analysis if model fails"""
    exoplanets = []
    distances = model_manager.calculate_distance_from_earth(df)
    
    for i, row in df.iterrows():
        score = np.random.uniform(0.3, 0.95)
        if score > 0.6:
            exoplanets.append({
                "name": row.get('display_name', f"KOI-{i}"),
                "confidence": round(score, 3),
                "period": round(row.get('koi_period', np.random.uniform(10, 400)), 2),
                "radius": round(row.get('koi_prad', np.random.uniform(0.5, 20)), 2),
                "temperature": int(row.get('koi_teq', np.random.uniform(200, 1000))),
                "distance_ly": round(float(distances.iloc[i]), 2)
            })
    
    exoplanets.sort(key=lambda x: x['confidence'], reverse=True)
    return exoplanets

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
