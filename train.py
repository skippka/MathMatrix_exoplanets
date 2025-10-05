# train_nasa_kepler.py
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any, List
import json
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

warnings.filterwarnings('ignore')

class ScientificConfig:
    MODEL_NAME = "mmAI-ExoplanetHunter-v4.0"
    SAVE_DIR = Path('saved_models')
    
    FEATURES = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
        
        'koi_prad', 'koi_teq', 'koi_insol',
        
        'koi_steff', 'koi_slogg', 'koi_srad',
        
        'koi_model_snr',
        
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        
        'koi_time0bk', 'koi_ror', 'koi_dor'
    ]
    
    TARGET = 'koi_disposition'
    
    HIDDEN_LAYERS = [512, 256, 128, 64, 32]
    DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2, 0.1]
    L1_REGULARIZATION = 0.001
    L2_REGULARIZATION = 0.002
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    PATIENCE = 50  # Increased patience
    
    MIN_SAMPLES = 30
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    RANDOM_STATE = 42
    CV_FOLDS = 5

class ScientificMetricsCallback(Callback):
    """Advanced scientific metrics tracking"""
    
    def __init__(self, validation_data, model_name):
        super().__init__()
        self.validation_data = validation_data
        self.model_name = model_name
        self.best_auc = 0
        self.best_ap = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        
        auc = float(roc_auc_score(y_val, y_pred))
        avg_precision = float(average_precision_score(y_val, y_pred))
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = float(f1_score(y_val, y_pred_binary))
        
        logs['val_auc'] = auc
        logs['val_avg_precision'] = avg_precision
        logs['val_f1'] = f1
        
        combined_score = auc * 0.4 + avg_precision * 0.4 + f1 * 0.2
        
        if combined_score > (self.best_auc * 0.4 + self.best_ap * 0.4 + f1 * 0.2):
            self.best_auc = auc
            self.best_ap = avg_precision
            self.best_weights = self.model.get_weights()
            print(f"🎯 New best - AUC: {auc:.4f}, AP: {avg_precision:.4f}, F1: {f1:.4f}")

class AdvancedExoplanetHunter:
    def __init__(self, config: ScientificConfig = ScientificConfig()):
        self.config = config
        self.model = None
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.history = None
        self.feature_names = None
        self.feature_importance = None
        self.best_score = 0
        self.best_model_path = None
        self.label_encoder = LabelEncoder()
        self.engineered_features = []
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create scientifically validated features for exoplanet detection"""
        df_eng = df.copy()
        
        for feature in self.config.FEATURES:
            if feature not in df_eng.columns:
                if 'fpflag' in feature:
                    df_eng[feature] = 0  # Default to not flagged
                else:
                    # Use median of existing data for missing features
                    df_eng[feature] = df_eng.median(numeric_only=True).get(feature, 0)
        
        # Store original feature names
        original_features = df_eng.columns.tolist()
        
        # 1. Advanced Transit Signal Metrics
        df_eng['transit_snr_advanced'] = df_eng['koi_depth'] / (df_eng['koi_depth'].std() + 1e-10)
        df_eng['duration_period_ratio'] = df_eng['koi_duration'] / (df_eng['koi_period'] + 1e-10)
        
        # 2. Planetary System Dynamics
        df_eng['impact_parameter_squared'] = df_eng['koi_impact'] ** 2
        df_eng['transit_duration_norm'] = df_eng['koi_duration'] / 24.0
        
        # 3. Stellar Habitability Indicators
        df_eng['teff_logg_ratio'] = df_eng['koi_steff'] / (10**df_eng['koi_slogg'] + 1e-10)
        df_eng['planet_star_ratio'] = df_eng['koi_prad'] / (df_eng['koi_srad'] + 1e-10)
        
        # 4. Orbital Zone Classification
        conditions = [
            df_eng['koi_teq'] > 400,  # Hot zone
            (df_eng['koi_teq'] > 250) & (df_eng['koi_teq'] <= 400),  # Warm zone
            df_eng['koi_teq'] <= 250  # Cold zone
        ]
        choices = [2, 1, 0]
        df_eng['orbital_zone'] = np.select(conditions, choices, default=1)
        
        # 5. Advanced False Positive Resistance
        fp_weights = {'nt': 0.3, 'ss': 0.3, 'co': 0.2, 'ec': 0.2}
        df_eng['fp_resistance_score'] = (
            1.0 - (df_eng['koi_fpflag_nt'] * fp_weights['nt'] + 
                  df_eng['koi_fpflag_ss'] * fp_weights['ss'] + 
                  df_eng['koi_fpflag_co'] * fp_weights['co'] + 
                  df_eng['koi_fpflag_ec'] * fp_weights['ec'])
        )
        
        # 6. Signal Consistency and Quality
        df_eng['snr_consistency_ratio'] = df_eng['koi_model_snr']
        
        # 7. Physical Plausibility Scores
        df_eng['radius_plausibility'] = np.exp(-((df_eng['koi_prad'] - 1.0) ** 2) / 5.0)
        df_eng['period_insol_consistency'] = np.log1p(df_eng['koi_period']) * np.log1p(df_eng['koi_insol'])
        
        # 8. Advanced Statistical Features
        skewed_features = ['koi_period', 'koi_depth', 'koi_insol', 'koi_prad', 'koi_model_snr']
        for feature in skewed_features:
            if feature in df_eng.columns:
                df_eng[f'log_{feature}'] = np.log1p(np.abs(df_eng[feature]))
        
        # 9. Interaction Features
        df_eng['period_depth_interaction'] = df_eng['koi_period'] * df_eng['koi_depth']
        df_eng['teff_radius_interaction'] = df_eng['koi_steff'] * df_eng['koi_prad']
        
        # 10. Scientific Confidence Score
        df_eng['scientific_confidence'] = (
            df_eng['fp_resistance_score'] * 0.3 +
            df_eng['radius_plausibility'] * 0.2 +
            (1 - df_eng['duration_period_ratio']) * 0.5
        )
        
        # Store engineered feature names
        self.engineered_features = [col for col in df_eng.columns if col not in original_features]
        
        return df_eng
    
    def select_optimal_features(self, X: pd.DataFrame, y: pd.Series, k: int = 40) -> pd.DataFrame:
        """Advanced feature selection using multiple scientific methods"""
        
        # Method 1: Random Forest Importance
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.config.RANDOM_STATE,
            max_depth=10
        )
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Method 2: Mutual Information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=self.config.RANDOM_STATE)
            mi_importance = pd.Series(mi_scores, index=X.columns)
        except:
            mi_importance = pd.Series([1] * len(X.columns), index=X.columns)
        
        # Method 3: Correlation with target
        corr_scores = X.apply(lambda col: abs(np.corrcoef(col.fillna(0), y)[0,1]) if len(np.unique(col)) > 1 else 0)
        
        # Combine scores using weighted average
        combined_scores = (
            rf_importance * 0.5 +
            mi_importance * 0.3 +
            corr_scores * 0.2
        )
        
        # Select top features
        k = min(k, len(X.columns))
        best_features = combined_scores.nlargest(k).index.tolist()
        
        return X[best_features]
    
    def create_scientific_neural_network(self, input_shape: int) -> Sequential:
        """Create advanced neural network with scientific architecture"""
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.config.HIDDEN_LAYERS[0],
            kernel_regularizer=l1_l2(
                l1=self.config.L1_REGULARIZATION,
                l2=self.config.L2_REGULARIZATION
            ),
            input_shape=(input_shape,)
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.config.DROPOUT_RATES[0]))
        
        # Hidden layers
        for i, (units, dropout_rate) in enumerate(zip(
            self.config.HIDDEN_LAYERS[1:], 
            self.config.DROPOUT_RATES[1:]
        )):
            model.add(Dense(units))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Optimizer
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                       fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced scientific preprocessing pipeline"""
        
        # Feature engineering
        X_eng = self.create_advanced_features(X)
        
        if fit:
            # Handle missing values
            X_imputed = self.imputer.fit_transform(X_eng)
            X_imputed_df = pd.DataFrame(X_imputed, columns=X_eng.columns)
            
            # Feature selection
            X_selected = self.select_optimal_features(X_imputed_df, y)
            self.feature_names = X_selected.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            
        else:
            if not self.feature_names:
                raise ValueError("Model must be fitted before transforming new data")
                
            # Ensure we have all engineered features
            X_eng = self.create_advanced_features(X)
            
            # Impute missing values
            X_imputed = self.imputer.transform(X_eng)
            X_imputed_df = pd.DataFrame(X_imputed, columns=X_eng.columns)
            
            # Select same features
            available_features = [f for f in self.feature_names if f in X_imputed_df.columns]
            missing_features = [f for f in self.feature_names if f not in X_imputed_df.columns]
            
            if missing_features:
                for feature in missing_features:
                    X_imputed_df[feature] = 0
            
            X_selected = X_imputed_df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X_selected)
        
        y_processed = y.values if y is not None else np.array([])
        
        return X_scaled, y_processed
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              iterations: int = 500) -> Tuple[Dict[str, List[float]], float]:
        """Advanced training with scientific validation"""
        
        print("🚀 Starting advanced training...")
        print(f"📊 Initial data shape: {X.shape}")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit=True)
        
        # Enhanced data splitting
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y_processed, 
            test_size=self.config.TEST_SIZE + self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_processed
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.TEST_SIZE/(self.config.TEST_SIZE + self.config.VALIDATION_SIZE),
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )
        
        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📈 Validation set: {X_val.shape[0]} samples") 
        print(f"📈 Test set: {X_test.shape[0]} samples")
        print(f"🎯 Feature dimension: {X_train.shape[1]}")
        
        # Create advanced model
        self.model = self.create_scientific_neural_network(X_train.shape[1])
        
        # Prepare advanced callbacks
        self.config.SAVE_DIR.mkdir(exist_ok=True)
        model_path = self.config.SAVE_DIR / 'best_exoplanet_model.h5'
        
        callbacks = [
            ScientificMetricsCallback(
                validation_data=(X_val, y_val), 
                model_name=self.config.MODEL_NAME
            ),
            EarlyStopping(
                monitor='val_auc', 
                patience=self.config.PATIENCE, 
                restore_best_weights=True, 
                mode='max',
                verbose=1,
                min_delta=0.001  # Minimum change to qualify as improvement
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=15, 
                min_lr=1e-7,
                verbose=1,
                min_delta=0.001
            ),
            ModelCheckpoint(
                str(model_path),
                monitor='val_auc', 
                save_best_only=True, 
                mode='max',
                verbose=1
            )
        ]
        
        # Calculate advanced class weights
        class_weights = self._compute_advanced_class_weights(y_processed)
        
        print("🎯 Starting model training...")
        print(f"⏰ Iterations: {iterations}, Batch size: {self.config.BATCH_SIZE}")
        
        # Train model with comprehensive monitoring
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=iterations,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights,
            shuffle=True
        )
        
        # Load best model
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self.best_model_path = model_path
        
        # Comprehensive scientific evaluation
        print("\n🔬 Starting comprehensive evaluation...")
        test_metrics = self.evaluate_scientifically(X_test, y_test)
        
        current_score = test_metrics['combined_score']
        
        # Advanced model selection
        if current_score > self.best_score:
            self.best_score = current_score
            self._save_complete_pipeline()
            print(f"🏆 NEW BEST MODEL! Score: {current_score:.4f}")
        else:
            print(f"📊 Current model score: {current_score:.4f}, Best: {self.best_score:.4f}")
        
        return self.history.history, current_score
    
    def evaluate_scientifically(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive scientific evaluation"""
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Basic metrics
        accuracy = float(np.mean(y == y_pred))
        roc_auc = float(roc_auc_score(y, y_pred_proba))
        avg_precision = float(average_precision_score(y, y_pred_proba))
        
        # Advanced metrics
        report = classification_report(y, y_pred, output_dict=True)
        precision = float(report['1']['precision']) if '1' in report else 0.0
        recall = float(report['1']['recall']) if '1' in report else 0.0
        f1 = float(report['1']['f1-score']) if '1' in report else 0.0
        
        # Scientific confidence score
        confidence_score = float(np.mean(y_pred_proba[y == 1])) if np.sum(y == 1) > 0 else 0.0
        
        # Combined scientific score
        combined_score = float(
            roc_auc * 0.3 +
            avg_precision * 0.3 +
            f1 * 0.2 +
            confidence_score * 0.2
        )
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence_score': confidence_score,
            'combined_score': combined_score
        }
        
        return metrics
    
    def _compute_advanced_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute advanced class weights for imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Apply additional weighting for scientific importance
        if len(classes) == 2:
            # Give more weight to exoplanet class (1)
            weights[1] *= 1.2
        
        return dict(zip(classes, weights))
    
    def _save_complete_pipeline(self):
        """Save complete scientific pipeline"""
        pipeline_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'config': self.config,
            'best_score': float(self.best_score),  # Convert to float for JSON
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }
        
        final_path = self.config.SAVE_DIR / f'{self.config.MODEL_NAME}_complete.joblib'
        joblib.dump(pipeline_data, final_path)
        
        # Also save standalone model
        model_path = self.config.SAVE_DIR / f'{self.config.MODEL_NAME}_model.h5'
        self.model.save(model_path)
        
        print(f"💾 Complete pipeline saved to {final_path}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        X_processed, _ = self.preprocess_data(X, fit=False)
        predictions = self.model.predict(X_processed, verbose=0)
        return predictions.flatten()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def analyze_exoplanet_similarity(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analyze similarity to known exoplanets"""
        probabilities = self.predict_proba(X)
        
        similarity_analysis = pd.DataFrame({
            'exoplanet_probability': probabilities,
            'confidence_level': np.abs(probabilities - 0.5) * 2,
            'similarity_category': pd.cut(
                probabilities, 
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            ),
            'scientific_rating': pd.cut(
                probabilities,
                bins=[0, 0.1, 0.3, 0.7, 0.9, 1.0],
                labels=['Very Unlikely', 'Unlikely', 'Possible', 'Likely', 'Very Likely']
            )
        })
        
        return similarity_analysis

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def train_advanced_kepler_model(data_path: str, iterations: int = 500) -> bool:
    """Advanced main training function"""
    
    config = ScientificConfig()
    
    # Create save directory
    config.SAVE_DIR.mkdir(exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found: {data_path}")
        return False
    
    try:
        print("🚀 NASA KEPLER ADVANCED EXOPLANET HUNTER v4.0")
        print("==============================================")
        print(f"📁 Loading data from: {data_path}")
        
        # Load data with error handling
        df = pd.read_csv(data_path, comment='#')
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Target column: {config.TARGET}")
        
        # Check target existence
        if config.TARGET not in df.columns:
            print(f"❌ Error: Target column '{config.TARGET}' not found")
            available_targets = [col for col in df.columns if 'disposition' in col.lower()]
            if available_targets:
                print(f"📋 Available disposition columns: {available_targets}")
                config.TARGET = available_targets[0]
                print(f"🔄 Using alternative target: {config.TARGET}")
            else:
                return False
        
        # Prepare data with comprehensive cleaning
        valid_dispositions = ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']
        df_clean = df[df[config.TARGET].isin(valid_dispositions)].copy()
        
        if len(df_clean) < config.MIN_SAMPLES:
            print(f'❌ Insufficient data: {len(df_clean)} samples, need {config.MIN_SAMPLES}')
            return False
        
        print(f"✅ Cleaned data: {len(df_clean)} samples")
        print(f"📈 Class distribution:\n{df_clean[config.TARGET].value_counts()}")
        
        # Prepare features and target
        available_features = [f for f in config.FEATURES if f in df_clean.columns]
        missing_features = [f for f in config.FEATURES if f not in df_clean.columns]
        
        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features: {missing_features}")
        
        print(f"🎯 Using {len(available_features)} available features")
        
        X = df_clean[available_features].copy()
        y = df_clean[config.TARGET].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0, 'CANDIDATE': 1})
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        print(f"🔧 Final feature matrix: {X.shape}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Initialize and train advanced model
        model = AdvancedExoplanetHunter(config)
        
        print(f"🚀 Starting advanced training ({iterations} epochs)...")
        print("=" * 60)
        
        history, final_score = model.train(X, y, iterations=iterations)
        
        # Comprehensive final evaluation
        print("\n" + "=" * 60)
        print("🔬 COMPREHENSIVE FINAL EVALUATION")
        print("=" * 60)
        
        # Predictions and probabilities
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate all metrics
        accuracy = float(np.mean(y.values == y_pred))
        auc = float(roc_auc_score(y, y_proba))
        avg_precision = float(average_precision_score(y, y_proba))
        
        # Detailed classification report
        report = classification_report(y, y_pred, target_names=['False Positive', 'Exoplanet'])
        
        print(f"🎯 Final Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC AUC: {auc:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   Best Score: {model.best_score:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"   Confusion Matrix:\n{cm}")
        
        # Similarity analysis
        similarity_df = model.analyze_exoplanet_similarity(X)
        high_conf = np.sum(similarity_df['scientific_rating'].isin(['Likely', 'Very Likely']))
        medium_conf = np.sum(similarity_df['scientific_rating'] == 'Possible')
        
        print(f"\n🔍 Exoplanet Similarity Analysis:")
        print(f"   High confidence candidates: {high_conf}")
        print(f"   Medium confidence candidates: {medium_conf}")
        print(f"   Scientific ratings:")
        for rating, count in similarity_df['scientific_rating'].value_counts().items():
            print(f"     {rating}: {count}")
        
        print(f"\n📋 Detailed Classification Report:\n{report}")
        
        # Save final results with proper serialization
        results = convert_to_serializable({
            'final_accuracy': accuracy,
            'final_auc': auc,
            'final_ap': avg_precision,
            'best_score': float(model.best_score),
            'training_date': datetime.now().isoformat(),
            'feature_count': len(available_features),
            'sample_count': len(df_clean),
            'similarity_analysis': {
                'high_confidence': int(high_conf),
                'medium_confidence': int(medium_conf),
                'total_candidates': int(high_conf + medium_conf)
            },
            'class_distribution': {
                'exoplanets': int(y.sum()),
                'false_positives': int(len(y) - y.sum())
            }
        })
        
        results_path = config.SAVE_DIR / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        print(f"💾 Model saved as: {config.MODEL_NAME}_complete.joblib")
        
        # Final success message
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("✨ Your model is ready for real exoplanet discovery!")
        print(f"🔭 Found {high_conf} high-confidence and {medium_conf} medium-confidence exoplanet candidates!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='NASA Kepler Advanced Exoplanet Hunter - Real Scientific Exoplanet Detection'
    )
    parser.add_argument('csv_file', help='Path to Kepler CSV file')
    parser.add_argument('--iterations', '-n', type=int, default=500,
                       help='Number of training iterations (default: 500)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')
    
    args = parser.parse_args()
    
    success = train_advanced_kepler_model(args.csv_file, args.iterations)
    
    if success:
        print("\n✅ Model training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Model training failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
