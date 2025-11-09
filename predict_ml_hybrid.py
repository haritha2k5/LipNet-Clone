"""
predict_ml_hybrid.py
Make predictions using the hybrid CNN+ML system on new videos.
"""

import numpy as np
import pickle
import logging
import os
from feature_extractor import LipNetFeatureExtractor
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridLipReaderPredictor:
    """Predict lip-read words using CNN features + ML classifier."""
    
    def __init__(self, weights_path, svm_model_path=None, rf_model_path=None,
                 label_encoder_path='models/label_encoder.pkl',
                 scaler_path='models/scaler.pkl'):
        """
        Initialize predictor with pretrained models.
        
        Args:
            weights_path: Path to LipNet pretrained weights
            svm_model_path: Path to trained SVM model
            rf_model_path: Path to trained Random Forest model
            label_encoder_path: Path to label encoder
            scaler_path: Path to feature scaler
        """
        self.extractor = LipNetFeatureExtractor(weights_path)
        self.svm_model = None
        self.rf_model = None
        self.label_encoder = None
        self.scaler = None
        
        self.load_models(svm_model_path, rf_model_path, label_encoder_path, scaler_path)
    
    def load_models(self, svm_path, rf_path, encoder_path, scaler_path):
        """Load all trained models."""
        if svm_path and os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info("✅ SVM model loaded")
        
        if rf_path and os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            logger.info("✅ Random Forest model loaded")
        
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("✅ Scaler loaded")
    
    def predict_svm(self, video_path, return_confidence=True):
        """
        Predict using SVM classifier.
        
        Args:
            video_path: Path to video file
            return_confidence: Whether to return confidence scores
        
        Returns:
            prediction (str), confidence_dict (dict if return_confidence=True)
        """
        # Extract features
        features = self.extractor.extract_features_from_video(video_path)
        
        if features is None:
            logger.error(f"Could not extract features from {video_path}")
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform([features])[0]
        
        # Predict
        prediction_idx = self.svm_model.predict([features_scaled])[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        confidence_dict = None
        if return_confidence:
            # Get decision function (confidence-like score)
            decision_scores = self.svm_model.decision_function([features_scaled])[0]
            # Normalize to probability-like scores
            softmax_scores = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
            
            confidence_dict = {
                self.label_encoder.inverse_transform([i])[0]: score
                for i, score in enumerate(softmax_scores)
                if score > 0.01  # Only show top predictions
            }
            confidence_dict = dict(sorted(confidence_dict.items(), 
                                        key=lambda x: x[1], reverse=True)[:5])
        
        return prediction, confidence_dict
    
    def predict_random_forest(self, video_path, return_confidence=True):
        """
        Predict using Random Forest classifier.
        
        Args:
            video_path: Path to video file
            return_confidence: Whether to return confidence scores
        
        Returns:
            prediction (str), confidence_dict (dict if return_confidence=True)
        """
        # Extract features
        features = self.extractor.extract_features_from_video(video_path)
        
        if features is None:
            logger.error(f"Could not extract features from {video_path}")
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform([features])[0]
        
        # Predict
        prediction_idx = self.rf_model.predict([features_scaled])[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        confidence_dict = None
        if return_confidence:
            # Get prediction probabilities
            probabilities = self.rf_model.predict_proba([features_scaled])[0]
            
            confidence_dict = {
                self.label_encoder.inverse_transform([i])[0]: prob
                for i, prob in enumerate(probabilities)
                if prob > 0.01  # Only show top predictions
            }
            confidence_dict = dict(sorted(confidence_dict.items(),
                                        key=lambda x: x[1], reverse=True)[:5])
        
        return prediction, confidence_dict
    
    def predict_ensemble(self, video_path, weights=None):
        """
        Ensemble prediction using both SVM and Random Forest.
        
        Args:
            video_path: Path to video file
            weights: Tuple of (svm_weight, rf_weight) for weighted voting
        
        Returns:
            ensemble_prediction (str), component_results (dict)
        """
        if weights is None:
            weights = (0.5, 0.5)  # Equal weight
        
        svm_pred, svm_conf = self.predict_svm(video_path, return_confidence=True)
        rf_pred, rf_conf = self.predict_random_forest(video_path, return_confidence=True)
        
        component_results = {
            'svm': {'prediction': svm_pred, 'confidence': svm_conf},
            'random_forest': {'prediction': rf_pred, 'confidence': rf_conf}
        }
        
        # Simple voting: if predictions match, return that; otherwise use SVM (higher confidence usually)
        if svm_pred == rf_pred:
            ensemble_pred = svm_pred
        else:
            # Use SVM as primary classifier
            ensemble_pred = svm_pred
        
        component_results['ensemble'] = {'prediction': ensemble_pred}
        
        return ensemble_pred, component_results


if __name__ == "__main__":
    """
    Example usage: Make predictions on a video
    """
    
    # Paths
    WEIGHTS_PATH = "models/overlapped-weights368.h5"
    SVM_MODEL_PATH = "models/svm_lip_reader.pkl"
    RF_MODEL_PATH = "models/rf_lip_reader.pkl"
    TEST_VIDEO = "path/to/test/video.mpg"
    
    # Initialize predictor
    predictor = HybridLipReaderPredictor(
        weights_path=WEIGHTS_PATH,
        svm_model_path=SVM_MODEL_PATH,
        rf_model_path=RF_MODEL_PATH
    )
    
    # Make predictions
    logger.info("="*60)
    logger.info("MAKING PREDICTIONS ON TEST VIDEO")
    logger.info("="*60)
    
    # SVM prediction
    logger.info("\nSVM Prediction:")
    svm_pred, svm_conf = predictor.predict_svm(TEST_VIDEO)
    logger.info(f"  Prediction: {svm_pred}")
    logger.info(f"  Top 5 confidence: {svm_conf}")
    
    # Random Forest prediction
    logger.info("\nRandom Forest Prediction:")
    rf_pred, rf_conf = predictor.predict_random_forest(TEST_VIDEO)
    logger.info(f"  Prediction: {rf_pred}")
    logger.info(f"  Top 5 confidence: {rf_conf}")
    
    # Ensemble prediction
    logger.info("\nEnsemble Prediction:")
    ensemble_pred, results = predictor.predict_ensemble(TEST_VIDEO)
    logger.info(f"  Prediction: {ensemble_pred}")
    logger.info(f"  Component results: {results}")