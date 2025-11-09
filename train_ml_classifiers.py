"""
train_ml_classifiers.py - FIXED with proper train/test split
"""

import numpy as np
import pickle
import logging
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import LipNetFeatureExtractor, FeatureDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDatasetFixed(FeatureDataset):
    """Fixed - extracts SPEAKER label with proper train/test split"""
    
    def extract_from_directory_with_split(self, base_path, speakers, split_ratio=0.75):
        """
        Extract from directory with SPEAKER labels.
        Splits each speaker's videos into train/test.
        """
        all_features = []
        all_labels = []
        all_speakers_list = []
        
        for speaker in speakers:
            speaker_path = os.path.join(base_path, speaker)
            if not os.path.exists(speaker_path):
                logger.warning(f"Not found: {speaker_path}")
                continue
            
            logger.info(f"\n📹 Processing: {speaker}")
            
            # Find videos
            video_files = sorted([f for f in os.listdir(speaker_path) 
                                 if f.endswith(('.mpg', '.mp4', '.avi', '.mov'))])
            
            logger.info(f"   Found {len(video_files)} videos")
            
            if len(video_files) == 0:
                continue
            
            processed = 0
            skipped = 0
            
            for idx, video_file in enumerate(video_files):
                video_path = os.path.join(speaker_path, video_file)
                
                try:
                    features = self.extractor.extract_features_from_video(video_path)
                    
                    if features is not None and len(features) > 0:
                        all_features.append(features)
                        all_labels.append(speaker)
                        all_speakers_list.append(speaker)
                        processed += 1
                    else:
                        skipped += 1
                except:
                    skipped += 1
                
                if (idx + 1) % max(1, len(video_files) // 5) == 0:
                    logger.info(f"   {idx + 1}/{len(video_files)} ({processed} ok, {skipped} skipped)")
            
            logger.info(f"   ✅ Processed {processed} from {speaker}")
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        logger.info(f"\n✅ Total extracted: {len(all_features)} features")
        
        # NOW DO 75/25 SPLIT
        logger.info(f"\n" + "="*70)
        logger.info(f"SPLITTING INTO TRAIN/TEST (75/25 per speaker)")
        logger.info(f"="*70)
        
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        # Split each speaker separately to ensure both sets have all speakers
        for speaker in sorted(set(all_labels)):
            speaker_mask = all_labels == speaker
            speaker_features = all_features[speaker_mask]
            speaker_labels = all_labels[speaker_mask]
            
            logger.info(f"\nSplitting {speaker}: {len(speaker_features)} videos")
            
            # 75% train, 25% test
            X_tr, X_te, y_tr, y_te = train_test_split(
                speaker_features, speaker_labels,
                train_size=0.75, test_size=0.25,
                random_state=42
            )
            
            logger.info(f"  → Train: {len(X_tr)} samples")
            logger.info(f"  → Test: {len(X_te)} samples")
            
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            X_test_list.append(X_te)
            y_test_list.append(y_te)
        
        # Combine all speakers
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        logger.info(f"\n✅ FINAL SPLIT:")
        logger.info(f"   Train: {X_train.shape} with classes {np.unique(y_train)}")
        logger.info(f"   Test: {X_test.shape} with classes {np.unique(y_test)}")
        
        return X_train, y_train, X_test, y_test


class MLClassifierTrainer:
    """Train ML classifiers on CNN features."""
    
    def __init__(self, weights_path=None):
        """Initialize trainer."""
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.svm_model = None
        self.rf_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        if weights_path:
            logger.info("Initializing feature extractor...")
            self.extractor = LipNetFeatureExtractor(weights_path)
            logger.info("✅ Feature extractor ready!")
    
    def extract_all_features(self, base_path, train_speakers, test_speakers):
        """Extract CNN features with proper train/test split."""
        logger.info("\n" + "="*70)
        logger.info("EXTRACTING AND SPLITTING FEATURES")
        logger.info("="*70)
        
        # Combine train and test speakers for extraction
        all_speakers = list(set(train_speakers + test_speakers))
        
        dataset = FeatureDatasetFixed(self.extractor)
        X_train, y_train, X_test, y_test = dataset.extract_from_directory_with_split(
            base_path, speakers=all_speakers, split_ratio=0.75
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("❌ No features extracted!")
            return None, None, None, None
        
        # Ensure 2D
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
        
        # Encode and scale
        all_labels = np.concatenate([y_train, y_test])
        self.label_encoder.fit(all_labels)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train_encoded, y_test_encoded
        
        logger.info("\n✅ Feature extraction complete!")
        logger.info(f"   Training: {self.X_train.shape}")
        logger.info(f"   Test: {self.X_test.shape}")
        logger.info(f"   Classes: {len(self.label_encoder.classes_)}")
        logger.info(f"   Class names: {self.label_encoder.classes_}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def train_svm(self, kernel='rbf', C=1.0, gamma='scale'):
        """Train SVM."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING SVM CLASSIFIER")
        logger.info("="*70)
        logger.info(f"Kernel: {kernel}, C: {C}")
        
        self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma, verbose=0)
        self.svm_model.fit(self.X_train, self.y_train)
        
        y_pred = self.svm_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logger.info(f"\n✅ SVM Complete! Accuracy: {accuracy*100:.2f}%")
        logger.info("\nClassification Report:")
        logger.info(classification_report(self.y_test, y_pred, zero_division=0))
        
        return self.svm_model, accuracy
    
    def train_random_forest(self, n_estimators=100, max_depth=20):
        """Train Random Forest."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING RANDOM FOREST CLASSIFIER")
        logger.info("="*70)
        logger.info(f"Trees: {n_estimators}, Max depth: {max_depth}")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        self.rf_model.fit(self.X_train, self.y_train)
        
        y_pred = self.rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        logger.info(f"\n✅ Random Forest Complete! Accuracy: {accuracy*100:.2f}%")
        logger.info("\nClassification Report:")
        logger.info(classification_report(self.y_test, y_pred, zero_division=0))
        
        return self.rf_model, accuracy
    
    def compare_models(self):
        """Compare models."""
        if self.svm_model is None or self.rf_model is None:
            logger.error("Both models must be trained!")
            return
        
        svm_pred = self.svm_model.predict(self.X_test)
        rf_pred = self.rf_model.predict(self.X_test)
        
        svm_acc = accuracy_score(self.y_test, svm_pred)
        rf_acc = accuracy_score(self.y_test, rf_pred)
        
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON")
        logger.info("="*70)
        logger.info(f"SVM Accuracy:           {svm_acc*100:.2f}%")
        logger.info(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
        logger.info(f"Best:                   {'SVM' if svm_acc > rf_acc else 'Random Forest'}")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            cm_svm = confusion_matrix(self.y_test, svm_pred)
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
            axes[0].set_title('SVM Confusion Matrix')
            axes[0].set_ylabel('True')
            axes[0].set_xlabel('Predicted')
            
            cm_rf = confusion_matrix(self.y_test, rf_pred)
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
            axes[1].set_title('Random Forest Confusion Matrix')
            axes[1].set_ylabel('True')
            axes[1].set_xlabel('Predicted')
            
            plt.tight_layout()
            plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
            logger.info("\n✅ Saved: confusion_matrices.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")
    
    def save_models(self, svm_path='models/svm_lip_reader.pkl',
                   rf_path='models/rf_lip_reader.pkl'):
        """Save models."""
        os.makedirs(os.path.dirname(svm_path) or '.', exist_ok=True)
        
        if self.svm_model:
            with open(svm_path, 'wb') as f:
                pickle.dump(self.svm_model, f)
            logger.info(f"✅ SVM saved: {svm_path}")
        
        if self.rf_model:
            with open(rf_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            logger.info(f"✅ RF saved: {rf_path}")
        
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("✅ Encoder and scaler saved")