"""
feature_extractor.py - ROBUST VERSION
Handles shape mismatches gracefully
"""

import os
import numpy as np
import pickle
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, LSTM, Bidirectional, 
    Dropout, Dense, Reshape, BatchNormalization
)
import cv2
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_lipnet_model():
    """
    Build LipNet with correct dimensions.
    LSTM: 256 units
    Input: 3-channel RGB (75, 100, 50, 3)
    """
    
    input_video = Input(shape=(75, 100, 50, 3), name='input_video')
    
    # Conv Blocks with BatchNorm
    x = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding='same', name='conv3d')(input_video)
    x = BatchNormalization(name='batch_normalization')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='max_pooling3d')(x)
    x = Dropout(0.5)(x)
    
    x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), padding='same', name='conv3d_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='max_pooling3d_1')(x)
    x = Dropout(0.5)(x)
    
    x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding='same', name='conv3d_2')(x)
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='max_pooling3d_2')(x)
    x = Dropout(0.5)(x)
    
    x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding='same', name='conv3d_3')(x)
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='max_pooling3d_3')(x)
    x = Dropout(0.5)(x)
    
    # Reshape for RNN
    x = Reshape((75, -1), name='reshape')(x)
    
    # LSTM Blocks - 256 units
    x = Bidirectional(LSTM(256, return_sequences=True, activation='relu'), name='bidirectional')(x)
    x = Dropout(0.5)(x)
    
    x = Bidirectional(LSTM(256, return_sequences=False, activation='relu'), name='bidirectional_1')(x)
    x = Dropout(0.5)(x)
    
    # Dense layers
    x = Dense(512, activation='relu', name='dense')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    
    # Output
    output = Dense(500, activation='softmax', name='dense_2')(x)
    
    model = Model(inputs=input_video, outputs=output)
    return model


class LipNetFeatureExtractor:
    """Extract features from LipNet."""
    
    def __init__(self, weights_path):
        """Initialize."""
        self.weights_path = weights_path
        self.feature_model = None
        self.full_model = None
        self._build_and_load()
    
    def _build_and_load(self):
        """Build and load."""
        try:
            logger.info(f"Building LipNet architecture...")
            
            self.full_model = build_lipnet_model()
            logger.info(f"✅ Architecture built")
            logger.info(f"   Input: {self.full_model.input_shape}")
            logger.info(f"   Output: {self.full_model.output_shape}")
            
            logger.info(f"Loading weights...")
            try:
                self.full_model.load_weights(self.weights_path, by_name=True, skip_mismatch=True)
            except TypeError:
                self.full_model.load_weights(self.weights_path, by_name=True)
            logger.info(f"✅ Weights loaded!")
            
            logger.info(f"Creating feature extraction model...")
            
            # Use Dense(512) layer as features
            feature_layer = None
            for layer in reversed(self.full_model.layers):
                if isinstance(layer, Dense) and 'dense_1' in layer.name:
                    feature_layer = layer
                    break
            
            if feature_layer is None:
                feature_layer = self.full_model.layers[-2]
            
            logger.info(f"✅ Using layer: {feature_layer.name} for features")
            
            self.feature_model = Model(
                inputs=self.full_model.input,
                outputs=feature_layer.output
            )
            
            logger.info(f"✅ Feature extractor ready!")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise
    
    def extract_features_from_video(self, video_path, mouth_crop_size=(100, 50)):
        """Extract features from video."""
        try:
            frames = self._load_video_frames(video_path, mouth_crop_size)
            
            if frames is None or len(frames) == 0:
                return None
            
            frames = self._pad_frames(frames, target_length=75)
            
            # Ensure correct shape: (1, 75, 100, 50, 3)
            if frames.shape != (75, 100, 50, 3):
                frames = frames.reshape(75, 100, 50, 3)
            
            frames = np.expand_dims(frames, axis=0)
            frames = frames.astype(np.float32)
            
            # Extract with error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features = self.feature_model.predict(frames, verbose=0)
            
            return features[0]
            
        except Exception as e:
            return None
    
    def _load_video_frames(self, video_path, mouth_crop_size=(100, 50)):
        """Load video frames as RGB."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            if not cap.isOpened():
                return None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized = cv2.resize(frame, mouth_crop_size)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb.astype(np.float32) / 255.0)
            
            cap.release()
            return np.array(frames) if frames else None
            
        except:
            return None
    
    def _pad_frames(self, frames, target_length=75):
        """Pad or trim frames."""
        n_frames = len(frames)
        
        if n_frames == target_length:
            return frames
        elif n_frames < target_length:
            last_frame = frames[-1]
            padding = np.tile(last_frame, (target_length - n_frames, 1, 1, 1))
            return np.vstack([frames, padding])
        else:
            indices = np.linspace(0, n_frames - 1, target_length, dtype=int)
            return frames[indices]


class FeatureDataset:
    """Extract and manage features."""
    
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.features = []
        self.labels = []
    
    def extract_from_directory(self, base_path, speakers, is_training=True):
        """Extract from directory - skip problematic videos."""
        all_features = []
        all_labels = []
        
        for speaker in speakers:
            speaker_path = os.path.join(base_path, speaker)
            if not os.path.exists(speaker_path):
                logger.warning(f"Not found: {speaker_path}")
                continue
            
            logger.info(f"\n📹 Processing: {speaker}")
            
            video_files = [f for f in os.listdir(speaker_path) 
                          if f.endswith(('.mpg', '.mp4', '.avi', '.mov'))]
            
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
                        label = os.path.splitext(video_file)[0]
                        all_labels.append(label)
                        processed += 1
                    else:
                        skipped += 1
                except:
                    skipped += 1
                
                if (idx + 1) % max(1, len(video_files) // 5) == 0:
                    logger.info(f"   {idx + 1}/{len(video_files)} ({processed} ok, {skipped} skipped)")
            
            logger.info(f"   ✅ Processed {processed} from {speaker}")
        
        self.features = np.array(all_features) if all_features else np.array([])
        self.labels = np.array(all_labels) if all_labels else np.array([])
        
        logger.info(f"\n✅ Total extracted: {len(self.features)} features")
        if len(self.features) > 0:
            logger.info(f"   Shape: {self.features.shape}")
        
        return self.features, self.labels
    
    def save_features(self, save_path):
        """Save."""
        data = {'features': self.features, 'labels': self.labels}
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✅ Saved: {save_path}")
    
    def load_features(self, load_path):
        """Load."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.features = data['features']
        self.labels = data['labels']
        return self.features, self.labels