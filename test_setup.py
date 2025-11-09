"""
test_setup.py
Verify your setup before running full training.
Run this first to debug any issues.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages are installed."""
    logger.info("="*60)
    logger.info("TEST 1: Checking Package Imports")
    logger.info("="*60)
    
    packages = {
        'tensorflow': 'Deep learning framework',
        'keras': 'Keras (part of TensorFlow)',
        'numpy': 'Numerical computing',
        'sklearn': 'scikit-learn (ML algorithms)',
        'cv2': 'OpenCV (video processing)',
        'matplotlib': 'Plotting',
        'pickle': 'Serialization',
    }
    
    missing = []
    for package_name, description in packages.items():
        try:
            __import__(package_name)
            logger.info(f"  ✅ {package_name:20} - {description}")
        except ImportError:
            logger.error(f"  ❌ {package_name:20} - {description}")
            missing.append(package_name)
    
    if missing:
        logger.error(f"\n❌ Missing packages: {', '.join(missing)}")
        logger.info("\nInstall with: pip install " + " ".join(missing))
        return False
    
    logger.info("\n✅ All packages installed!")
    return True


def test_file_structure():
    """Test if required files/directories exist."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Checking File Structure")
    logger.info("="*60)
    
    required_paths = {
        'models/': 'Models directory (for weights)',
        'data/': 'Data directory (for GRID dataset)',
    }
    
    missing_paths = []
    for path, description in required_paths.items():
        if os.path.exists(path):
            logger.info(f"  ✅ {path:20} - {description}")
        else:
            logger.warning(f"  ⚠️  {path:20} - {description} (creating...)")
            os.makedirs(path, exist_ok=True)
    
    return True


def test_weights():
    """Check if LipNet weights are downloaded."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Checking LipNet Weights")
    logger.info("="*60)
    
    weights_path = 'models/overlapped-weights368.h5'
    
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        logger.info(f"  ✅ Weights found: {weights_path}")
        logger.info(f"     File size: {size_mb:.1f} MB")
        return True
    else:
        logger.error(f"  ❌ Weights not found: {weights_path}")
        logger.info("\n   Download from:")
        logger.info("   https://github.com/rizkiarm/LipNet/tree/master/evaluation/models")
        logger.info("\n   Download 'overlapped-weights368.h5' and save to 'models/' folder")
        return False


def test_data_structure():
    """Check if GRID data is properly structured."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Checking GRID Data Structure")
    logger.info("="*60)
    
    base_path = 'data'
    speakers = ['s1_processed', 's2_processed', 's3_processed', 's4_processed']
    
    if not os.path.exists(base_path):
        logger.warning(f"  ⚠️  Data directory '{base_path}' not found")
        logger.info("   Create your GRID dataset structure:")
        logger.info("   data/")
        logger.info("   ├── s1_processed/")
        logger.info("   │   ├── video1.mpg")
        logger.info("   │   ├── align/")
        logger.info("   │   │   └── video1.align")
        logger.info("   ├── s2_processed/")
        logger.info("   └── s4_processed/")
        return False
    
    found_speakers = []
    for speaker in speakers:
        speaker_path = os.path.join(base_path, speaker)
        if os.path.exists(speaker_path):
            # Count videos
            videos = [f for f in os.listdir(speaker_path) 
                     if f.endswith(('.mpg', '.mp4', '.avi'))]
            logger.info(f"  ✅ {speaker:20} - {len(videos)} videos")
            found_speakers.append(speaker)
        else:
            logger.warning(f"  ⚠️  {speaker:20} - not found")
    
    if len(found_speakers) >= 2:
        logger.info(f"\n✅ Found {len(found_speakers)}/{len(speakers)} speakers (minimum 2 required)")
        return True
    else:
        logger.error(f"\n❌ Not enough speakers found (need at least 2)")
        return False


def test_tensorflow():
    """Test TensorFlow GPU support."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: TensorFlow Configuration")
    logger.info("="*60)
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        tf_version = tf.__version__
        logger.info(f"  ✅ TensorFlow version: {tf_version}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"  ✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"     - {gpu}")
        else:
            logger.info(f"  ⚠️  No GPU detected (CPU mode)")
            logger.info("     Training will be slower. CPU is OK for this project.")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ TensorFlow error: {e}")
        return False


def test_sklearn():
    """Test scikit-learn classifier imports."""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: scikit-learn ML Classifiers")
    logger.info("="*60)
    
    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        logger.info("  ✅ SVC (Support Vector Classifier)")
        logger.info("  ✅ RandomForestClassifier")
        logger.info("  ✅ StandardScaler")
        logger.info("  ✅ LabelEncoder")
        
        return True
        
    except ImportError as e:
        logger.error(f"  ❌ scikit-learn import error: {e}")
        return False


def print_summary(results):
    """Print test summary."""
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    test_names = [
        "Imports",
        "File Structure",
        "Weights",
        "Data Structure",
        "TensorFlow",
        "scikit-learn"
    ]
    
    all_passed = all(results)
    
    for name, passed in zip(test_names, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {name:20} {status}")
    
    logger.info("\n" + "="*60)
    
    if all_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("\nYou're ready to run: python quick_start.py")
        logger.info("\nMake sure to edit quick_start.py with your data paths first!")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.info("\nFix the issues above before running training.")
    
    logger.info("="*60)
    
    return all_passed


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " HYBRID LIP READING SETUP TEST ".center(58) + "║")
    logger.info("╚" + "="*58 + "╝")
    logger.info("\nThis script verifies your environment is ready for training.")
    logger.info("It checks packages, files, data, and TensorFlow configuration.\n")
    
    results = [
        test_imports(),
        test_file_structure(),
        test_weights(),
        test_data_structure(),
        test_tensorflow(),
        test_sklearn(),
    ]
    
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())