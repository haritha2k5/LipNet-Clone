"""
quick_start.py - FIXED
Proper train/test split: Mix videos from each speaker
"""

import logging
import os
from train_ml_classifiers import MLClassifierTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_start_hybrid_system():
    """Hybrid Lip Reading with proper train/test split"""
    
    # ==================== CONFIGURATION ====================
    WEIGHTS_PATH = "models/overlapped-weights368.h5"
    BASE_DATA_PATH = "data"
    
    # ✅ FIX: Use ALL 4 speakers for training, proper split per speaker
    # This will extract ~750 videos per speaker for training (75%)
    # And ~250 videos per speaker for testing (25%)
    # So test set will have all 4 speakers!
    TRAIN_SPEAKERS = ["s1_processed", "s2_processed", "s3_processed", "s4_processed"]
    TEST_SPEAKERS = ["s1_processed", "s2_processed", "s3_processed", "s4_processed"]
    
    SVM_KERNEL = "rbf"
    SVM_C = 1.0
    
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    # =======================================================
    
    logger.info("="*70)
    logger.info("HYBRID LIP READING SYSTEM - SETUP VERIFICATION")
    logger.info("="*70)
    
    # Verify paths
    if not os.path.exists(WEIGHTS_PATH):
        logger.error(f"❌ Weights not found: {WEIGHTS_PATH}")
        logger.info("Download: https://github.com/rizkiarm/LipNet/tree/master/evaluation/models")
        return
    logger.info(f"✅ Weights: {WEIGHTS_PATH}")
    
    if not os.path.exists(BASE_DATA_PATH):
        logger.error(f"❌ Data directory not found: {BASE_DATA_PATH}")
        return
    logger.info(f"✅ Data: {BASE_DATA_PATH}")
    
    # Check speakers
    all_speakers = TRAIN_SPEAKERS + TEST_SPEAKERS
    for speaker in all_speakers:
        path = os.path.join(BASE_DATA_PATH, speaker)
        if os.path.exists(path):
            logger.info(f"✅ {speaker}")
        else:
            logger.error(f"❌ {speaker} NOT FOUND")
            return
    
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    # Initialize
    try:
        logger.info("\n🔧 Initializing trainer...")
        trainer = MLClassifierTrainer(weights_path=WEIGHTS_PATH)
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return
    
    # Extract features
    try:
        logger.info("\n📊 STEP 1: FEATURE EXTRACTION")
        X_train, y_train, X_test, y_test = trainer.extract_all_features(
            BASE_DATA_PATH, TRAIN_SPEAKERS, TEST_SPEAKERS
        )
        if X_train is None:
            return
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        return
    
    # Train SVM
    try:
        logger.info("\n🤖 STEP 2: SVM TRAINING")
        svm_model, svm_acc = trainer.train_svm(kernel=SVM_KERNEL, C=SVM_C)
    except Exception as e:
        logger.error(f"❌ SVM training failed: {e}")
        return
    
    # Train Random Forest
    try:
        logger.info("\n🌲 STEP 3: RANDOM FOREST TRAINING")
        rf_model, rf_acc = trainer.train_random_forest(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH
        )
    except Exception as e:
        logger.error(f"❌ Random Forest training failed: {e}")
        return
    
    # Compare
    try:
        logger.info("\n📈 STEP 4: MODEL COMPARISON")
        trainer.compare_models()
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
    
    # Save
    try:
        logger.info("\n💾 STEP 5: SAVING MODELS")
        trainer.save_models()
    except Exception as e:
        logger.error(f"❌ Saving failed: {e}")
        return
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   SVM Accuracy:           {svm_acc*100:.2f}%")
    logger.info(f"   Random Forest Accuracy: {rf_acc*100:.2f}%")
    logger.info(f"   Best Model:             {'SVM' if svm_acc > rf_acc else 'Random Forest'}")
    logger.info(f"\n✅ Models saved to models/")
    logger.info(f"✅ Confusion matrices saved to confusion_matrices.png")
    logger.info("\n🎉 Ready for course submission!")


if __name__ == "__main__":
    quick_start_hybrid_system()