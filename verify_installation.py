#!/usr/bin/env python3
"""
Verify that emodb-utils is correctly installed and all dependencies are working.
Run this after installing with `pip install -e .`
"""

import sys
import importlib

# Apply torchaudio compatibility patch before importing speechbrain
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _list_audio_backends():
            """Dummy function for compatibility with older SpeechBrain versions."""
            return ['soundfile']
        torchaudio.list_audio_backends = _list_audio_backends
        if not hasattr(torchaudio, 'get_audio_backend'):
            torchaudio.get_audio_backend = lambda: 'soundfile'
except ImportError:
    pass


def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        return False


def main():
    """Run verification tests."""
    print("=" * 70)
    print("EmoDb Utils Installation Verification")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Test core dependencies
    print("Testing core dependencies:")
    all_passed &= test_import("numpy", "NumPy")
    all_passed &= test_import("pandas", "Pandas")
    all_passed &= test_import("torch", "PyTorch")
    all_passed &= test_import("torchaudio", "TorchAudio")
    all_passed &= test_import("librosa", "Librosa")
    all_passed &= test_import("sklearn", "Scikit-learn")
    all_passed &= test_import("soundfile", "SoundFile")
    all_passed &= test_import("speechbrain", "SpeechBrain")
    all_passed &= test_import("hyperpyyaml", "HyperPyYAML")
    print()
    
    # Test utils package
    print("Testing utils package:")
    all_passed &= test_import("utils", "utils package")
    all_passed &= test_import("utils.features_extraction", "utils.features_extraction")
    all_passed &= test_import("utils.training", "utils.training")
    all_passed &= test_import("utils.audio_processing", "utils.audio_processing")
    print()
    
    # Test specific imports
    print("Testing specific functionality:")
    try:
        from utils.features_extraction.extract_mfcc import extract_mfcc_features
        print("✓ MFCC extraction import")
    except ImportError as e:
        print(f"✗ MFCC extraction import: {e}")
        all_passed = False
    
    try:
        from utils.features_extraction.create_loso_splits import create_loso_splits
        print("✓ LOSO splits import")
    except ImportError as e:
        print(f"✗ LOSO splits import: {e}")
        all_passed = False
    
    try:
        from utils.training import train_speaker_model, train_all_speakers
        print("✓ Training utilities import")
    except ImportError as e:
        print(f"✗ Training utilities import: {e}")
        all_passed = False
    
    print()
    
    # Test compatibility patch
    print("Testing compatibility:")
    try:
        import torchaudio
        if hasattr(torchaudio, 'list_audio_backends'):
            print("✓ TorchAudio compatibility patch applied")
        else:
            print("⚠ TorchAudio compatibility patch not detected (may not be needed)")
    except Exception as e:
        print(f"⚠ TorchAudio compatibility check failed: {e}")
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✓ All verification tests passed!")
        print("✓ emodb-utils is correctly installed and ready to use.")
        print()
        print("Next steps:")
        print("  - Extract features: python -c 'from utils.features_extraction.extract_mfcc import extract_mfcc_from_dataset; extract_mfcc_from_dataset()'")
        print("  - Train models: python train_ecapa_models.py --speaker 03")
        print("  - See QUICKSTART.md for more examples")
        return 0
    else:
        print("✗ Some verification tests failed.")
        print("  Please check the error messages above and ensure all dependencies are installed.")
        print("  Try: pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
