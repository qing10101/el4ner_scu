"""
Installation Test Script for Hierarchical NER System
Run this to verify everything is set up correctly.
"""

import sys
import importlib
from typing import List, Tuple


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed.
    
    Args:
        package_name: Display name of the package
        import_name: Import name (if different from package name)
    
    Returns:
        Tuple of (success, message)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✓ {package_name} (version {version})"
    except ImportError as e:
        return False, f"✗ {package_name} - NOT INSTALLED"
    except Exception as e:
        return False, f"✗ {package_name} - ERROR: {str(e)}"


def check_pytorch():
    """Check PyTorch installation and GPU availability."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"✓ PyTorch {version} with GPU ({gpu_name})"
        else:
            return True, f"✓ PyTorch {version} (CPU only - GPU not available)"
    except ImportError:
        return False, "✗ PyTorch - NOT INSTALLED"
    except Exception as e:
        return False, f"✗ PyTorch - ERROR: {str(e)}"


def check_transformers_models():
    """Check if transformers can download models."""
    try:
        from transformers import AutoTokenizer
        # Try to load a small tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return True, "✓ Transformers model loading works"
    except Exception as e:
        return False, f"⚠ Transformers model loading - WARNING: {str(e)[:100]}"


def check_hierarchical_ner_files():
    """Check if hierarchical NER files exist."""
    import os
    
    required_files = [
        'hierarchical_ner.py',
        'hierarchical_ner_config.py',
        'hierarchical_ner_api.py'
    ]
    
    optional_files = [
        'compare_hierarchical_ner.py',
        'integration_example.py',
        'HIERARCHICAL_NER_README.md',
        'QUICKSTART.md'
    ]
    
    results = []
    all_present = True
    
    for file in required_files:
        if os.path.exists(file):
            results.append(f"  ✓ {file}")
        else:
            results.append(f"  ✗ {file} - MISSING (REQUIRED)")
            all_present = False
    
    for file in optional_files:
        if os.path.exists(file):
            results.append(f"  ✓ {file}")
        else:
            results.append(f"  ⚠ {file} - missing (optional)")
    
    return all_present, "\n".join(results)


def test_basic_functionality():
    """Test basic functionality of the hierarchical NER system."""
    try:
        from hierarchical_ner_api import extract_entities
        
        test_text = "Apple Inc. is in California."
        
        print("\n  Testing basic entity extraction...")
        print(f"  Input: '{test_text}'")
        
        # This will trigger model downloads on first run
        entities = extract_entities(test_text, preset='balanced')
        
        if entities:
            print(f"  ✓ Found {len(entities)} entities:")
            for e in entities:
                print(f"    - {e['entity']} ({e['type']})")
            return True, "✓ Basic functionality test PASSED"
        else:
            return False, "✗ Basic functionality test FAILED (no entities found)"
            
    except Exception as e:
        return False, f"✗ Basic functionality test FAILED: {str(e)}"


def main():
    """Run all installation checks."""
    
    print("="*70)
    print("HIERARCHICAL NER - INSTALLATION TEST")
    print("="*70)
    
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Required packages
    print("\n" + "-"*70)
    print("CHECKING REQUIRED PACKAGES")
    print("-"*70)
    
    required_packages = [
        ('NumPy', 'numpy'),
        ('SciPy', 'scipy'),
        ('Tabulate', 'tabulate'),
        ('tqdm', 'tqdm'),
    ]
    
    all_required_ok = True
    
    for package_name, import_name in required_packages:
        success, message = check_package(package_name, import_name)
        print(message)
        if not success:
            all_required_ok = False
    
    # PyTorch special check
    success, message = check_pytorch()
    print(message)
    if not success:
        all_required_ok = False
    
    # Transformers
    success, message = check_package('Transformers', 'transformers')
    print(message)
    if not success:
        all_required_ok = False
    
    # Check if transformers can load models
    success, message = check_transformers_models()
    print(message)
    
    # Check hierarchical NER files
    print("\n" + "-"*70)
    print("CHECKING HIERARCHICAL NER FILES")
    print("-"*70)
    
    success, message = check_hierarchical_ner_files()
    print(message)
    
    if not success:
        all_required_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_required_ok:
        print("✓ ALL REQUIRED COMPONENTS INSTALLED")
        print("="*70)
        
        # Optional: Test basic functionality
        print("\n" + "-"*70)
        print("TESTING BASIC FUNCTIONALITY (This may take a minute...)")
        print("-"*70)
        
        try:
            success, message = test_basic_functionality()
            print(f"\n{message}")
            
            if success:
                print("\n" + "="*70)
                print("🎉 INSTALLATION SUCCESSFUL!")
                print("="*70)
                print("\nYou're ready to use the Hierarchical NER system!")
                print("\nNext steps:")
                print("  1. Read QUICKSTART.md for basic usage")
                print("  2. Run: python hierarchical_ner_api.py")
                print("  3. Run: python compare_hierarchical_ner.py")
                print("="*70)
                return 0
            else:
                print("\n⚠ WARNING: Basic functionality test failed.")
                print("Installation may be incomplete.")
                return 1
                
        except Exception as e:
            print(f"\n⚠ Could not test functionality: {e}")
            print("Please try running the examples manually.")
            return 1
    else:
        print("✗ INSTALLATION INCOMPLETE")
        print("="*70)
        print("\nSome required components are missing.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements_hierarchical.txt")
        print("\nOr install individually:")
        print("  pip install torch transformers scipy numpy tabulate tqdm")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

