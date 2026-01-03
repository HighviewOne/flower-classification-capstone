#!/usr/bin/env python3
"""
Test script for the flower classification prediction service.

Usage:
    python test_service.py [--url URL]

Examples:
    python test_service.py                           # Test local service
    python test_service.py --url http://server:9696  # Test remote service
"""

import argparse
import requests
import base64
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "flower_photos"


def get_test_images():
    """Get one test image from each class in the dataset."""
    test_images = {}
    
    if not DATA_DIR.exists():
        print(f"Warning: Dataset not found at {DATA_DIR}")
        print("Using URL-based tests instead (may be less reliable)")
        return None
    
    # Get first image from each class
    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            if images:
                test_images[class_dir.name] = images[0]
    
    return test_images


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    url = f"{base_url}/health"
    print(f"\n{'='*50}")
    print(f"Testing: GET {url}")
    print('='*50)
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_prediction_with_file(base_url: str, flower_type: str, image_path: Path) -> bool:
    """Test prediction with a local image file."""
    url = f"{base_url}/predict"
    print(f"\n{'='*50}")
    print(f"Testing: POST {url}")
    print(f"Image: {flower_type} ({image_path.name})")
    print('='*50)
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/jpeg")}
            response = requests.post(url, files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predicted = result.get("prediction", "unknown")
            confidence = result.get("confidence", 0)
            
            print(f"Predicted: {predicted} ({confidence:.1%})")
            
            # Check if prediction matches expected
            is_correct = predicted.lower() == flower_type.lower()
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"Expected: {flower_type} → {status}")
            
            return is_correct
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_prediction_with_base64(base_url: str, flower_type: str, image_path: Path) -> bool:
    """Test prediction with base64-encoded image."""
    url = f"{base_url}/predict"
    print(f"\n{'='*50}")
    print(f"Testing: POST {url} (base64)")
    print(f"Image: {flower_type}")
    print('='*50)
    
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = requests.post(
            url,
            json={"image_base64": image_base64},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predicted = result.get("prediction", "unknown")
            confidence = result.get("confidence", 0)
            
            print(f"Predicted: {predicted} ({confidence:.1%})")
            return True
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test flower classification service")
    parser.add_argument(
        "--url",
        default="http://localhost:9696",
        help="Base URL of the prediction service"
    )
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    
    print("\n" + "=" * 50)
    print("FLOWER CLASSIFICATION - SERVICE TEST")
    print("=" * 50)
    print(f"Testing service at: {base_url}")
    
    # Test health endpoint
    health_ok = test_health(base_url)
    
    if not health_ok:
        print("\n❌ Health check failed! Is the service running?")
        print("\nStart the service with:")
        print("  python src/predict.py")
        sys.exit(1)
    
    # Get test images from dataset
    test_images = get_test_images()
    
    if not test_images:
        print("\n❌ No test images found. Please download the dataset first:")
        print("  python src/download_data.py")
        sys.exit(1)
    
    # Test predictions with file uploads
    results = []
    for flower_type, image_path in sorted(test_images.items()):
        is_correct = test_prediction_with_file(base_url, flower_type, image_path)
        results.append((flower_type, is_correct))
    
    # Test base64 encoding with one image
    print("\n" + "-" * 50)
    print("Testing base64 encoding...")
    first_flower, first_path = list(test_images.items())[0]
    test_prediction_with_base64(base_url, first_flower, first_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for flower_type, is_correct in results:
        status = "✓" if is_correct else "✗"
        print(f"  {status} {flower_type}")
    
    print("-" * 50)
    print(f"Passed: {passed}/{total} ({passed/total:.0%})")
    
    if passed == total:
        print("\n✅ All tests passed!")
        sys.exit(0)
    elif passed >= total * 0.8:
        print("\n⚠️ Most tests passed (some misclassifications expected)")
        sys.exit(0)
    else:
        print("\n❌ Too many tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
