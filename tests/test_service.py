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
import sys

# Sample flower image URLs for testing
TEST_IMAGES = {
    "sunflower": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/800px-Sunflower_sky_backdrop.jpg",
    "rose": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Rosa_rubiginosa_1.jpg/800px-Rosa_rubiginosa_1.jpg",
    "daisy": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Leucanthemum_vulgare_%27Filigran%27_Flower_2200px.jpg/800px-Leucanthemum_vulgare_%27Filigran%27_Flower_2200px.jpg",
    "tulip": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Tulip_-_florescence.jpg/800px-Tulip_-_florescence.jpg",
    "dandelion": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Taraxacum_officinale_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-135.jpg/800px-Taraxacum_officinale_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-135.jpg"
}


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


def test_prediction(base_url: str, flower_type: str, image_url: str) -> bool:
    """Test prediction with a sample image URL."""
    url = f"{base_url}/predict"
    print(f"\n{'='*50}")
    print(f"Testing: POST {url}")
    print(f"Image: {flower_type}")
    print('='*50)
    
    try:
        response = requests.post(
            url,
            json={"image_url": image_url},
            timeout=30
        )
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
        sys.exit(1)
    
    # Test predictions
    results = []
    for flower_type, image_url in TEST_IMAGES.items():
        is_correct = test_prediction(base_url, flower_type, image_url)
        results.append((flower_type, is_correct))
    
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
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
