#!/usr/bin/env python3
"""
Quick validation script to run a subset of heavy tests
Usage: python quick_validate.py [num_images]
"""

import subprocess
import sys

def main():
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"Running quick validation with {num_images} images...")
    print("=" * 80)
    
    cmd = f"make run_test NUM_IMAGES={num_images}"
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
