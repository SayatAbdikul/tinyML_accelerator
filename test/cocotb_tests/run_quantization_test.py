#!/usr/bin/env python3
"""
Runner script for quantization module tests
"""

import os
import subprocess
import sys

def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "=" * 70)
    print("  Quantization Module Unit Test")
    print("=" * 70)
    print(f"\nTest directory: {test_dir}")
    
    # Run the test using the Makefile
    cmd = f"cd {test_dir} && make -f Makefile.quantization"
    
    print(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("  ✅ Quantization tests completed successfully")
        print("=" * 70)
        print("\nGenerated files:")
        print(f"  - Waveform: {test_dir}/dump.vcd")
        print(f"  - Results: {test_dir}/results.xml")
    else:
        print("\n" + "=" * 70)
        print("  ❌ Quantization tests failed")
        print("=" * 70)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
