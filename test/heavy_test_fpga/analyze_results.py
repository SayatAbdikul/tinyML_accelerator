#!/usr/bin/env python3
"""
Analyze test results from CSV file
===================================

This script provides analysis and visualization of test results
from the heavy test suite.

Usage:
    python analyze_results.py [test_results.csv]
"""

import csv
import sys
from collections import defaultdict

def analyze_results(csv_file='test_results.csv'):
    """Analyze test results from CSV file"""
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        print("Run tests first: make quick_test")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if not results:
        print("No results found in CSV file")
        return
    
    # Calculate statistics
    total = len(results)
    rtl_correct = sum(1 for r in results if r['rtl_correct'] == 'True')
    golden_correct = sum(1 for r in results if r['golden_correct'] == 'True')
    exact_matches = sum(1 for r in results if r['exact_match'] == 'True')
    
    # Per-class statistics
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0, 'mismatches': 0})
    
    for r in results:
        label = int(r['label'])
        per_class[label]['total'] += 1
        if r['rtl_correct'] == 'True':
            per_class[label]['correct'] += 1
        if r['exact_match'] == 'False':
            per_class[label]['mismatches'] += 1
    
    # Error statistics
    errors = []
    for r in results:
        if r['max_error'] != 'N/A':
            try:
                errors.append(int(r['max_error']))
            except ValueError:
                pass
    
    # Find failures
    failures = [r for r in results if r['rtl_correct'] == 'False' or r['exact_match'] == 'False']
    
    # Print report
    print("=" * 80)
    print("TEST RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {csv_file}")
    print(f"Total tests: {total}")
    
    print(f"\n{'─' * 80}")
    print("OVERALL STATISTICS")
    print(f"{'─' * 80}")
    print(f"RTL Correct:     {rtl_correct:5d} / {total:5d} ({rtl_correct/total*100:.2f}%)")
    print(f"Golden Correct:  {golden_correct:5d} / {total:5d} ({golden_correct/total*100:.2f}%)")
    print(f"Exact Matches:   {exact_matches:5d} / {total:5d} ({exact_matches/total*100:.2f}%)")
    print(f"Failures:        {len(failures):5d} / {total:5d} ({len(failures)/total*100:.2f}%)")
    
    if errors:
        print(f"\nError Statistics:")
        print(f"  Average error: {sum(errors)/len(errors):.4f}")
        print(f"  Maximum error: {max(errors)}")
        print(f"  Minimum error: {min(errors)}")
    
    print(f"\n{'─' * 80}")
    print("PER-CLASS ACCURACY")
    print(f"{'─' * 80}")
    print(f"{'Digit':<8} {'Total':<8} {'Correct':<8} {'Accuracy':<12} {'Mismatches':<12}")
    print(f"{'─' * 80}")
    
    for digit in sorted(per_class.keys()):
        stats = per_class[digit]
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{digit:<8} {stats['total']:<8} {stats['correct']:<8} {accuracy:<11.2f}% {stats['mismatches']:<12}")
    
    if failures:
        print(f"\n{'─' * 80}")
        print(f"FAILURES (showing first 20)")
        print(f"{'─' * 80}")
        print(f"{'Test':<6} {'Label':<7} {'RTL':<7} {'Golden':<8} {'Error':<8} {'Match':<8}")
        print(f"{'─' * 80}")
        
        for i, f in enumerate(failures[:20]):
            test_idx = f['test_idx']
            label = f['label']
            rtl_pred = f['rtl_prediction']
            golden_pred = f['golden_prediction']
            max_error = f['max_error']
            exact_match = 'Yes' if f['exact_match'] == 'True' else 'No'
            
            print(f"{test_idx:<6} {label:<7} {rtl_pred:<7} {golden_pred:<8} {max_error:<8} {exact_match:<8}")
        
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more failures")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Pass criteria
    pass_accuracy = (rtl_correct / total * 100) >= 85.0
    pass_exact_match = (exact_matches / total * 100) >= 95.0
    pass_max_error = (max(errors) if errors else 0) <= 3
    
    print(f"RTL Accuracy ≥ 85%:      {'✓ PASS' if pass_accuracy else '✗ FAIL'}")
    print(f"Exact match rate ≥ 95%:  {'✓ PASS' if pass_exact_match else '✗ FAIL'}")
    print(f"Max error ≤ 3:           {'✓ PASS' if pass_max_error else '✗ FAIL'}")
    
    overall = pass_accuracy and pass_exact_match and pass_max_error
    print(f"\nOVERALL: {'✅ PASS' if overall else '❌ FAIL'}")
    print("=" * 80)


if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_results.csv'
    analyze_results(csv_file)
