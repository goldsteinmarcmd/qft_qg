#!/usr/bin/env python3
"""
Quantum Gravity Framework Test Suite

This script runs all integration tests and numerical validations for the
quantum gravity framework, generating a comprehensive test report.
"""

import os
import sys
import unittest
import time
import datetime
import importlib
import json

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import test modules
import tests.test_qft_qg_integration
import tests.test_numerical_validations

def run_all_tests():
    """Run all tests and generate a report."""
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add integration tests
    test_suite.addTest(unittest.makeSuite(tests.test_qft_qg_integration.TestQFTQGIntegration))
    
    # Add numerical validations
    test_suite.addTest(unittest.makeSuite(tests.test_numerical_validations.TestNumericalValidations))
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print("=" * 80)
    print("Quantum Gravity Framework Test Suite")
    print("=" * 80)
    print(f"Running tests at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Calculate run time
    run_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Time taken: {run_time:.2f} seconds")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(parent_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to file
    results_file = os.path.join(results_dir, f"test_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("Quantum Gravity Framework Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests: {result.testsRun}\n")
        f.write(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Failed: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Time taken: {run_time:.2f} seconds\n\n")
        
        if result.failures:
            f.write("Failures:\n")
            for test, traceback in result.failures:
                f.write(f"\n{test}\n")
                f.write("-" * 40 + "\n")
                f.write(f"{traceback}\n")
        
        if result.errors:
            f.write("Errors:\n")
            for test, traceback in result.errors:
                f.write(f"\n{test}\n")
                f.write("-" * 40 + "\n")
                f.write(f"{traceback}\n")
    
    print(f"\nTest results saved to: {results_file}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 