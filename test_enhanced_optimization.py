#!/usr/bin/env python
"""
Test Enhanced Optimization Implementation

This script tests the enhanced optimization implementation for the QFT-QG framework.
"""

import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_optimization_implementation import EnhancedOptimizer
    print("✅ Enhanced optimization implementation imported successfully")
except ImportError as e:
    print(f"❌ Failed to import enhanced optimization: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic optimization functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        # Create optimizer
        optimizer = EnhancedOptimizer(
            use_gpu=True,
            use_parallel=True,
            cache_size=1000,
            memory_limit_gb=4.0
        )
        print("✅ Optimizer created successfully")
        
        # Test system info
        print(f"System info: {optimizer.system_info}")
        
        # Test GPU acceleration
        if optimizer.use_gpu:
            print("Testing GPU acceleration...")
            result = optimizer.gpu_accelerated_spectral_dimension(1.0)
            print(f"GPU spectral dimension result: {result}")
        
        # Test caching
        print("Testing caching...")
        def test_calc(x):
            return x * x + 1
        
        result1 = optimizer.smart_cache('test_calc', (5.0,), test_calc)
        result2 = optimizer.smart_cache('test_calc', (5.0,), test_calc)  # Should be cached
        print(f"Caching test results: {result1}, {result2}")
        print(f"Cache stats: {optimizer.cache_stats}")
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("\n" + "="*60)
    print("TESTING PARALLEL PROCESSING")
    print("="*60)
    
    try:
        optimizer = EnhancedOptimizer(use_parallel=True)
        
        # Test parameter scan
        param_ranges = {
            'diffusion_time': (0.1, 10.0),
            'energy_scale': (1e6, 1e9)
        }
        
        print("Running parallel parameter scan...")
        start_time = time.time()
        results = optimizer.parallel_parameter_scan(param_ranges, num_points=20)
        end_time = time.time()
        
        print(f"Parameter scan completed in {end_time - start_time:.3f} seconds")
        print(f"Total combinations: {results['total_combinations']}")
        print(f"Workers used: {results['workers_used']}")
        
        print("✅ Parallel processing tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization capabilities."""
    print("\n" + "="*60)
    print("TESTING MEMORY OPTIMIZATION")
    print("="*60)
    
    try:
        optimizer = EnhancedOptimizer(memory_limit_gb=2.0)
        
        # Test memory-efficient calculations
        print("Running memory optimization test...")
        memory_results = optimizer._optimize_memory_usage()
        
        print(f"Memory optimization results:")
        print(f"  Initial memory: {memory_results['initial_memory_gb']:.2f} GB")
        print(f"  Final memory: {memory_results['final_memory_gb']:.2f} GB")
        print(f"  Memory savings: {memory_results['memory_savings_gb']:.2f} GB")
        print(f"  Cached calculations: {memory_results['cached_calculations']}")
        
        print("✅ Memory optimization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    print("\n" + "="*60)
    print("TESTING GPU ACCELERATION")
    print("="*60)
    
    try:
        optimizer = EnhancedOptimizer(use_gpu=True)
        
        if not optimizer.use_gpu:
            print("⚠️  GPU not available, skipping GPU tests")
            return True
        
        print("Running GPU acceleration benchmark...")
        gpu_results = optimizer._benchmark_gpu_acceleration()
        
        print(f"GPU acceleration results:")
        print(f"  GPU available: {gpu_results['gpu_available']}")
        if gpu_results['gpu_available']:
            print(f"  Best speedup: {gpu_results['best_speedup']:.2f}x")
            print(f"  Available backends: {list(gpu_results['backends'].keys())}")
        
        print("✅ GPU acceleration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU acceleration test failed: {e}")
        return False

def test_caching():
    """Test caching capabilities."""
    print("\n" + "="*60)
    print("TESTING CACHING")
    print("="*60)
    
    try:
        optimizer = EnhancedOptimizer(cache_size=1000)
        
        print("Running caching benchmark...")
        caching_results = optimizer._benchmark_caching()
        
        print(f"Caching results:")
        print(f"  Cache efficiency: {caching_results['cache_efficiency']:.1%}")
        print(f"  Cache hit rate: {caching_results['cache_hit_rate']:.1%}")
        print(f"  Cache hits: {caching_results['cache_hits']}")
        print(f"  Cache misses: {caching_results['cache_misses']}")
        
        print("✅ Caching tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive optimization test."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE OPTIMIZATION TEST")
    print("="*60)
    
    try:
        optimizer = EnhancedOptimizer(
            use_gpu=True,
            use_parallel=True,
            cache_size=5000,
            memory_limit_gb=4.0
        )
        
        print("Running comprehensive optimization...")
        results = optimizer.run_comprehensive_optimization()
        
        print("\nComprehensive optimization completed!")
        print(f"Results keys: {list(results.keys())}")
        
        # Print summary
        optimizer.print_optimization_summary()
        
        print("✅ Comprehensive optimization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all optimization tests."""
    print("Enhanced Optimization Implementation Test")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Parallel Processing", test_parallel_processing),
        ("Memory Optimization", test_memory_optimization),
        ("GPU Acceleration", test_gpu_acceleration),
        ("Caching", test_caching),
        ("Comprehensive Test", run_comprehensive_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimization tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 