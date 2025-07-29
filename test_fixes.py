#!/usr/bin/env python
"""
Test script to verify numerical fixes work.
"""

import numpy as np
import warnings

# Import the fixed modules
from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
from quantum_gravity_framework.category_theory import CategoryTheoryGeometry

def test_spectral_dimension_fix():
    """Test that spectral dimension calculation no longer produces NaN."""
    print("Testing spectral dimension fix...")
    
    # Create quantum spacetime with small size
    qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10)
    
    # Test spectral dimension calculation
    diffusion_times = [0.1, 1.0, 10.0, 100.0]
    
    for dt in diffusion_times:
        try:
            dim = qst.compute_spectral_dimension(dt)
            print(f"  Diffusion time {dt}: dimension = {dim}")
            
            # Check for NaN
            if np.isnan(dim):
                print(f"    ERROR: NaN detected for diffusion time {dt}")
                return False
            elif dim < 0 or dim > 10:
                print(f"    WARNING: Unphysical dimension {dim} for diffusion time {dt}")
                
        except Exception as e:
            print(f"    ERROR: Exception for diffusion time {dt}: {e}")
            return False
    
    print("  ✅ Spectral dimension fix working!")
    return True

def test_category_theory_fix():
    """Test that category theory initialization no longer hangs."""
    print("Testing category theory fix...")
    
    try:
        # Create category theory geometry with minimal size
        ctg = CategoryTheoryGeometry(dim=4, n_points=5)  # Very small
        
        print(f"  ✅ Category theory initialization successful!")
        print(f"    Objects: {len(ctg.objects)}")
        print(f"    Morphisms: {len(ctg.morphisms)}")
        print(f"    2-morphisms: {len(ctg.two_morphisms)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Category theory initialization failed: {e}")
        return False

def test_tensor_network_fix():
    """Test that tensor network builds with increased bond dimension."""
    print("Testing tensor network fix...")
    
    try:
        from quantum_gravity_framework.non_perturbative_path_integral import NonPerturbativePathIntegral
        
        # Create path integral with minimal size
        nppi = NonPerturbativePathIntegral(dim=2, lattice_size=4, beta=1.0, coupling=0.1)
        
        # Build tensor network
        tn = nppi.build_tensor_network(bond_dim=8)
        
        print(f"  ✅ Tensor network built successfully!")
        print(f"    Bond dimension: 8")
        print(f"    Lattice size: 4")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Tensor network build failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Numerical Fixes")
    print("=" * 30)
    
    tests = [
        test_spectral_dimension_fix,
        test_category_theory_fix,
        test_tensor_network_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All fixes working! Ready to run full analysis.")
    else:
        print("❌ Some fixes need more work.")

if __name__ == "__main__":
    main() 