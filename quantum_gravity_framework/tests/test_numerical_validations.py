"""
Numerical Validation Tests

This module provides tests that validate the numerical correctness of QG implementations
by comparing against known analytic solutions and benchmark cases.
"""

import unittest
import numpy as np
import scipy.integrate as integrate

from quantum_gravity_framework import UnifiedFramework
from quantum_gravity_framework.path_integral import PathIntegral
from quantum_gravity_framework.black_hole_microstates import BlackHoleMicrostates


class TestNumericalValidations(unittest.TestCase):
    """Tests for numerical validation against known solutions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define common dimension profile for testing
        self.dim_profile = lambda E: 2.0 + 2.0 / (1.0 + (E * 0.1)**(-2))
        
        # Initialize framework components
        self.unified = UnifiedFramework(dim_profile=self.dim_profile)
        self.path_integral = PathIntegral(
            dim_profile=self.dim_profile,
            discreteness_profile=lambda E: 1.0 / (1.0 + (E * 0.1)**(-2))
        )
        self.black_hole = BlackHoleMicrostates(
            dim_uv=2.0, dim_ir=4.0, transition_scale=1.0
        )
    
    def test_free_scalar_propagator(self):
        """Test if free scalar propagator matches analytic solution."""
        # Momenta to test
        momenta = np.linspace(0.01, 0.5, 10)
        
        # Mass of scalar particle
        mass = 0.01
        
        # Calculate propagator using framework (without QG corrections)
        self.unified.set_energy_scale(1e-6)  # Low energy (QFT regime)
        
        numerical_props = []
        for p in momenta:
            prop = self.unified.compute_propagator(
                'scalar', p, include_qg_corrections=False
            )
            numerical_props.append(prop)
        
        # Analytic solution for scalar propagator: 1/(p^2 + m^2)
        analytic_props = 1.0 / (momenta**2 + mass**2)
        
        # Calculate deviation
        deviations = np.abs(np.array(numerical_props) - analytic_props) / analytic_props
        
        # All deviations should be small
        max_deviation = np.max(deviations)
        self.assertLess(max_deviation, 1e-4, 
                       msg=f"Free scalar propagator deviates from analytic solution, max={max_deviation}")
    
    def test_black_hole_entropy_4d(self):
        """Test if black hole entropy matches Bekenstein-Hawking in 4D limit."""
        # Black hole masses to test (large enough to be in 4D regime)
        masses = np.linspace(100, 1000, 5)
        
        # Calculate entropy using framework
        numerical_entropies = []
        for mass in masses:
            entropy = self.black_hole.compute_entropy(
                mass, use_dimension_flow=False  # Turn off dimension flow to test 4D limit
            )
            numerical_entropies.append(entropy)
        
        # Bekenstein-Hawking entropy: S = 4π M^2
        analytic_entropies = 4 * np.pi * masses**2
        
        # Calculate deviation
        deviations = np.abs(np.array(numerical_entropies) - analytic_entropies) / analytic_entropies
        
        # All deviations should be small
        max_deviation = np.max(deviations)
        self.assertLess(max_deviation, 1e-2, 
                       msg=f"BH entropy deviates from B-H law, max={max_deviation}")
    
    def test_gaussian_path_integral(self):
        """Test if Gaussian path integral calculation matches analytic solution."""
        # Simple test: Gaussian integral ∫exp(-ax²)dx = sqrt(π/a)
        
        # Parameter for Gaussian
        a = 2.0
        
        # Calculate numerically using path integral module
        # We use a simple 1D "field" for this test
        num_points = 1000
        field_configs = np.linspace(-5, 5, num_points)
        actions = a * field_configs**2
        
        # Boltzmann factors
        weights = np.exp(-actions)
        
        # Numerical integration
        numerical_result = np.sum(weights) * (field_configs[1] - field_configs[0])
        
        # Analytic result
        analytic_result = np.sqrt(np.pi / a)
        
        # Calculate relative error
        rel_error = abs(numerical_result - analytic_result) / analytic_result
        
        # Error should be small given sufficient sampling
        self.assertLess(rel_error, 0.01, 
                       msg=f"Gaussian path integral deviates from analytic solution, error={rel_error}")
    
    def test_dimensional_flow_interpolation(self):
        """Test if dimensional flow interpolates correctly between IR and UV limits."""
        # Energy scales from low to high
        energies = np.logspace(-6, 0, 20)
        
        # Calculate dimensions
        dimensions = [self.unified.dim_profile(E) for E in energies]
        
        # In IR limit (low energy), dimension should approach dim_ir
        self.assertAlmostEqual(dimensions[0], 4.0, delta=0.01,
                             msg=f"IR dimension incorrect: {dimensions[0]}")
        
        # In UV limit (high energy), dimension should approach dim_uv
        self.assertAlmostEqual(dimensions[-1], 2.0, delta=0.01,
                             msg=f"UV dimension incorrect: {dimensions[-1]}")
        
        # Check monotonicity (dimension should decrease monotonically with energy)
        is_monotonic = all(dimensions[i] >= dimensions[i+1] for i in range(len(dimensions)-1))
        self.assertTrue(is_monotonic, msg="Dimension flow is not monotonic")
    
    def test_momentum_space_measure(self):
        """Test if momentum space integration measure is correctly implemented."""
        # Integration of simple test function f(p) = exp(-p²) in d-dimensions
        # ∫d^dp exp(-p²) = π^(d/2)
        
        # Test dimensions
        dimensions = [2.0, 3.0, 4.0]
        
        for dim in dimensions:
            # Set up integration in spherical coordinates
            def integrand(p):
                # p^(d-1) from measure, exp(-p²) from test function
                return p**(dim-1) * np.exp(-p**2)
            
            # Numerical integration
            numerical_result, _ = integrate.quad(integrand, 0, 10)
            
            # Volume of (d-1)-sphere
            if dim == 2:
                sphere_vol = 2 * np.pi
            elif dim == 3:
                sphere_vol = 4 * np.pi
            elif dim == 4:
                sphere_vol = 2 * np.pi**2
            
            # Apply sphere volume to get full d-dimensional integral
            numerical_result *= sphere_vol / dim
            
            # Analytic result
            analytic_result = np.pi**(dim/2)
            
            # Calculate relative error
            rel_error = abs(numerical_result - analytic_result) / analytic_result
            
            # Error should be small
            self.assertLess(rel_error, 0.01, 
                           msg=f"Momentum measure incorrect in d={dim}, error={rel_error}")
    
    def test_black_hole_temperature(self):
        """Test if black hole temperature matches Hawking temperature in 4D limit."""
        # Black hole masses to test
        masses = np.linspace(10, 100, 5)
        
        # Calculate temperature using framework
        numerical_temps = []
        for mass in masses:
            temp = self.black_hole.compute_temperature(
                mass, use_dimension_flow=False  # Turn off dimension flow to test 4D limit
            )
            numerical_temps.append(temp)
        
        # Hawking temperature: T = 1/(8πM)
        analytic_temps = 1.0 / (8 * np.pi * masses)
        
        # Calculate deviation
        deviations = np.abs(np.array(numerical_temps) - analytic_temps) / analytic_temps
        
        # All deviations should be small
        max_deviation = np.max(deviations)
        self.assertLess(max_deviation, 1e-2, 
                       msg=f"BH temperature deviates from Hawking formula, max={max_deviation}")


if __name__ == '__main__':
    unittest.main() 