"""
Integration Tests for QFT and QG Integration

This module tests the consistency and correctness of integrating Quantum Field Theory (QFT)
and Quantum Gravity (QG) components across different energy scales.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

from quantum_gravity_framework import UnifiedFramework
from quantum_gravity_framework.path_integral import PathIntegral
from quantum_gravity_framework.unified_coupling import UnifiedCouplingFramework
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class TestQFTQGIntegration(unittest.TestCase):
    """Test the integration between QFT and QG components."""
    
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
        self.coupling = UnifiedCouplingFramework(dim_profile=self.dim_profile)
        self.rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
    
    def test_dimension_consistency(self):
        """Test if dimension profiles are consistent across frameworks."""
        # Test at different energy scales
        energies = [1e-6, 1e-3, 0.1, 0.5, 1.0]
        
        for E in energies:
            dim_unified = self.unified.dim_profile(E)
            dim_rg = self.rg.compute_spectral_dimension(E)
            
            # Dimensions should be consistent across frameworks
            # Allow for small numerical differences
            self.assertAlmostEqual(dim_unified, dim_rg, delta=1e-10,
                                  msg=f"Dimension inconsistency at E={E}")
    
    def test_propagator_low_energy_limit(self):
        """Test if propagators reduce to QFT forms at low energies."""
        # Low energy scale (should be in QFT regime)
        low_energy = 1e-6
        
        # Test momentum
        test_momentum = 0.01
        
        # Set energy scale
        self.unified.set_energy_scale(low_energy)
        
        # Compute propagator with and without QG corrections
        prop_with_qg = self.unified.compute_propagator(
            'scalar', test_momentum, include_qg_corrections=True
        )
        prop_without_qg = self.unified.compute_propagator(
            'scalar', test_momentum, include_qg_corrections=False
        )
        
        # At low energies, corrections should be negligible
        # Relative difference should be very small
        rel_diff = abs(prop_with_qg - prop_without_qg) / abs(prop_without_qg)
        
        self.assertLess(rel_diff, 1e-4, 
                       msg=f"QG corrections not negligible at low energy E={low_energy}")
    
    def test_coupling_unification(self):
        """Test if couplings unify correctly at high energies."""
        # High energy scale (near Planck)
        high_energy = 0.8
        
        # Compute running couplings
        couplings = self.coupling.compute_running_couplings(
            energy_range=(1e-6, 1.0),
            couplings=['g_1', 'g_2', 'g_3'],
            num_points=50
        )
        
        # Extract coupling values at high energy
        g_values = []
        energy_idx = np.abs(couplings['energy_scales'] - high_energy).argmin()
        
        for name in ['g_1', 'g_2', 'g_3']:
            g_values.append(couplings['trajectories'][name][energy_idx])
        
        # Check if couplings converge at high energy
        # Calculate relative spread
        mean_g = np.mean(g_values)
        max_dev = np.max(np.abs(np.array(g_values) - mean_g)) / mean_g
        
        # At high energies, couplings should be close
        self.assertLess(max_dev, 0.1, 
                       msg=f"Couplings don't unify at high energy, max deviation={max_dev}")
    
    def test_effective_action_consistency(self):
        """Test if effective action is consistent across energy scales."""
        # Test at different energy scales
        energies = [1e-6, 1e-3, 0.1, 0.5]
        
        # Simple field configuration
        field_config = np.ones((5, 5)) * 0.1
        
        # Results for each energy
        actions = []
        
        for E in energies:
            # Set energy scale
            self.unified.set_energy_scale(E)
            
            # Compute effective action
            action_result = self.unified.compute_effective_action(E, field_config)
            actions.append(action_result['action_value'])
        
        # Check for monotonic behavior (action should vary smoothly)
        for i in range(1, len(actions)):
            # Action shouldn't jump discontinuously
            rel_change = abs(actions[i] - actions[i-1]) / abs(actions[i-1])
            
            self.assertLess(rel_change, 10.0, 
                           msg=f"Action changes discontinuously between E={energies[i-1]} and E={energies[i]}")
    
    def test_scattering_amplitude_consistency(self):
        """Test if scattering amplitudes are consistent between frameworks."""
        # Test scattering process
        process = '2to2'
        particle_types = ['scalar', 'scalar']
        
        # Test at different energies
        energies = [1e-5, 1e-3, 0.05]
        
        for E in energies:
            # Compute via path integral
            pi_result = self.path_integral.compute_scattering_amplitude(
                process, E, particle_types
            )
            
            # Compute via unified framework
            self.unified.set_energy_scale(E)
            initial_state = {'particle_type': 'scalar', 'energy': E}
            final_state = {'particle_type': 'scalar', 'energy': E}
            uf_result = self.unified.compute_transition_amplitudes(
                initial_state, final_state, E
            )
            
            # Extract amplitudes
            pi_amp = pi_result['total_amplitude']
            uf_amp = uf_result['amplitude']
            
            # Calculate relative difference
            if abs(pi_amp) > 1e-10 and abs(uf_amp) > 1e-10:
                rel_diff = abs(pi_amp - uf_amp) / abs(pi_amp)
                
                # Allow for numerical differences, but they should be small
                self.assertLess(rel_diff, 0.5,
                              msg=f"Amplitude inconsistency at E={E}, rel_diff={rel_diff}")
    
    def test_qft_emergence(self):
        """Test if QFT properly emerges from QG at low energies."""
        # Compute emergence data
        emergence_results = self.unified.demonstrate_qft_emergence(
            energy_range=(1e-6, 1.0), num_points=20
        )
        
        # Check QFT limit: at low energies, deviation should be small
        low_E_idx = 0  # Lowest energy index
        deviation = emergence_results['propagators']['deviation'][low_E_idx]
        
        self.assertLess(deviation, 1e-3,
                      msg=f"QFT doesn't emerge properly at low energy, deviation={deviation}")
        
        # Dimension should approach 4 at low energies
        low_E_dim = emergence_results['dimensions'][low_E_idx]
        self.assertAlmostEqual(low_E_dim, 4.0, delta=0.1,
                             msg=f"Dimension doesn't approach 4 at low energy, dim={low_E_dim}")
    
    def test_mapping_consistency(self):
        """Test if QFT-QG mappings are consistent."""
        # Define test QFT object
        qft_object = {
            'particle_type': 'scalar',
            'mass': 0.001,
            'momentum': 0.01,
            'wavefunction': np.array([0.1, 0.2, 0.1])
        }
        
        # Map QFT → QG → QFT at low energy
        low_energy = 1e-5
        self.unified.set_energy_scale(low_energy)
        
        qg_object = self.unified.qft_to_qg_mapping(qft_object, low_energy)
        qft_back = self.unified.qg_to_qft_mapping(qg_object, low_energy)
        
        # Original properties should be preserved in round-trip
        self.assertEqual(qft_back['particle_type'], qft_object['particle_type'])
        self.assertAlmostEqual(qft_back['mass'], qft_object['mass'], delta=1e-6)
        self.assertAlmostEqual(qft_back['momentum'], qft_object['momentum'], delta=1e-6)


if __name__ == '__main__':
    unittest.main() 