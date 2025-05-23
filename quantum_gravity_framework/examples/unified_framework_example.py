"""
Unified Framework Example

This script demonstrates the use of the UnifiedFramework class which bridges
quantum field theory and quantum gravity, showing their unified treatment
across energy scales.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for direct execution
sys.path.insert(0, os.path.abspath('..'))

# Import the UnifiedFramework
from quantum_gravity_framework import UnifiedFramework

def main():
    print("Creating unified QFT-QG framework...")
    
    # Create a custom dimensional profile (UV=2, IR=4)
    dim_profile = lambda E: 2.0 + 2.0 / (1.0 + (E * 0.1)**(-2))
    
    # Initialize the unified framework with this profile
    unified = UnifiedFramework(dim_profile=dim_profile)
    
    # Example 1: Comparing propagators across energy scales
    print("\nExample 1: Scalar Propagator Across Energy Scales")
    
    # Test momentum value (kept constant)
    test_momentum = 0.01
    
    # Energy scales from far below to near Planck scale
    energies = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
    
    print(f"Computing scalar propagator at fixed momentum p = {test_momentum}")
    print("Energy (Planck)  |  QFT Only  |  With QG Corrections")
    print("-" * 55)
    
    for energy in energies:
        unified.set_energy_scale(energy)
        
        # Compute propagator with and without QG corrections
        prop_qft = unified.compute_propagator('scalar', test_momentum, include_qg_corrections=False)
        prop_with_qg = unified.compute_propagator('scalar', test_momentum, include_qg_corrections=True)
        
        # Print comparison
        print(f"{energy:12.6e}  |  {prop_qft:10.4e}  |  {prop_with_qg:10.4e}")
    
    # Example 2: Effective action calculation
    print("\nExample 2: Effective Action Calculation")
    
    # Simple field configuration
    field_config = np.ones((5, 5)) * 0.1
    
    print("Computing effective action for a sample field configuration")
    print("Energy (Planck)  |  Dimension  |  Action Value")
    print("-" * 55)
    
    for energy in [1e-5, 1e-3, 0.05, 0.5]:
        # Set energy scale
        unified.set_energy_scale(energy)
        
        # Compute effective action
        action_result = unified.compute_effective_action(energy, field_config)
        
        # Print result
        print(f"{energy:12.6e}  |  {action_result['dimension']:10.4f}  |  {action_result['action_value']:10.4e}")
    
    # Example 3: Demonstrate and visualize QFT emergence
    print("\nExample 3: Demonstrating QFT Emergence from QG")
    
    # Compute comparison across energy scales
    emergence_results = unified.demonstrate_qft_emergence(
        energy_range=(1e-6, 1.0), 
        num_points=40
    )
    
    # Create visualization
    fig = unified.visualize_qft_emergence(emergence_results)
    plt.savefig("qft_emergence_from_qg.png")
    print("  Visualization saved to 'qft_emergence_from_qg.png'")
    
    # Example 4: Compute transition amplitudes
    print("\nExample 4: Transition Amplitudes")
    
    # Define sample states
    initial_state = {'particle_type': 'scalar', 'energy': 0.5}
    final_state = {'particle_type': 'scalar', 'energy': 0.6}
    
    print("Computing transition amplitudes across energy scales")
    print("Energy (Planck)  |  Amplitude  |  Framework")
    print("-" * 55)
    
    for energy in [1e-5, 1e-3, 0.05, 0.5]:
        # Compute amplitude
        amplitude_result = unified.compute_transition_amplitudes(
            initial_state, final_state, energy
        )
        
        # Print result
        print(f"{energy:12.6e}  |  {abs(amplitude_result['amplitude']):10.4f}  |  {amplitude_result['framework']}")
    
    # Example 5: QFT-QG mapping
    print("\nExample 5: Mapping Between QFT and QG")
    
    # Create a QFT object (simplified representation)
    qft_object = {
        'particle_type': 'scalar',
        'mass': 0.001,
        'momentum': 0.01,
        'wavefunction': np.array([0.1, 0.2, 0.1])
    }
    
    print("Mapping a QFT object to QG representation")
    
    # Low energy scale (QFT regime)
    low_energy = 1e-5
    unified.set_energy_scale(low_energy)
    
    # Map to QG
    qg_rep = unified.qft_to_qg_mapping(qft_object, low_energy)
    
    print(f"At energy E = {low_energy:.2e}:")
    print(f"  Original QFT object: particle={qft_object['particle_type']}, mass={qft_object['mass']:.4f}")
    print(f"  QG representation: {qg_rep}")
    
    # High energy scale (QG regime)
    high_energy = 0.5
    unified.set_energy_scale(high_energy)
    
    # Map to QG
    qg_rep = unified.qft_to_qg_mapping(qft_object, high_energy)
    
    print(f"At energy E = {high_energy:.2f}:")
    print(f"  Original QFT object: particle={qft_object['particle_type']}, mass={qft_object['mass']:.4f}")
    print(f"  QG representation: {qg_rep}")

if __name__ == "__main__":
    main() 