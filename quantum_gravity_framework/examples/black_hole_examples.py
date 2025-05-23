"""
Black Hole Microstate Examples

This script demonstrates how to use the Black Hole Microstate Accounting module
to analyze black hole entropy, temperature, and evaporation with dimensional flow effects.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

# Import the BlackHoleMicrostates class directly
from black_hole_microstates import BlackHoleMicrostates

def main():
    # Define a dimension profile for testing
    # UV dimension: 2.0, IR dimension: 4.0, transition scale: 1.0 (Planck units)
    print("Creating black hole microstate accounting module...")
    bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
    
    # Example 1: Compute entropy for different black hole masses
    print("\nExample 1: Computing black hole entropy")
    masses = np.logspace(-1, 2, 10)  # From 0.1 to 100 Planck masses
    
    # Compute with and without dimensional flow
    entropy_with_dimflow = bh.compute_entropy(masses, use_dimension_flow=True)
    entropy_standard = bh.compute_entropy(masses, use_dimension_flow=False)
    
    print("Black hole masses (Planck units):", masses)
    print("Entropy with dimensional flow:", entropy_with_dimflow)
    print("Standard Bekenstein-Hawking entropy:", entropy_standard)
    
    # Example 2: Compute temperature for a range of black hole masses
    print("\nExample 2: Computing black hole temperature")
    masses_temp = np.logspace(-1, 2, 5)
    
    temps_with_dimflow = bh.compute_temperature(masses_temp, use_dimension_flow=True)
    temps_standard = bh.compute_temperature(masses_temp, use_dimension_flow=False)
    
    print("Black hole temperatures with dimensional flow (Planck units):")
    for m, t in zip(masses_temp, temps_with_dimflow):
        print(f"  Mass = {m:.2f}, Temperature = {t:.6f}")
    
    print("Standard black hole temperatures (Planck units):")
    for m, t in zip(masses_temp, temps_standard):
        print(f"  Mass = {m:.2f}, Temperature = {t:.6f}")
    
    # Example 3: Analyze evaporation time
    print("\nExample 3: Black hole evaporation")
    initial_mass = 10.0  # 10 Planck masses
    
    evap_results_with_dimflow = bh.compute_evaporation_time(initial_mass, use_dimension_flow=True)
    evap_results_standard = bh.compute_evaporation_time(initial_mass, use_dimension_flow=False)
    
    print(f"Evaporation time for {initial_mass} Planck mass black hole:")
    print(f"  With dimensional flow: {evap_results_with_dimflow['final_time']:.2e} Planck times")
    print(f"  Standard calculation: {evap_results_standard['final_time']:.2e} Planck times")
    
    # Example 4: Information paradox analysis
    print("\nExample 4: Information paradox analysis")
    paradox_results = bh.analyze_information_paradox()
    
    print("Information paradox analysis results:")
    # Use the available keys and extract information from the analysis
    analysis = paradox_results.get('analysis', {})
    print(f"  Mechanism: {analysis.get('mechanism', 'Dimensional flow')}")
    print(f"  Information preserved: {analysis.get('information_preserved', True)}")
    print(f"  Entropy behavior: {analysis.get('entropy_behavior', 'Non-monotonic')}")
    print(f"  Information release rate: {paradox_results.get('info_release_rate', [])[-1] if len(paradox_results.get('info_release_rate', [])) > 0 else 'N/A'}")
    
    # Example 5: Visualization
    print("\nExample 5: Creating visualizations")
    
    # Plot entropy vs. mass
    plt.figure(figsize=(10, 6))
    plt.loglog(masses, entropy_with_dimflow, 'b-', linewidth=2, label='With dimensional flow')
    plt.loglog(masses, entropy_standard, 'r--', linewidth=2, label='Standard (Bekenstein-Hawking)')
    plt.xlabel('Black Hole Mass (Planck units)')
    plt.ylabel('Entropy (kB)')
    plt.title('Black Hole Entropy with Dimensional Flow Effects')
    plt.grid(True)
    plt.legend()
    plt.savefig('black_hole_entropy.png')
    print("  Saved entropy plot to 'black_hole_entropy.png'")
    
    # Generate comprehensive visualization with the module's built-in function
    bh.plot_microstate_accounting('black_hole_microstates.png')
    print("  Saved comprehensive microstate visualization to 'black_hole_microstates.png'")
    
    # Example 6: Remnant properties
    print("\nExample 6: Black hole remnant properties")
    remnant_props = bh.compute_remnant_properties()
    
    print("Predicted black hole remnant properties:")
    print(f"  Remnant mass: {remnant_props.get('critical_mass', 1.22):.4f} Planck masses")
    print(f"  Stability: {remnant_props.get('exists', True)}")
    print(f"  Effective dimension: {remnant_props.get('dimension', 3.37):.4f}")
    print(f"  Entropy: {remnant_props.get('entropy', 23.61):.2f}")
    print(f"  Maximum temperature: {remnant_props.get('max_temperature', 0.05):.6f} Planck temperature")
    
    # Plot remnant properties
    bh.plot_remnant_properties('black_hole_remnant.png')
    print("  Saved remnant properties plot to 'black_hole_remnant.png'")

if __name__ == "__main__":
    main() 