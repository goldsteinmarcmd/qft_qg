#!/usr/bin/env python
"""
Quantum Gravity Categorical Experimental Predictions

This script connects the advanced category theory approach to quantum gravity
with potential experimental signatures and phenomenological constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import networkx as nx

from quantum_gravity_framework.category_theory import CategoryTheoryGeometry
from quantum_gravity_framework.phenomenology import QuantumGravityPhenomenology

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

def categorical_to_phenomenology(ctg):
    """
    Extract physical predictions from categorical structures.
    
    Parameters:
    -----------
    ctg : CategoryTheoryGeometry
        Categorical quantum geometry model
        
    Returns:
    --------
    dict
        Physical predictions derived from category structure
    """
    predictions = {}
    
    # 1. Calculate effective dimension
    # Use cohomology dimensions as proxy for effective spacetime dimension
    cohomology = ctg.sheaf_cohomology('measurement')
    spectral_dim = 4.0 - cohomology['H^1']['dimension'] * 0.5
    predictions['spectral_dimension'] = spectral_dim
    
    # 2. Calculate minimum length from categorical structure
    # Using the average "size" of morphisms between lowest-dim objects
    point_distances = []
    
    for morph_id, morph in ctg.morphisms.items():
        if 'distance' in morph['properties']:
            point_distances.append(morph['properties']['distance'])
            
    if point_distances:
        min_length = min(point_distances) * 1e-35  # Scale to Planck length
    else:
        min_length = 1.616e-35  # Default to Planck length
        
    predictions['min_length'] = min_length
    
    # 3. Extract time-of-flight delay parameter from category structure
    # Use the average phase of 2-morphisms as a proxy for energy-dependent effects
    phases = []
    
    for tm_id, tm in ctg.two_morphisms.items():
        if 'phase' in tm['properties']:
            phases.append(tm['properties']['phase'])
            
    if phases:
        avg_phase = np.mean(phases)
        # Map phase to LIV parameter (simple model)
        liv_parameter = 1e-20 * (avg_phase / np.pi)
    else:
        liv_parameter = 1e-20  # Default
        
    predictions['liv_parameter'] = liv_parameter
    
    # 4. Extract black hole entropy corrections
    # Using topos logic as proxy for quantum corrections to BH entropy
    # Count superposition truth values as entropy contribution
    superposition_count = 0
    
    for obj_id in ctg.objects:
        statement = {'type': 'atomic', 'object': obj_id, 'property': 'is_quantum'}
        result = ctg.evaluate_topos_logic(statement)
        if result == 'superposition':
            superposition_count += 1
            
    # Logarithmic entropy correction
    entropy_correction = -0.5 * np.log(superposition_count + 1) / np.log(len(ctg.objects))
    predictions['entropy_correction'] = entropy_correction
    
    return predictions


def simulate_observables(category_predictions):
    """
    Simulate experimental observables based on category-derived predictions.
    
    Parameters:
    -----------
    category_predictions : dict
        Predictions from categorical structure
        
    Returns:
    --------
    dict
        Simulated experimental observations
    """
    # Initialize phenomenology module
    qgp = QuantumGravityPhenomenology()
    
    # 1. GRB time delay (using derived LIV parameter)
    liv_param = category_predictions['liv_parameter']
    
    # Convert to QG energy scale
    qg_scale = 1.22e19 * (1e-20 / liv_param)  # GeV
    
    grb_sim = qgp.simulate_observable('grb_time_delay', {
        'energy_range': [1e-3, 1e3],  # GeV
        'redshift': 1.5,
        'n_photons': 200,
        'qg_scale': qg_scale,
        'qg_power': 1  # Linear suppression
    })
    
    # 2. Interferometer noise (using derived minimum length)
    min_length = category_predictions['min_length']
    qgp.planck_length = min_length  # Override with derived value
    
    interf_sim = qgp.simulate_observable('interferometer_noise', {
        'arm_length': 4000.0,  # LIGO-like
        'min_freq': 10.0,
        'max_freq': 2000.0,
        'n_points': 200
    })
    
    # 3. Black hole echo (using derived entropy correction)
    entropy_corr = category_predictions['entropy_correction']
    
    # Map entropy correction to echo model
    if entropy_corr < -0.3:
        qg_model = 'firewall'
    elif entropy_corr < -0.1:
        qg_model = 'fuzzball'
    else:
        qg_model = 'gravastar'
        
    bh_sim = qgp.simulate_observable('black_hole_echo', {
        'mass_1': 36.0,
        'mass_2': 29.0,
        'distance': 1e9,
        'qg_model': qg_model
    })
    
    return {
        'grb': grb_sim,
        'interferometer': interf_sim,
        'black_hole': bh_sim
    }


def plot_results(category_predictions, observables):
    """
    Plot the predicted experimental signatures.
    
    Parameters:
    -----------
    category_predictions : dict
        Predictions from categorical structure
    observables : dict
        Simulated experimental observations
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Plot GRB time delay
    ax1 = fig.add_subplot(221)
    
    # Sort by energy for better visualization
    energies = np.array(observables['grb']['energies'])
    delays = np.array(observables['grb']['delays'])
    sort_idx = np.argsort(energies)
    
    ax1.scatter(energies[sort_idx], delays[sort_idx], alpha=0.7)
    ax1.plot(energies[sort_idx], delays[sort_idx], 'r-', alpha=0.3)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Photon Energy (GeV)')
    ax1.set_ylabel('Time Delay (s)')
    ax1.set_title('GRB Photon Arrival Delays\nQG Energy Scale: {:.2e} GeV'.format(
        observables['grb']['qg_scale']))
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 2. Plot interferometer noise
    ax2 = fig.add_subplot(222)
    
    freqs = observables['interferometer']['frequencies']
    std_noise = observables['interferometer']['standard_noise']
    qg_noise = observables['interferometer']['qg_noise']
    total_noise = observables['interferometer']['total_noise']
    
    ax2.loglog(freqs, std_noise, 'b-', label='Standard Noise')
    ax2.loglog(freqs, qg_noise, 'r-', label='QG Noise')
    ax2.loglog(freqs, total_noise, 'k-', label='Total Noise')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Strain Noise (1/√Hz)')
    ax2.set_title('Interferometer Strain Noise\nMin Length: {:.2e} m'.format(
        category_predictions['min_length']))
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 3. Plot BH echo waveform (simplified)
    ax3 = fig.add_subplot(223)
    
    # Create a simplified waveform
    time = np.linspace(0, 0.2, 1000)  # seconds
    
    # Main merger waveform - simplified chirp
    amp = 1.0
    freq = 150.0  # Hz
    decay = 20.0  # decay rate
    
    main_wave = amp * np.sin(2 * np.pi * freq * time) * np.exp(-decay * time)
    
    # Echo parameters
    echo_delay = observables['black_hole']['echo_delay']
    echo_amp = observables['black_hole']['echo_amplitude']
    echo_freq = observables['black_hole']['echo_frequency']
    
    # Shift the main waveform to create echo
    echo_wave = np.zeros_like(time)
    echo_idx = np.where(time >= echo_delay)[0]
    if len(echo_idx) > 0:
        echo_time = time[echo_idx] - echo_delay
        echo_wave[echo_idx] = echo_amp * np.sin(2 * np.pi * echo_freq * echo_time) * np.exp(-decay * echo_time)
    
    # Plot waveform with echo
    ax3.plot(time, main_wave, 'b-', label='Main Signal')
    ax3.plot(time, echo_wave, 'r-', label='QG Echo')
    ax3.plot(time, main_wave + echo_wave, 'k-', alpha=0.3, label='Combined')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Strain')
    ax3.set_title('Black Hole Merger Echoes\nModel: {} (Entropy Corr: {:.3f})'.format(
        observables['black_hole']['qg_model'], category_predictions['entropy_correction']))
    ax3.axvline(echo_delay, color='r', linestyle='--', alpha=0.5, label='Echo Delay')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()
    
    # 4. Plot categorical structure visualization (simplified)
    ax4 = fig.add_subplot(224)
    
    # Create a visualization of the category structure using a graph
    # This is highly simplified - a full visualization would be more complex
    G = nx.DiGraph()
    
    # Add nodes for objects (just a sample for visualization)
    cmap = get_cmap('viridis')
    
    # Get a subset of objects for demonstration
    sample_obj_ids = list(category_predictions.get('sample_objects', 
                                                  ['p0', 'p1', 'p2', 'd1_o0', 'd1_o1']))
    
    # Node positions
    pos = {}
    for i, obj_id in enumerate(sample_obj_ids):
        angle = 2 * np.pi * i / len(sample_obj_ids)
        pos[obj_id] = (np.cos(angle), np.sin(angle))
        G.add_node(obj_id)
    
    # Add some edges as sample morphisms
    for i in range(len(sample_obj_ids)):
        src = sample_obj_ids[i]
        tgt = sample_obj_ids[(i+1) % len(sample_obj_ids)]
        G.add_edge(src, tgt)
        
        # Also add some cross-links for 2-morphisms
        if i % 2 == 0 and i+2 < len(sample_obj_ids):
            tgt2 = sample_obj_ids[(i+2) % len(sample_obj_ids)]
            G.add_edge(src, tgt2, style='dashed')
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=range(len(sample_obj_ids)), 
                          cmap=cmap, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edges with different styles
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style', 'solid') == 'solid']
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=1, alpha=0.5, 
                          style='dashed', edge_color='r')
    
    ax4.set_title('Categorical Structure\nSpec. Dim: {:.2f}'.format(
        category_predictions['spectral_dimension']))
    ax4.axis('off')
    
    # Set overall title and adjust layout
    plt.suptitle('Quantum Gravity Categorical Predictions', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig('qg_categorical_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Quantum Gravity: Categorical Approach to Experimental Predictions")
    print("=" * 65)
    
    # 1. Initialize categorical quantum geometry
    print("\nInitializing categorical quantum geometry...")
    ctg = CategoryTheoryGeometry(dim=4, n_points=30)
    
    # 2. Derive physical predictions from category structure
    print("Extracting physical predictions from categorical structure...")
    category_predictions = categorical_to_phenomenology(ctg)
    
    # Add sample objects for visualization
    sample_objects = []
    for obj_id in list(ctg.objects.keys())[:5]:  # Just take first 5 for demo
        sample_objects.append(obj_id)
    category_predictions['sample_objects'] = sample_objects
    
    # 3. Simulate experimental observables
    print("Simulating experimental signatures...")
    observables = simulate_observables(category_predictions)
    
    # 4. Plot results
    print("Plotting results...")
    plot_results(category_predictions, observables)
    
    # 5. Summary
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Spectral Dimension: {category_predictions['spectral_dimension']:.3f}")
    print(f"Minimum Length Scale: {category_predictions['min_length']:.3e} m")
    print(f"LIV Parameter: {category_predictions['liv_parameter']:.3e}")
    print(f"BH Entropy Correction: {category_predictions['entropy_correction']:.3f}")
    print("-" * 50)
    
    print("\nPredicted Experimental Signatures:")
    print(f"- GRB Time Delay: QG Energy Scale = {observables['grb']['qg_scale']:.3e} GeV")
    print(f"- Interferometer: QG Noise at 100 Hz = {observables['interferometer']['qg_noise'][50]:.3e} strain/√Hz")
    print(f"- BH Echo: {observables['black_hole']['qg_model']} model, delay = {observables['black_hole']['echo_delay']:.6f} s")
    
    print("\nSimulation complete. Visualizations saved to 'qg_categorical_predictions.png'") 