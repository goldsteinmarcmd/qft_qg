"""
Advanced Quantum Gravity Framework: Usage Examples

This module provides examples of how to use the quantum gravity framework
components together to model quantum spacetime and solve QFT problems in
quantum gravity contexts.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import networkx as nx
import pandas as pd
import time

from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms, RelationalTimeFramework
from quantum_gravity_framework.holographic_duality import ExtendedHolographicDuality
from quantum_gravity_framework.quantum_black_hole import QuantumBlackHole
from quantum_gravity_framework.category_theory import CategoryTheoryGeometry
from quantum_gravity_framework.high_energy_collisions import HighEnergyCollisionSimulator
from quantum_gravity_framework.numerical_simulations import DiscretizedSpacetime, PathIntegralMonteCarlo, TensorNetworkStates


def demonstrate_uv_regularization():
    """
    Demonstrate how quantum gravity provides UV regularization for QFT.
    
    One of the key insights from QG is the natural regularization of
    ultraviolet divergences in quantum field theories.
    """
    print("\n=== UV Regularization from Quantum Gravity ===")
    
    # Create a quantum spacetime structure
    qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0)
    
    # Define a range of energy scales (from IR to UV)
    energy_scales = np.logspace(-3, 3, 100)  # From 10^-3 to 10^3 in Planck units
    
    # Compute spectral dimension at each scale
    dimensions = [qst.compute_spectral_dimension(1.0/E) for E in energy_scales]
    
    # Standard QFT vacuum energy (no regularization)
    # Using dim=4 for standard 4D spacetime
    qft_vacuum_energy = [E**4 for E in energy_scales]
    
    # QG corrected vacuum energy
    # Using the scale-dependent dimension to regularize
    qg_vacuum_energy = [E**dimensions[i] for i, E in enumerate(energy_scales)]
    
    # For high energies, dimension flow to 2 helps regulate divergences
    # High energy/short distance limit: multiply by exp(-E²)
    for i, E in enumerate(energy_scales):
        if E > 1.0:  # Above Planck scale
            qg_vacuum_energy[i] *= np.exp(-(E/1.0)**2)
            
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Spectral dimension flow
    plt.subplot(2, 1, 1)
    plt.semilogx(energy_scales, dimensions)
    plt.xlabel("Energy Scale (Planck units)")
    plt.ylabel("Spectral Dimension")
    plt.title("Spectral Dimension Flow in Quantum Spacetime")
    plt.grid(True)
    plt.axhline(y=4.0, color='r', linestyle='--', alpha=0.5, label="Classical dimension")
    plt.axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label="Asymptotic dimension")
    plt.legend()
    
    # Plot 2: Vacuum energy comparison
    plt.subplot(2, 1, 2)
    plt.loglog(energy_scales, qft_vacuum_energy, 'r-', label="Standard QFT (E⁴)")
    plt.loglog(energy_scales, qg_vacuum_energy, 'g-', label="QG Regulated")
    plt.xlabel("Energy Scale (Planck units)")
    plt.ylabel("Vacuum Energy Density")
    plt.title("UV Regularization of Vacuum Energy from Quantum Gravity")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("uv_regularization.png")
    print("Generated plot: uv_regularization.png")


def demonstrate_black_hole_information():
    """
    Demonstrate black hole information recovery using quantum gravity.
    
    The black hole information paradox is a critical test for quantum gravity
    approaches, requiring unitary evolution and information preservation.
    """
    print("\n=== Black Hole Information Paradox Resolution ===")
    
    # Create a quantum black hole
    qbh = QuantumBlackHole(mass=100.0)  # Mass in Planck units
    
    # Compute the Page curve for black hole evaporation
    page_data = qbh.compute_page_curve(n_steps=100)
    
    # Extract data
    stages = page_data['evaporation_stages']
    hawking_entropy = page_data['hawking_entropy']
    page_entropy = page_data['page_entropy']
    remnant_entropy = page_data['remnant_entropy']
    
    # Compute horizon area throughout evaporation
    areas = [16 * np.pi * (100.0 * (1-s)**(1/3))**2 for s in stages]
    
    # Compute interior quantum geometry at mid-evaporation
    mid_interior = qbh.interior_quantum_geometry(time_steps=8)
    interior_curvature = list(mid_interior['curvature'].values())
    
    # Compute Hawking radiation correlations
    correlations = qbh.compute_hawking_correlations(n_modes=6)
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Page curve
    plt.subplot(2, 2, 1)
    plt.plot(stages, hawking_entropy, 'r-', label="Hawking (information loss)")
    plt.plot(stages, page_entropy, 'g-', label="Page (unitary)")
    plt.plot(stages, remnant_entropy, 'b--', label="Remnant")
    plt.xlabel("Evaporation Progress")
    plt.ylabel("Entropy")
    plt.title("Page Curve for Black Hole Evaporation")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Quantum correlations in radiation
    plt.subplot(2, 2, 2)
    plt.imshow(correlations['qg_correlations'] - correlations['hawking_correlations'], 
               cmap='coolwarm', origin='lower')
    plt.colorbar(label="Additional Correlations")
    plt.xlabel("Mode j")
    plt.ylabel("Mode i")
    plt.title("Quantum Gravity Correlations in Radiation")
    
    # Plot 3: Horizon area vs entropy
    plt.subplot(2, 2, 3)
    plt.plot(areas, hawking_entropy, 'r-', label="Hawking entropy")
    plt.xlabel("Horizon Area (Planck units)")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Horizon Area")
    plt.grid(True)
    area_entropy_line = [a/4 for a in areas]  # Bekenstein-Hawking: S = A/4
    plt.plot(areas, area_entropy_line, 'k--', label="S = A/4")
    plt.legend()
    
    # Plot 4: Histogram of interior curvature values (singularity resolution)
    plt.subplot(2, 2, 4)
    plt.hist(interior_curvature, bins=20)
    plt.xlabel("Quantum Curvature")
    plt.ylabel("Frequency")
    plt.title("Interior Quantum Curvature Distribution\n(Bounded = Singularity Resolved)")
    plt.grid(True)
    plt.axvline(x=1.0, color='r', linestyle='--', label="Classical Singularity")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("black_hole_information.png")
    print("Generated plot: black_hole_information.png")


def demonstrate_holographic_principle():
    """
    Demonstrate the holographic principle using extended holographic duality.
    
    The holographic principle is a cornerstone of quantum gravity,
    suggesting that the information in a volume of space can be encoded
    on its boundary.
    """
    print("\n=== Holographic Principle Demonstration ===")
    
    # Create a holographic model
    holo = ExtendedHolographicDuality(boundary_dim=3, bulk_dim=4)
    
    # Select a series of boundary regions of increasing size
    boundary_nodes = holo.bulk_geometry['boundary_nodes']
    n_regions = 8
    region_sizes = [len(boundary_nodes) // (2**i) for i in range(n_regions)]
    region_sizes.sort()  # Increasing size
    
    # Compute entanglement entropy for each region using Ryu-Takayanagi
    areas = []
    entropies = []
    region_volumes = []
    
    for size in region_sizes:
        region = boundary_nodes[:size]
        rt_result = holo.compute_ryu_takayanagi(region)
        
        areas.append(rt_result['minimal_surface_area'])
        entropies.append(rt_result['holographic_entropy'])
        region_volumes.append(size)  # Simplified volume measure
    
    # Compute entanglement wedge for a specific region
    mid_region = boundary_nodes[:len(boundary_nodes)//2]
    wedge_result = holo.modular_hamiltonian(mid_region)
    entanglement_wedge = wedge_result['entanglement_wedge']
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Area vs Entropy (test of Ryu-Takayanagi)
    plt.subplot(2, 2, 1)
    plt.plot(areas, entropies, 'bo-')
    plt.xlabel("Minimal Surface Area")
    plt.ylabel("Entanglement Entropy")
    plt.title("Ryu-Takayanagi Formula Verification")
    plt.grid(True)
    # Plot S = A/4G line
    area_range = np.linspace(min(areas), max(areas), 100)
    plt.plot(area_range, area_range / 4, 'r--', label="S = A/4G")
    plt.legend()
    
    # Plot 2: Volume vs Area (test of holographic scaling)
    plt.subplot(2, 2, 2)
    plt.plot(region_volumes, areas, 'go-')
    plt.xlabel("Boundary Region Size")
    plt.ylabel("Minimal Surface Area")
    plt.title("Holographic Scaling")
    plt.grid(True)
    # Plot expected scaling for comparison (depends on geometry)
    plt.plot(region_volumes, [v**(2/3) * 2 for v in region_volumes], 'r--', 
             label="Expected Scaling")
    plt.legend()
    
    # Plot 3: Volume vs Entropy (subadditivity test)
    plt.subplot(2, 2, 3)
    plt.plot(region_volumes, entropies, 'mo-')
    plt.xlabel("Boundary Region Size")
    plt.ylabel("Entanglement Entropy")
    plt.title("Entropy Scaling with Region Size")
    plt.grid(True)
    # Plot logarithmic scaling for comparison (conformal field theory)
    plt.plot(region_volumes, [np.log(v) * 3 for v in region_volumes], 'r--', 
             label="Log Scaling (CFT)")
    plt.legend()
    
    # Plot 4: Histogram of curvature in entanglement wedge
    plt.subplot(2, 2, 4)
    curvatures = [holo.bulk_geometry['curvature'].get(node, 0) for node in entanglement_wedge]
    plt.hist(curvatures, bins=20)
    plt.xlabel("Bulk Curvature")
    plt.ylabel("Frequency")
    plt.title("Curvature Distribution in Entanglement Wedge")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("holographic_principle.png")
    print("Generated plot: holographic_principle.png")


def demonstrate_quantum_reference_frames():
    """
    Demonstrate quantum reference frames using relational dynamics.
    
    The problem of time in quantum gravity can be addressed through
    relational observables, where physical quantities are measured
    relative to other dynamical variables.
    """
    print("\n=== Quantum Reference Frames and Relational Time ===")
    
    # Create a relational framework with several quantum systems
    rtf = RelationalTimeFramework(n_systems=5, dim_system=8)
    
    # Define a parameter range for evolution
    param_values = np.linspace(-2, 2, 50)
    
    # Evolve different systems relative to each other
    results1 = rtf.evolve_relational(0, 1, param_values)  # System 0 as clock for system 1
    results2 = rtf.evolve_relational(1, 0, param_values)  # System 1 as clock for system 0
    results3 = rtf.evolve_relational(0, 2, param_values)  # System 0 as clock for system 2
    
    # Extract data
    clock_0_for_1 = [r['reference_param'] for r in results1]
    system_1_evol = [r['target_expectation'] for r in results1]
    
    clock_1_for_0 = [r['reference_param'] for r in results2]
    system_0_evol = [r['target_expectation'] for r in results2]
    
    clock_0_for_2 = [r['reference_param'] for r in results3]
    system_2_evol = [r['target_expectation'] for r in results3]
    
    # Compute conditional probabilities between systems
    cond_probs = []
    reference_values = np.linspace(-1, 1, 10)
    
    for ref_val in reference_values:
        # Probability of system 1 having value 0.5 given system 0 has value ref_val
        prob = rtf.conditional_probability(0, rtf.systems[0]['observables']['Z'], ref_val,
                                        1, rtf.systems[1]['observables']['X'], 0.5)
        cond_probs.append(prob['conditional_probability'])
    
    # Compute some correlations between systems in various "temporal" arrangements
    correlations = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            # Simple correlation measure between systems
            state_i = rtf.systems[i]['state']
            state_j = rtf.systems[j]['state']
            
            # Correlation is roughly the overlap of states
            corr = np.abs(np.vdot(state_i, state_j))**2
            correlations[i, j] = corr
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Relational evolution of different systems
    plt.subplot(2, 2, 1)
    plt.plot(clock_0_for_1, system_1_evol, 'r-', label="System 1 rel. to 0")
    plt.plot(clock_0_for_2, system_2_evol, 'g-', label="System 2 rel. to 0")
    plt.xlabel("Clock System 0 Parameter")
    plt.ylabel("Target System Expectation Value")
    plt.title("Relational Evolution")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Reciprocal relational evolution
    plt.subplot(2, 2, 2)
    plt.plot(clock_0_for_1, system_1_evol, 'r-', label="System 1 rel. to 0")
    plt.plot(clock_1_for_0, system_0_evol, 'b-', label="System 0 rel. to 1")
    plt.xlabel("Clock Parameter")
    plt.ylabel("Target System Expectation Value")
    plt.title("Reciprocal Relational Evolution")
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Conditional probabilities
    plt.subplot(2, 2, 3)
    plt.plot(reference_values, cond_probs, 'mo-')
    plt.xlabel("Reference System Value")
    plt.ylabel("Conditional Probability")
    plt.title("P(System 1 = 0.5 | System 0 = x)")
    plt.grid(True)
    
    # Plot 4: Correlation matrix between systems
    plt.subplot(2, 2, 4)
    plt.imshow(correlations, cmap='viridis', origin='lower')
    plt.colorbar(label="Correlation Strength")
    plt.xlabel("System j")
    plt.ylabel("System i")
    plt.title("Correlations Between Systems")
    
    plt.tight_layout()
    plt.savefig("quantum_reference_frames.png")
    print("Generated plot: quantum_reference_frames.png")


def demonstrate_category_theory_geometry():
    """
    Demonstrate the category theory approach to quantum geometry.
    
    Category theory provides powerful algebraic structures for modeling
    quantum spacetime and its symmetries.
    """
    print("\n=== Category Theory Geometry Demonstration ===")
    
    # Create a category theory model of quantum geometry
    ctg = CategoryTheoryGeometry(dim=4, n_points=30)
    
    # Analyze the categorical structure
    n_objects = len(ctg.objects)
    n_morphisms = len(ctg.morphisms)
    n_2morphisms = len(ctg.two_morphisms)
    
    # Compute quantum groupoid structure
    qg_structure = ctg.compute_quantum_groupoid_structure()
    
    # Compute curvature between various object pairs
    curvatures = {}
    obj_ids = list(ctg.objects.keys())
    
    for i in range(min(10, len(obj_ids))):
        for j in range(i+1, min(10, len(obj_ids))):
            obj1, obj2 = obj_ids[i], obj_ids[j]
            curv = ctg.compute_2morphism_curvature(obj1, obj2)
            curvatures[(obj1, obj2)] = curv
    
    # Extract data from 2-inner products
    inner_products = []
    phases = []
    
    for (m1, m2), value in ctg.two_inner_products.items():
        if isinstance(value, complex):
            inner_products.append(abs(value))
            phases.append(np.angle(value))
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Structure counts
    plt.subplot(2, 2, 1)
    categories = ['Objects', 'Morphisms', '2-Morphisms', 'Invertible\nMorphisms', 'Orbits']
    counts = [n_objects, n_morphisms, n_2morphisms, 
              len(qg_structure['invertible_morphisms']), len(qg_structure['orbits'])]
    
    plt.bar(categories, counts, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylabel("Count")
    plt.title("Category Structure")
    plt.xticks(rotation=45)
    
    # Plot 2: Curvature distribution
    plt.subplot(2, 2, 2)
    plt.hist(list(curvatures.values()), bins=20)
    plt.xlabel("Curvature Value")
    plt.ylabel("Frequency")
    plt.title("2-Morphism Curvature Distribution")
    plt.grid(True)
    
    # Plot 3: Inner product magnitudes
    plt.subplot(2, 2, 3)
    plt.hist(inner_products, bins=20)
    plt.xlabel("Inner Product Magnitude")
    plt.ylabel("Frequency")
    plt.title("2-Inner Product Distribution")
    plt.grid(True)
    
    # Plot 4: Inner product phases
    plt.subplot(2, 2, 4)
    plt.hist(phases, bins=20)
    plt.xlabel("Phase (radians)")
    plt.ylabel("Frequency")
    plt.title("Inner Product Phase Distribution")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("category_theory_geometry.png")
    print("Generated plot: category_theory_geometry.png")


def demonstrate_integrated_framework():
    """
    Demonstrate how different quantum gravity approaches can be integrated
    to solve QFT problems.
    """
    print("\n=== Integrated Quantum Gravity Framework ===")
    
    # Create components from different approaches
    qst = QuantumSpacetimeAxioms(dim=4)
    qbh = QuantumBlackHole(mass=50.0)
    holo = ExtendedHolographicDuality()
    ctg = CategoryTheoryGeometry(dim=4, n_points=20)
    
    # ---- Example 1: Cross-framework computation ----
    # Compute spacetime dimension at black hole horizon temperature
    horizon_temp = qbh.temperature
    
    # Use horizon temperature to set the diffusion time for spectral dimension
    diffusion_time = 1.0 / horizon_temp
    bh_spectral_dim = qst.compute_spectral_dimension(diffusion_time)
    
    # ---- Example 2: Integrated QFT calculation ----
    # Calculate effective action with contributions from multiple approaches
    
    # First create an energy scale range
    energy_scales = np.logspace(-2, 2, 20)
    
    # Compute dimensional contribution from QST (dimension flow)
    dimensions = [qst.compute_spectral_dimension(1.0/E) for E in energy_scales]
    
    # Compute holographic contribution (RT formula)
    boundary_nodes = holo.bulk_geometry['boundary_nodes']
    region = boundary_nodes[:len(boundary_nodes)//2]
    rt_result = holo.compute_ryu_takayanagi(region)
    
    # Get minimal surface area
    minimal_area = rt_result['minimal_surface_area']
    
    # Compute black hole contribution (Hawking radiation)
    radiation_spectrum = qbh.compute_radiation_spectrum(omega_range=energy_scales)
    
    # Compute categorical structure contribution (2-morphism phases)
    phases = []
    for tm_id, tm_data in ctg.two_morphisms.items():
        if 'phase' in tm_data['properties']:
            phases.append(tm_data['properties']['phase'])
    
    # Create effective action terms
    effective_action = []
    
    for i, E in enumerate(energy_scales):
        # Term 1: Dimensional regularization from QST
        dim_term = E**(dimensions[i])
        
        # Term 2: Holographic contribution
        holo_term = np.exp(-minimal_area * E)
        
        # Term 3: Hawking radiation contribution
        hawking_term = 1.0 / (np.exp(E / horizon_temp) - 1.0 + 1e-10)
        
        # Term 4: Categorical structure contribution
        cat_term = np.mean([np.sin(p * E) for p in phases]) if phases else 0
        
        # Combine terms (this is a simplified toy model)
        action = dim_term + holo_term + hawking_term + cat_term
        effective_action.append(action)
    
    # Plot the integrated results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Spectral dimension at black hole horizon
    plt.subplot(2, 2, 1)
    plt.plot(energy_scales, dimensions, 'r-')
    plt.axhline(y=bh_spectral_dim, color='b', linestyle='--', 
               label=f"BH Horizon: d={bh_spectral_dim:.2f}")
    plt.axvline(x=horizon_temp, color='g', linestyle='--', 
               label=f"BH Temperature: T={horizon_temp:.2f}")
    plt.xlabel("Energy Scale (Planck units)")
    plt.ylabel("Spectral Dimension")
    plt.title("Spacetime Dimension Flow vs Black Hole Scale")
    plt.grid(True)
    plt.xscale('log')
    plt.legend()
    
    # Plot 2: Combined effective action
    plt.subplot(2, 2, 2)
    plt.loglog(energy_scales, effective_action, 'b-')
    plt.xlabel("Energy Scale (Planck units)")
    plt.ylabel("Effective Action")
    plt.title("Combined Quantum Gravity Effective Action")
    plt.grid(True)
    
    # Plot 3: Comparison of QG approaches for entropy
    plt.subplot(2, 2, 3)
    
    # Create data series
    entropy_energies = np.linspace(10, 100, 10)  # Mass values in Planck units
    
    # Black hole entropy (QuantumBlackHole)
    bh_entropies = []
    for mass in entropy_energies:
        qbh_temp = QuantumBlackHole(mass=mass)
        entropy = qbh_temp.compute_entropy()
        bh_entropies.append(entropy['total_entropy'])
    
    # Holographic entropy (ExtendedHolographicDuality)
    holo_entropies = []
    for i, mass in enumerate(entropy_energies):
        # Use mass to scale the boundary region size
        size = int(len(boundary_nodes) * mass / max(entropy_energies))
        size = max(1, min(size, len(boundary_nodes)))
        region = boundary_nodes[:size]
        rt_result = holo.compute_ryu_takayanagi(region)
        holo_entropies.append(rt_result['holographic_entropy'])
    
    # Plot comparison
    plt.plot(entropy_energies, bh_entropies, 'r-', label="QBH")
    plt.plot(entropy_energies, holo_entropies, 'g-', label="Holographic")
    plt.xlabel("Energy Scale (Planck units)")
    plt.ylabel("Entropy")
    plt.title("Entropy Scaling Across QG Approaches")
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Approach integration diagram (conceptual)
    plt.subplot(2, 2, 4)
    
    # Create a simple diagram showing how approaches connect
    # This uses a simplified network visualization
    G = nx.Graph()
    G.add_node("QST", position=(0, 1))
    G.add_node("BH", position=(1, 1))
    G.add_node("HOLO", position=(1, 0))
    G.add_node("CAT", position=(0, 0))
    G.add_node("QG", position=(0.5, 0.5))
    
    G.add_edge("QST", "QG", weight=2)
    G.add_edge("BH", "QG", weight=2)
    G.add_edge("HOLO", "QG", weight=2)
    G.add_edge("CAT", "QG", weight=2)
    G.add_edge("QST", "BH", weight=1)
    G.add_edge("BH", "HOLO", weight=1)
    G.add_edge("HOLO", "CAT", weight=1)
    G.add_edge("CAT", "QST", weight=1)
    
    pos = nx.get_node_attributes(G, 'position')
    
    # draw the graph
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue',
           font_weight='bold', font_size=10)
    
    # draw edges with varying thickness based on weight
    edge_width = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7)
    
    plt.axis('off')
    plt.title("Integration of Quantum Gravity Approaches")
    
    plt.tight_layout()
    plt.savefig("integrated_framework.png")
    print("Generated plot: integrated_framework.png")


def demonstrate_integrated_qg_phenomenology():
    """
    Demonstrate the integration of numerical simulations with high energy collision predictions.
    
    This example shows how discretized spacetime simulations can be used to inform
    high energy physics predictions where quantum gravity becomes relevant.
    """
    print("\n=== Integrated QG Phenomenology ===")
    
    # Set up a discretized spacetime with a quantum metric perturbation
    print("Setting up discretized spacetime with quantum metric perturbation...")
    space = DiscretizedSpacetime(dim=4, size=6)
    
    # Define a metric perturbation based on a quantum gravity model
    def quantum_metric(coords):
        # Start with Minkowski metric
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Compute radius from origin
        r = np.linalg.norm(coords)
        
        # Skip if at origin to avoid division by zero
        if r < 1e-6:
            return metric
        
        # Quantum gravity perturbation decreases with distance (toy model)
        # In a real model, this would be derived from the QG theory
        perturbation = 0.1 * np.exp(-r**2 / 10.0) / (r + 1e-6)
        
        # Apply anisotropic perturbation
        for i in range(4):
            for j in range(4):
                if i != j:
                    # Off-diagonal terms
                    metric[i,j] += perturbation * coords[i] * coords[j] / (r**2)
        
        # Ensure the metric is symmetric
        metric = (metric + metric.T) / 2.0
        
        return metric
    
    # Set the metric
    space.set_metric(quantum_metric)
    
    # Compute curvature
    curvature = space.compute_discrete_curvature()
    print(f"Spacetime curvature from QG effects: {curvature:.6f}")
    
    # Run Monte Carlo to extract physics
    print("\nRunning Monte Carlo simulation...")
    mc = PathIntegralMonteCarlo(space, action_type='einstein_hilbert')
    mc.num_thermalization = 10
    mc.num_samples = 30
    mc_results = mc.run_simulation()
    
    # Now set up a high energy collision simulator that takes into account the QG results
    print("\nInitializing collision simulator with Monte Carlo results...")
    
    # Use the curvature measure to set the QG scale
    # This is a toy model connection - in reality would be more complex
    qg_scale_modifier = 1.0 / (1.0 + abs(curvature))
    qg_scale = 1e19 * qg_scale_modifier
    
    collision_sim = HighEnergyCollisionSimulator(qg_scale=qg_scale)
    
    # Use the results from our Monte Carlo simulation to modify the QFT parameters
    # This is a toy model of how numerical QG results would feed into particle physics
    qft_qg_integration = collision_sim.qft_qg
    
    # In a real implementation, this would modify QFT parameters based on numerical QG results
    # For demonstration, we just print the values
    print(f"QG-modified Planck scale: {qg_scale:.4e} GeV")
    print(f"Average action from MC: {mc_results['avg_action']:.6f}")
    print(f"Average volume: {mc_results['avg_volume']:.6f}")
    
    # Simulate collisions using our QG-modified model
    print("\nSimulating high energy collisions with QG-modified parameters...")
    collision_results = collision_sim.simulate_collision('higgs', 
                                                     energy_range=(1e3, 14e3, 5),
                                                     n_events=500)
    
    # Calculate the QG effect significance
    sig_estimate = collision_sim.significance_estimate('higgs', luminosity=3000)
    
    # Plot cross section
    fig = collision_sim.plot_cross_sections('higgs', save_path='integrated_higgs_xsec.png')
    
    # Compare standard QFT vs. numerical QG effects
    std_xsec = collision_results['standard_xsec']
    qg_xsec = collision_results['qg_xsec']
    energies = collision_results['energies'] / 1e3  # Convert to TeV
    
    # Create a comparison table
    data = []
    for i, energy in enumerate(energies):
        data.append({
            'Energy (TeV)': f"{energy:.1f}",
            'Standard Cross Section (pb)': f"{std_xsec[i]:.2f}",
            'QG-Modified Cross Section (pb)': f"{qg_xsec[i]:.2f}",
            'Ratio': f"{qg_xsec[i]/std_xsec[i]:.4f}"
        })
    
    comparison_df = pd.DataFrame(data)
    print("\nCross section comparison at different energies:")
    print(comparison_df.to_string(index=False))
    
    # Calculate required luminosity for discovery of QG effects
    print(f"\nRequired luminosity for 5σ discovery: {min(sig_estimate['discovery_luminosity'].values()):.1f} fb⁻¹")
    
    return space, mc, collision_sim


# Example 8: Specific Physics Scenario Simulations
def demonstrate_physics_scenarios():
    """
    Demonstrate simulations of specific physical scenarios that could test QG predictions.
    
    This example shows how to simulate early universe physics and high-energy
    collision scenarios with quantum gravity effects.
    """
    print("\n=== Specific Physics Scenario Simulations ===")
    
    # Early Universe Simulation
    print("\nSimulating early universe with QG effects...")
    cosmos = EarlyUniverseSimulation(qg_scale=1e19, dim=4, size=10)
    
    # Run a simple simulation from near Planck time to end of inflation
    results = cosmos.simulate_evolution(t_start=1e-40, t_end=1e-32, num_points=100)
    
    # Analyze critical events in cosmic history
    events = cosmos.analyze_critical_events()
    print("\nAnalysis of critical events in cosmic history:")
    for event, data in events.items():
        print(f"  {event}: t = {data['time']:.2e} s, T = {data['temperature']:.2e} GeV")
    
    # Plot the evolution of cosmological parameters
    cosmos.plot_evolution(save_path='early_universe_evolution.png')
    
    # Compare with standard cosmology
    cosmos.plot_comparison_with_standard_cosmology(save_path='qg_vs_standard_cosmology.png')
    
    # High-Energy Scenarios
    print("\nSimulating high-energy collisions with QG effects...")
    scenarios = HighEnergyScenarios(qg_scale=1e19, dim=4)
    
    # Simulate specific LHC scenarios
    scenarios.simulate_lhc_scenarios(
        processes=['higgs'],
        energies=[13e3],
        n_events=1000  # Reduced for example
    )
    
    # Analyze threshold effects
    print("\nAnalyzing energy threshold effects for QG visibility...")
    scenarios.analyze_threshold_effects(
        energy_range=(10e3, 100e3),
        n_points=10,  # Reduced for example
        processes=['higgs']
    )
    
    # Plot results
    scenarios.plot_lhc_predictions(save_path='lhc_predictions.png')
    scenarios.plot_threshold_effects(save_path='threshold_effects.png')
    
    # Generate scenario report
    report = scenarios.generate_scenario_report()
    print("\nScenario Report Summary:")
    print("\n".join(report.split("\n")[:10]) + "\n...")
    
    return cosmos, scenarios


# Example 9: Experimental Predictions Interface
def demonstrate_experimental_predictions():
    """
    Demonstrate the interface with established QFT calculations to make
    precise predictions for existing experiments.
    """
    print("\n=== Experimental Predictions Interface ===")
    
    # Initialize the experimental predictions interface
    predictor = ExperimentalPredictions(qg_scale=1e19, dim=4)
    
    # Generate LHC predictions
    print("\nGenerating LHC predictions...")
    higgs_pred = predictor.generate_lhc_predictions('higgs_cross_section', {
        'energy': 13e3,
        'process': 'higgs',
        'n_events': 1000  # Reduced for example
    })
    
    # Calculate significance
    significance = predictor.calculate_statistical_significance('LHC', 'higgs_cross_section', luminosity=3000)
    print(f"Significance: {significance['significance']:.2f}σ at 3000 fb⁻¹")
    
    # Create mock experimental data for comparison
    higgs_data = pd.DataFrame({
        'value': [57.0],
        'uncertainty': [3.0]
    })
    
    # Load experimental data
    predictor.load_experimental_data('LHC', 'higgs_cross_section', higgs_data)
    
    # Compare with data
    comparison = predictor.compare_with_data('LHC', 'higgs_cross_section')
    print("\nComparison with experimental data:")
    print(f"  Data: {comparison['data_value']:.2f} ± {comparison['data_uncertainty']:.2f} pb")
    print(f"  Standard QFT: {comparison['standard_prediction']:.2f} pb (χ² = {comparison['chi2_standard']:.2f})")
    print(f"  QG prediction: {comparison['qg_prediction']:.2f} pb (χ² = {comparison['chi2_qg']:.2f})")
    print(f"  Bayes factor: {comparison['bayes_factor']:.2f}")
    
    # Generate gravitational wave predictions
    print("\nGenerating gravitational wave predictions...")
    gw_pred = predictor.generate_gravitational_wave_predictions('merger_waveform', {
        'mass_ratio': 1.0,
        'total_mass': 65.0,
        'distance': 400.0
    })
    
    # Calculate significance
    gw_sig = predictor.calculate_statistical_significance('GW', 'merger_waveform')
    print(f"GW detection SNR: {gw_sig['snr']:.2f}")
    
    # Generate cosmic ray predictions
    print("\nGenerating cosmic ray predictions...")
    cr_pred = predictor.generate_cosmic_ray_predictions('uhecr_spectrum')
    
    # Generate visualization plots
    predictor.plot_lhc_comparison('higgs_cross_section', save_path='lhc_data_comparison.png')
    predictor.plot_gravitational_wave_comparison('merger_waveform', save_path='gw_prediction.png')
    predictor.plot_cosmic_ray_comparison('uhecr_spectrum', save_path='cr_prediction.png')
    
    # Generate comprehensive report
    report = predictor.generate_comprehensive_report(save_path='qg_predictions_report.txt')
    print("\nPrediction Report Summary:")
    print("\n".join(report.split("\n")[:10]) + "\n...")
    
    return predictor


# Example 10: Advanced Visualization Tools
def demonstrate_visualization_tools():
    """
    Demonstrate the advanced visualization tools for quantum gravity concepts.
    """
    print("\n=== Advanced Visualization Tools ===")
    
    # Quantum Spacetime Visualization
    print("\nCreating quantum spacetime visualizations...")
    
    # Initialize the visualizer
    qst_viz = QuantumSpacetimeVisualizer(style='publication')
    
    # Generate a test causal graph
    causal_graph = nx.DiGraph()
    for i in range(30):
        causal_graph.add_node(i)
    
    # Add causal links (simplified model)
    for i in range(25):
        for j in range(i+1, 30):
            if np.random.random() < 0.1:
                causal_graph.add_edge(i, j)
    
    # Visualize causal structure
    qst_viz.plot_causal_structure(causal_graph, save_path='causal_structure.png')
    
    # Create test data for spectral dimension visualization
    diffusion_times = np.logspace(-3, 3, 100)
    dimensions = 2 + 2 / (1 + (diffusion_times/0.1)**0.5)
    
    # Visualize spectral dimension
    qst_viz.plot_spectral_dimension(dimensions, diffusion_times, 
                                  save_path='spectral_dimension.png')
    
    # Create test data for entanglement entropy visualization
    region_sizes = np.linspace(1, 100, 30)
    entropies = 3 * region_sizes**(2/3) + 0.2 * np.random.random(30)
    
    # Visualize entanglement entropy
    qst_viz.plot_entanglement_entropy(region_sizes, entropies, 
                                    save_path='entanglement_entropy.png')
    
    # Black Hole Visualization
    print("\nCreating quantum black hole visualizations...")
    
    # Initialize the visualizer
    bh_viz = QuantumBlackHoleVisualizer(style='dark')
    
    # Create test data for Hawking radiation
    frequencies = np.linspace(0.1, 3, 100)
    temperature = 0.5
    thermal = frequencies**3 / (np.exp(frequencies / temperature) - 1)
    qg_spectrum = thermal * (1 + 0.2 * np.sin(frequencies * 5))
    
    # Visualize Hawking radiation
    bh_viz.plot_hawking_radiation(frequencies, qg_spectrum, 
                                temperature=temperature,
                                save_path='hawking_radiation.png')
    
    # Create test data for evaporation curve
    times = np.linspace(0, 100, 50)
    m0 = 100.0
    t_evap = 120.0
    std_masses = m0 * (1 - times/t_evap)**(1/3)
    qg_masses = std_masses * (1 + 0.1 * np.exp(-times/10))
    
    # Visualize evaporation curve
    bh_viz.plot_evaporation_curve(times, qg_masses, 
                                save_path='evaporation_curve.png')
    
    # Create test data for interior geometry
    radii = np.linspace(0.1, 10, 50)
    classical = 1.0 / radii**3
    qg_curvature = classical * (1 - np.exp(-radii))
    
    # Visualize interior geometry
    bh_viz.visualize_interior_geometry(radii, qg_curvature, 
                                     save_path='interior_geometry.png')
    
    # Holographic Visualization
    print("\nCreating holographic duality visualizations...")
    
    # Initialize the visualizer
    holo_viz = HolographicVisualizer()
    
    # Create test data for bulk-boundary mapping
    n_bulk = 200
    n_boundary = 50
    
    # Generate bulk points
    bulk_r = np.random.uniform(0, 0.9, n_bulk)
    bulk_theta = np.random.uniform(0, 2*np.pi, n_bulk)
    bulk_phi = np.random.uniform(0, np.pi, n_bulk)
    bulk_x = bulk_r * np.sin(bulk_phi) * np.cos(bulk_theta)
    bulk_y = bulk_r * np.sin(bulk_phi) * np.sin(bulk_theta)
    bulk_z = bulk_r * np.cos(bulk_phi)
    bulk_points = np.column_stack([bulk_x, bulk_y, bulk_z])
    
    # Generate boundary points
    boundary_theta = np.linspace(0, 2*np.pi, n_boundary)
    boundary_phi = np.linspace(0, np.pi, n_boundary)
    boundary_theta, boundary_phi = np.meshgrid(boundary_theta, boundary_phi)
    boundary_theta = boundary_theta.flatten()[:n_boundary]
    boundary_phi = boundary_phi.flatten()[:n_boundary]
    boundary_x = np.sin(boundary_phi) * np.cos(boundary_theta)
    boundary_y = np.sin(boundary_phi) * np.sin(boundary_theta)
    boundary_z = np.cos(boundary_phi)
    boundary_points = np.column_stack([boundary_x, boundary_y, boundary_z])
    
    # Visualize bulk-boundary mapping
    holo_viz.visualize_bulk_boundary_mapping(bulk_points, boundary_points, 
                                          save_path='bulk_boundary.png')
    
    # Create test data for entanglement wedge
    # Select a region on the boundary
    region_indices = np.random.choice(n_boundary, size=n_boundary//4, replace=False)
    boundary_region = boundary_points[region_indices]
    
    # Generate wedge points (simplified model)
    wedge_indices = np.random.choice(n_bulk, size=n_bulk//3, replace=False)
    wedge_points = bulk_points[wedge_indices]
    
    # Generate minimal surface (simplified model)
    n_minimal = 30
    min_theta = np.linspace(0, 2*np.pi, n_minimal)
    min_r = 0.5 + 0.1 * np.cos(3 * min_theta)
    min_x = min_r * np.cos(min_theta)
    min_y = min_r * np.sin(min_theta)
    min_z = np.random.uniform(-0.2, 0.2, n_minimal)
    minimal_surface = np.column_stack([min_x, min_y, min_z])
    
    # Visualize entanglement wedge
    holo_viz.plot_entanglement_wedge(boundary_region, wedge_points, 
                                   minimal_surface,
                                   save_path='entanglement_wedge.png')
    
    print("\nVisualization plots generated successfully!")
    return qst_viz, bh_viz, holo_viz


# Example 11: Theory Benchmarking
def demonstrate_theory_benchmarks():
    """
    Demonstrate benchmarking against other quantum gravity approaches.
    """
    print("\n=== Theory Benchmarking ===")
    
    # Initialize the benchmarking tools
    benchmarks = TheoryBenchmarks(dim=4, qg_scale=1e19)
    
    # Run dimensional flow benchmark
    print("\nBenchmarking dimensional flow across theories...")
    dim_results = benchmarks.benchmark_dimensional_flow()
    
    # Show some example results
    theories = list(benchmarks.theories.keys())
    print("\nAsymptotic spectral dimensions:")
    for theory in theories:
        if theory in dim_results:
            uv_dim = dim_results[theory][-1]
            ir_dim = dim_results[theory][0]
            print(f"  {benchmarks.theories[theory]['name']}: UV={uv_dim:.2f}, IR={ir_dim:.2f}")
    
    # Run black hole entropy benchmark
    print("\nBenchmarking black hole entropy across theories...")
    bh_results = benchmarks.benchmark_black_hole_entropy()
    
    # Run Lorentz violation benchmark
    print("\nBenchmarking Lorentz violation across theories...")
    lv_results = benchmarks.benchmark_lorentz_violation()
    
    # Run qualitative benchmark
    print("\nBenchmarking qualitative aspects across theories...")
    qual_df = benchmarks.benchmark_qualitative_aspects()
    
    # Run prediction comparison
    print("\nBenchmarking specific predictions across theories...")
    pred_df = benchmarks.benchmark_prediction_comparison()
    
    # Create visualization plots
    benchmarks.plot_dimensional_flow_comparison(save_path='dimension_comparison.png')
    benchmarks.plot_black_hole_entropy_comparison(save_path='entropy_comparison.png')
    benchmarks.plot_lorentz_violation_comparison(save_path='lorentz_comparison.png')
    benchmarks.plot_qualitative_comparison(save_path='qualitative_comparison.png')
    
    # Generate comprehensive report
    report = benchmarks.generate_comprehensive_report(save_path='theory_comparison_report.txt')
    print("\nTheory Comparison Report Summary:")
    print("\n".join(report.split("\n")[:10]) + "\n...")
    
    return benchmarks


def run_examples():
    """Run all the example demonstrations."""
    print("Running Quantum Gravity Framework Examples...")
    
    # Original examples
    demonstrate_uv_regularization()
    demonstrate_black_hole_information()
    demonstrate_holographic_principle()
    demonstrate_quantum_reference_frames()
    demonstrate_category_theory_geometry()
    demonstrate_integrated_framework()
    demonstrate_integrated_qg_phenomenology()
    
    # New examples
    demonstrate_physics_scenarios()
    demonstrate_experimental_predictions()
    demonstrate_visualization_tools()
    demonstrate_theory_benchmarks()
    
    print("\nAll examples completed. Generated plot images demonstrate the framework capabilities.")


if __name__ == "__main__":
    # Import additional modules for the new examples
    from quantum_gravity_framework.simulation_scenarios import EarlyUniverseSimulation, HighEnergyScenarios
    from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions
    from quantum_gravity_framework.visualization_tools import QuantumSpacetimeVisualizer, QuantumBlackHoleVisualizer, HolographicVisualizer
    from quantum_gravity_framework.theory_benchmarks import TheoryBenchmarks
    
    run_examples() 