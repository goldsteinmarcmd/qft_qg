# Quantum Gravity Framework - Tutorial Examples

## 🔬 Common Use Cases and Detailed Examples

This document provides detailed tutorials and examples for common use cases with the Quantum Gravity Framework.

## 📚 Getting Started Examples

### **Example 1: Basic Spectral Dimension Calculation**

**Goal**: Calculate and visualize spectral dimension as a function of energy.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms

# Initialize the framework
qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10)

# Create energy range
energies = np.logspace(-3, 3, 100)

# Calculate spectral dimensions
dimensions = []
for energy in energies:
    dimension = qst.compute_spectral_dimension(energy)
    dimensions.append(dimension)

# Create visualization
plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 2, 1)
plt.loglog(energies, dimensions, 'b-', linewidth=2)
plt.xlabel('Energy Scale (GeV)')
plt.ylabel('Spectral Dimension')
plt.title('Spectral Dimension vs Energy Scale')
plt.grid(True, alpha=0.3)

# Linear scale for transition region
plt.subplot(2, 2, 2)
transition_mask = (energies > 0.1) & (energies < 10)
plt.plot(energies[transition_mask], dimensions[transition_mask], 'r-', linewidth=2)
plt.xlabel('Energy Scale (GeV)')
plt.ylabel('Spectral Dimension')
plt.title('Transition Region (Linear Scale)')
plt.grid(True, alpha=0.3)

# Dimension difference from classical
plt.subplot(2, 2, 3)
dimension_diff = np.array(dimensions) - 4.0
plt.semilogx(energies, dimension_diff, 'g-', linewidth=2)
plt.xlabel('Energy Scale (GeV)')
plt.ylabel('Dimension Difference')
plt.title('Deviation from Classical 4D')
plt.grid(True, alpha=0.3)

# Zoom on quantum regime
plt.subplot(2, 2, 4)
quantum_mask = energies > 1.0
plt.loglog(energies[quantum_mask], dimensions[quantum_mask], 'm-', linewidth=2)
plt.xlabel('Energy Scale (GeV)')
plt.ylabel('Spectral Dimension')
plt.title('Quantum Gravity Regime')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_dimension_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Spectral dimension at low energy: {dimensions[0]:.3f}")
print(f"Spectral dimension at high energy: {dimensions[-1]:.3f}")
print(f"Transition energy: ~1 GeV")
```

### **Example 2: Renormalization Group Flow Analysis**

**Goal**: Analyze how coupling constants evolve with energy scale.

```python
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG

# Initialize RG framework
rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Compute RG flow
results = rg.compute_rg_flow(scale_range=(1e-6, 1e3), num_points=100)

# Extract results
scales = results['scales']
couplings = results['couplings']
dimensions = results['dimensions']

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Coupling evolution
axes[0, 0].loglog(scales, couplings['strong'], 'r-', linewidth=2, label='Strong')
axes[0, 0].loglog(scales, couplings['electromagnetic'], 'b-', linewidth=2, label='EM')
axes[0, 0].loglog(scales, couplings['weak'], 'g-', linewidth=2, label='Weak')
axes[0, 0].set_xlabel('Energy Scale (GeV)')
axes[0, 0].set_ylabel('Coupling Strength')
axes[0, 0].set_title('Coupling Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Dimension evolution
axes[0, 1].loglog(scales, dimensions, 'k-', linewidth=2)
axes[0, 1].set_xlabel('Energy Scale (GeV)')
axes[0, 1].set_ylabel('Spectral Dimension')
axes[0, 1].set_title('Dimension Evolution')
axes[0, 1].grid(True, alpha=0.3)

# Unification plot
axes[0, 2].loglog(scales, couplings['strong'], 'r-', linewidth=2, label='Strong')
axes[0, 2].loglog(scales, couplings['electromagnetic'], 'b-', linewidth=2, label='EM')
axes[0, 2].loglog(scales, couplings['weak'], 'g-', linewidth=2, label='Weak')
axes[0, 2].set_xlabel('Energy Scale (GeV)')
axes[0, 2].set_ylabel('Coupling Strength')
axes[0, 2].set_title('Gauge Coupling Unification')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Linear scale for transition
transition_mask = (scales > 0.1) & (scales < 10)
axes[1, 0].plot(scales[transition_mask], dimensions[transition_mask], 'k-', linewidth=2)
axes[1, 0].set_xlabel('Energy Scale (GeV)')
axes[1, 0].set_ylabel('Spectral Dimension')
axes[1, 0].set_title('Dimension Transition (Linear)')
axes[1, 0].grid(True, alpha=0.3)

# Coupling ratios
axes[1, 1].loglog(scales, couplings['strong'] / couplings['electromagnetic'], 'r-', linewidth=2, label='Strong/EM')
axes[1, 1].loglog(scales, couplings['weak'] / couplings['electromagnetic'], 'g-', linewidth=2, label='Weak/EM')
axes[1, 1].set_xlabel('Energy Scale (GeV)')
axes[1, 1].set_ylabel('Coupling Ratio')
axes[1, 1].set_title('Coupling Ratios')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Quantum corrections
quantum_corrections = 1.0 - dimensions / 4.0
axes[1, 2].semilogx(scales, quantum_corrections, 'm-', linewidth=2)
axes[1, 2].set_xlabel('Energy Scale (GeV)')
axes[1, 2].set_ylabel('Quantum Correction')
axes[1, 2].set_title('Quantum Corrections')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rg_flow_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key results
unification_scale = rg.compute_unification_scale()
print(f"Gauge unification scale: {unification_scale:.2e} GeV")
print(f"Low-energy dimension: {dimensions[0]:.3f}")
print(f"High-energy dimension: {dimensions[-1]:.3f}")
```

### **Example 3: Experimental Predictions**

**Goal**: Generate experimental predictions for LHC and future colliders.

```python
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions

# Initialize experimental predictions
exp_pred = ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Calculate Higgs pT corrections for different colliders
colliders = {
    'LHC_Run3': {'energy': 13.6e3, 'luminosity': 300.0},
    'HL_LHC': {'energy': 14.0e3, 'luminosity': 3000.0},
    'FCC': {'energy': 100e3, 'luminosity': 30000.0}
}

results = {}
for name, params in colliders.items():
    corrections = exp_pred.higgs_pt_corrections(
        energy=params['energy'], 
        pT_range=(0, 1000)
    )
    results[name] = corrections

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Higgs pT spectrum for each collider
colors = ['blue', 'red', 'green']
for i, (name, result) in enumerate(results.items()):
    pT_values = result['pT_values']
    corrections = result['corrections']
    significance = result['significance']
    
    axes[0, 0].semilogx(pT_values, corrections, color=colors[i], 
                         linewidth=2, label=f'{name} (σ={significance:.2e})')

axes[0, 0].set_xlabel('Higgs pT (GeV)')
axes[0, 0].set_ylabel('QG Correction')
axes[0, 0].set_title('Higgs pT Corrections')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Significance comparison
collider_names = list(results.keys())
significances = [results[name]['significance'] for name in collider_names]
axes[0, 1].bar(collider_names, significances, color=colors[:len(collider_names)])
axes[0, 1].set_ylabel('Statistical Significance')
axes[0, 1].set_title('Detection Significance')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Energy dependence
energies = np.logspace(1, 5, 100)
energy_significances = []
for energy in energies:
    result = exp_pred.higgs_pt_corrections(energy=energy, pT_range=(0, 1000))
    energy_significances.append(result['significance'])

axes[1, 0].loglog(energies, energy_significances, 'k-', linewidth=2)
axes[1, 0].set_xlabel('Collision Energy (GeV)')
axes[1, 0].set_ylabel('Statistical Significance')
axes[1, 0].set_title('Energy Dependence')
axes[1, 0].grid(True, alpha=0.3)

# Luminosity requirements
target_significance = 5.0  # 5σ discovery
luminosity_requirements = []
for energy in energies:
    # Calculate required luminosity for 5σ
    result = exp_pred.higgs_pt_corrections(energy=energy, pT_range=(0, 1000))
    required_lumi = target_significance**2 / result['significance']**2
    luminosity_requirements.append(required_lumi)

axes[1, 1].loglog(energies, luminosity_requirements, 'm-', linewidth=2)
axes[1, 1].set_xlabel('Collision Energy (GeV)')
axes[1, 1].set_ylabel('Required Luminosity (fb⁻¹)')
axes[1, 1].set_title('Luminosity Requirements for 5σ')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experimental_predictions_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("Experimental Predictions Summary:")
for name, result in results.items():
    print(f"{name}: σ = {result['significance']:.2e}")
```

## 🔬 Advanced Examples

### **Example 4: Quantum Gravity Detection Analysis**

**Goal**: Comprehensive analysis of QG detection prospects.

```python
from quantum_gravity_framework.comprehensive_qg_detection import ComprehensiveQGDetection

# Run comprehensive analysis
cqg = ComprehensiveQGDetection()
results = cqg.run_all_detection_methods()

# Extract key results
comprehensive = results['comprehensive']
quantum_optics = results['quantum_optics']
precision_em = results['precision_em']
multi_force = results['multi_force']

# Create summary visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Effect size comparison
methods = ['Quantum Optics', 'Precision EM', 'Multi-Force']
max_effects = [
    max([v for k, v in quantum_optics.items() if k != 'summary' and isinstance(v, dict) and 'qg_phase_shift' in v]),
    max([v for k, v in precision_em.items() if k != 'summary' and isinstance(v, dict) and 'qg_frequency_shift' in v]),
    max([v for k, v in multi_force.items() if k != 'summary' and isinstance(v, dict) and 'total_effect' in v])
]

axes[0, 0].bar(methods, max_effects, color=['blue', 'red', 'green'])
axes[0, 0].set_ylabel('Maximum QG Effect')
axes[0, 0].set_title('Effect Size Comparison')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Detection feasibility
detectable_counts = [
    quantum_optics['summary']['detectable_experiments'],
    precision_em['summary']['detectable_experiments'],
    multi_force['summary']['detectable_experiments']
]

total_counts = [
    quantum_optics['summary']['total_experiments'],
    precision_em['summary']['total_experiments'],
    multi_force['summary']['total_experiments']
]

detectable_ratios = [d/t for d, t in zip(detectable_counts, total_counts)]
axes[0, 1].bar(methods, detectable_ratios, color=['blue', 'red', 'green'])
axes[0, 1].set_ylabel('Detection Success Rate')
axes[0, 1].set_title('Detection Feasibility')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Precision requirements
current_precision = 1e-18
precision_improvements = [current_precision / effect for effect in max_effects]
axes[0, 2].bar(methods, precision_improvements, color=['blue', 'red', 'green'])
axes[0, 2].set_ylabel('Required Precision Improvement')
axes[0, 2].set_title('Precision Requirements')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].set_yscale('log')
axes[0, 2].grid(True, alpha=0.3)

# Effect size distribution
all_effects = comprehensive['all_effects']
axes[1, 0].hist(np.log10(all_effects), bins=20, alpha=0.7, color='purple')
axes[1, 0].set_xlabel('log₁₀(Effect Size)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Effect Size Distribution')
axes[1, 0].grid(True, alpha=0.3)

# Method comparison
method_names = comprehensive['all_methods']
effect_sizes = comprehensive['all_effects']
colors = ['blue' if 'quantum_optics' in name else 'red' if 'precision_em' in name else 'green' for name in method_names]

axes[1, 1].scatter(range(len(effect_sizes)), effect_sizes, c=colors, alpha=0.6)
axes[1, 1].set_ylabel('Effect Size')
axes[1, 1].set_title('All Methods Comparison')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

# Summary statistics
stats = ['Max Effect', 'Min Effect', 'Mean Effect', 'Median Effect']
values = [
    comprehensive['max_effect'],
    comprehensive['min_effect'],
    comprehensive['mean_effect'],
    comprehensive['median_effect']
]

axes[1, 2].bar(stats, values, color='orange')
axes[1, 2].set_ylabel('Effect Size')
axes[1, 2].set_title('Summary Statistics')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_detection_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comprehensive summary
print(f"\nComprehensive Detection Analysis Summary:")
print(f"Total methods analyzed: {comprehensive['total_methods']}")
print(f"Potentially detectable: {comprehensive['detectable_methods']}")
print(f"Best method: {comprehensive['best_method']}")
print(f"Best effect size: {comprehensive['best_effect']:.2e}")
print(f"Current precision: {comprehensive['current_precision']:.2e}")
```

### **Example 5: Black Hole Physics**

**Goal**: Analyze black hole entropy and evaporation with quantum gravity effects.

```python
from quantum_gravity_framework.black_hole_microstates import BlackHoleMicrostates

# Initialize black hole analysis
bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Analyze entropy for different masses
masses = np.logspace(0, 3, 100)
entropies_classical = []
entropies_quantum = []

for mass in masses:
    # Classical entropy (Hawking)
    entropy_classical = bh.compute_entropy(mass, use_dimension_flow=False)
    entropies_classical.append(entropy_classical)
    
    # Quantum gravity entropy
    entropy_quantum = bh.compute_entropy(mass, use_dimension_flow=True)
    entropies_quantum.append(entropy_quantum)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Entropy scaling
axes[0, 0].loglog(masses, entropies_classical, 'b-', linewidth=2, label='Classical (Hawking)')
axes[0, 0].loglog(masses, entropies_quantum, 'r-', linewidth=2, label='Quantum Gravity')
axes[0, 0].set_xlabel('Black Hole Mass (M_Pl)')
axes[0, 0].set_ylabel('Entropy')
axes[0, 0].set_title('Black Hole Entropy Scaling')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Entropy ratio
entropy_ratio = np.array(entropies_quantum) / np.array(entropies_classical)
axes[0, 1].semilogx(masses, entropy_ratio, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Black Hole Mass (M_Pl)')
axes[0, 1].set_ylabel('Quantum/Classical Entropy Ratio')
axes[0, 1].set_title('Quantum Corrections to Entropy')
axes[0, 1].grid(True, alpha=0.3)

# Temperature analysis
temperatures_classical = [1.0 / mass for mass in masses]
temperatures_quantum = [1.0 / mass * (1 + 0.1 * np.exp(-mass/10)) for mass in masses]

axes[0, 2].loglog(masses, temperatures_classical, 'b-', linewidth=2, label='Classical')
axes[0, 2].loglog(masses, temperatures_quantum, 'r-', linewidth=2, label='Quantum')
axes[0, 2].set_xlabel('Black Hole Mass (M_Pl)')
axes[0, 2].set_ylabel('Temperature (T_Pl)')
axes[0, 2].set_title('Black Hole Temperature')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Information paradox resolution
remnant_mass = 1.2  # Planck masses
evaporation_times = []
for mass in masses:
    if mass > remnant_mass:
        # Classical evaporation time
        time_classical = mass**3
        # Quantum corrected evaporation time
        time_quantum = time_classical * (1 + 0.05 * np.exp(-mass/5))
        evaporation_times.append(time_quantum)
    else:
        evaporation_times.append(np.inf)

axes[1, 0].loglog(masses, evaporation_times, 'm-', linewidth=2)
axes[1, 0].axvline(x=remnant_mass, color='red', linestyle='--', label='Remnant Mass')
axes[1, 0].set_xlabel('Black Hole Mass (M_Pl)')
axes[1, 0].set_ylabel('Evaporation Time (t_Pl)')
axes[1, 0].set_title('Black Hole Evaporation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Microstate counting
microstate_counts = []
for mass in masses:
    if mass > remnant_mass:
        # Number of microstates
        count = np.exp(entropies_quantum[masses.tolist().index(mass)])
        microstate_counts.append(count)
    else:
        microstate_counts.append(1)

axes[1, 1].loglog(masses, microstate_counts, 'c-', linewidth=2)
axes[1, 1].axvline(x=remnant_mass, color='red', linestyle='--', label='Remnant Mass')
axes[1, 1].set_xlabel('Black Hole Mass (M_Pl)')
axes[1, 1].set_ylabel('Number of Microstates')
axes[1, 1].set_title('Microstate Counting')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Quantum corrections
quantum_corrections = (np.array(entropies_quantum) - np.array(entropies_classical)) / np.array(entropies_classical)
axes[1, 2].semilogx(masses, quantum_corrections, 'k-', linewidth=2)
axes[1, 2].set_xlabel('Black Hole Mass (M_Pl)')
axes[1, 2].set_ylabel('Relative Quantum Correction')
axes[1, 2].set_title('Quantum Corrections')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('black_hole_physics_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key results
print(f"\nBlack Hole Physics Summary:")
print(f"Remnant mass: {remnant_mass} M_Pl")
print(f"Maximum entropy correction: {max(quantum_corrections):.3f}")
print(f"Minimum entropy correction: {min(quantum_corrections):.3f}")
```

## 🎯 Best Practices

### **1. Parameter Selection**
- Start with default parameters
- Adjust based on your specific physics question
- Document parameter choices and rationale

### **2. Error Analysis**
- Always include uncertainty estimates
- Check for numerical stability
- Validate against known limits

### **3. Visualization**
- Use log scales for energy-dependent quantities
- Include error bars where appropriate
- Make plots publication-ready

### **4. Documentation**
- Document your analysis workflow
- Save intermediate results
- Include parameter files with your results

---

*These tutorial examples demonstrate common use cases and best practices for the Quantum Gravity Framework. For detailed API documentation, see the API Reference.* 