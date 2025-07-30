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
axes[0, 2].set_title('Gauge Unification')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Beta functions
beta_strong = np.gradient(couplings['strong'], np.log(scales))
beta_em = np.gradient(couplings['electromagnetic'], np.log(scales))
beta_weak = np.gradient(couplings['weak'], np.log(scales))

axes[1, 0].semilogx(scales, beta_strong, 'r-', linewidth=2, label='Strong')
axes[1, 0].semilogx(scales, beta_em, 'b-', linewidth=2, label='EM')
axes[1, 0].semilogx(scales, beta_weak, 'g-', linewidth=2, label='Weak')
axes[1, 0].set_xlabel('Energy Scale (GeV)')
axes[1, 0].set_ylabel('Beta Function')
axes[1, 0].set_title('Beta Functions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Dimension vs coupling correlation
axes[1, 1].scatter(dimensions, couplings['strong'], c=scales, cmap='viridis', s=50)
axes[1, 1].set_xlabel('Spectral Dimension')
axes[1, 1].set_ylabel('Strong Coupling')
axes[1, 1].set_title('Dimension-Coupling Correlation')
axes[1, 1].grid(True, alpha=0.3)

# Transition region zoom
transition_mask = (scales > 0.1) & (scales < 10)
axes[1, 2].plot(scales[transition_mask], dimensions[transition_mask], 'k-', linewidth=2)
axes[1, 2].set_xlabel('Energy Scale (GeV)')
axes[1, 2].set_ylabel('Spectral Dimension')
axes[1, 2].set_title('Transition Region')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rg_flow_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key results
unification_scale = rg.compute_unification_scale()
print(f"Gauge unification scale: {unification_scale:.2e} GeV")
print(f"UV dimension: {dimensions[-1]:.2f}")
print(f"IR dimension: {dimensions[0]:.2f}")
```

### **Example 3: Experimental Predictions**

**Goal**: Generate and analyze experimental predictions for quantum gravity effects.

```python
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions

# Initialize experimental predictions
exp_pred = ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# LHC predictions
lhc_energy = 13.6e3  # GeV
lhc_results = exp_pred.lhc_predictions(energy=lhc_energy)

print("=== LHC Predictions ===")
print(f"Energy: {lhc_energy/1000:.1f} TeV")
print(f"Significance: {lhc_results['significance']:.3f}σ")
print(f"Cross section enhancement: {lhc_results['cross_section_enhancement']:.2e}")
print(f"Higgs pT corrections: {lhc_results['higgs_pt_corrections']:.2e}")

# Gravitational wave predictions
gw_frequency = 100  # Hz
gw_results = exp_pred.gravitational_wave_predictions(frequency=gw_frequency)

print("\n=== Gravitational Wave Predictions ===")
print(f"Frequency: {gw_frequency} Hz")
print(f"Strain modification: {gw_results['strain_modification']:.2e}")
print(f"Phase shift: {gw_results['phase_shift']:.2e}")

# Create comprehensive experimental analysis
experiments = {
    'LHC': {'energy': 13.6e3, 'precision': 1e-18},
    'HL-LHC': {'energy': 14.0e3, 'precision': 1e-19},
    'FCC': {'energy': 100e3, 'precision': 1e-20},
    'LIGO': {'frequency': 100, 'precision': 1e-22},
    'ET': {'frequency': 100, 'precision': 1e-23}
}

results_summary = {}
for exp_name, params in experiments.items():
    if 'energy' in params:
        # Collider experiment
        results = exp_pred.lhc_predictions(energy=params['energy'])
        results['type'] = 'collider'
        results['energy'] = params['energy']
    else:
        # Gravitational wave experiment
        results = exp_pred.gravitational_wave_predictions(frequency=params['frequency'])
        results['type'] = 'gravitational_wave'
        results['frequency'] = params['frequency']
    
    results['precision'] = params['precision']
    results['detectable'] = results['significance'] > 5.0
    results_summary[exp_name] = results

# Create experimental comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Significance comparison
collider_exps = [exp for exp, results in results_summary.items() 
                 if results['type'] == 'collider']
collider_sig = [results_summary[exp]['significance'] for exp in collider_exps]

axes[0, 0].bar(collider_exps, collider_sig, color=['red', 'orange', 'yellow'])
axes[0, 0].set_ylabel('Significance (σ)')
axes[0, 0].set_title('Collider Experiment Significance')
axes[0, 0].tick_params(axis='x', rotation=45)

# Precision vs effect size
all_exps = list(results_summary.keys())
precisions = [results_summary[exp]['precision'] for exp in all_exps]
effects = [results_summary[exp]['significance'] for exp in all_exps]

axes[0, 1].loglog(precisions, effects, 'bo', markersize=8)
axes[0, 1].set_xlabel('Experimental Precision')
axes[0, 1].set_ylabel('QG Effect Size')
axes[0, 1].set_title('Precision vs Effect Size')
axes[0, 1].grid(True, alpha=0.3)

# Add experiment labels
for i, exp in enumerate(all_exps):
    axes[0, 1].annotate(exp, (precisions[i], effects[i]), 
                         xytext=(5, 5), textcoords='offset points')

# Detectability analysis
detectable = [results_summary[exp]['detectable'] for exp in all_exps]
colors = ['green' if d else 'red' for d in detectable]

axes[1, 0].bar(all_exps, [1 if d else 0 for d in detectable], color=colors)
axes[1, 0].set_ylabel('Detectable (1=Yes, 0=No)')
axes[1, 0].set_title('Detection Prospects')
axes[1, 0].tick_params(axis='x', rotation=45)

# Energy scale comparison
energies = []
for exp in all_exps:
    if results_summary[exp]['type'] == 'collider':
        energies.append(results_summary[exp]['energy'] / 1e3)  # Convert to TeV
    else:
        energies.append(results_summary[exp]['frequency'])

axes[1, 1].loglog(energies, effects, 'go', markersize=8)
axes[1, 1].set_xlabel('Energy Scale (TeV) / Frequency (Hz)')
axes[1, 1].set_ylabel('QG Effect Size')
axes[1, 1].set_title('Energy Scale vs Effect Size')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experimental_predictions_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n=== Experimental Summary ===")
for exp_name, results in results_summary.items():
    status = "✅ DETECTABLE" if results['detectable'] else "❌ NOT DETECTABLE"
    print(f"{exp_name}: {status} ({results['significance']:.3f}σ)")
```

## 🔬 Advanced Examples

### **Example 4: Quantum Optics QG Detection**

**Goal**: Analyze quantum optics experiments for quantum gravity detection.

```python
from quantum_gravity_framework.quantum_optics_qg import QuantumOpticsQG

# Initialize quantum optics QG analysis
qo_qg = QuantumOpticsQG()

# Single photon interference
interference_results = qo_qg.single_photon_interference()
print("=== Single Photon Interference ===")
print(f"QG phase shift: {interference_results['qg_phase_shift']:.2e} radians")
print(f"Current precision: {interference_results['current_precision']:.2e} radians")
print(f"Detectable: {interference_results['detectable']}")
print(f"Improvement needed: {interference_results['improvement_needed']:.2e}x")

# Quantum state evolution
evolution_results = qo_qg.quantum_state_evolution()
print("\n=== Quantum State Evolution ===")
print(f"QG decoherence: {evolution_results['qg_decoherence']:.2e}")
print(f"Current precision: {evolution_results['current_precision']:.2e}")
print(f"Detectable: {evolution_results['detectable']}")

# Precision phase measurements
phase_results = qo_qg.precision_phase_measurements()
print("\n=== Precision Phase Measurements ===")
print(f"QG phase shift: {phase_results['qg_phase_shift']:.2e} radians")
print(f"Current precision: {phase_results['current_precision']:.2e} radians")
print(f"Detectable: {phase_results['detectable']}")

# Quantum entanglement with gravity
entanglement_results = qo_qg.quantum_entanglement_gravity()
print("\n=== Quantum Entanglement with Gravity ===")
print(f"QG entanglement measure: {entanglement_results['qg_entanglement']:.2e}")
print(f"Current precision: {entanglement_results['current_precision']:.2e}")
print(f"Detectable: {entanglement_results['detectable']}")

# Create comprehensive analysis
all_experiments = {
    'Single Photon': interference_results,
    'State Evolution': evolution_results,
    'Phase Measurements': phase_results,
    'Entanglement': entanglement_results
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

experiment_names = list(all_experiments.keys())
qg_effects = [all_experiments[exp]['qg_phase_shift'] for exp in experiment_names]
current_precisions = [all_experiments[exp]['current_precision'] for exp in experiment_names]
detectable = [all_experiments[exp]['detectable'] for exp in experiment_names]

# Effect size comparison
axes[0, 0].loglog(experiment_names, qg_effects, 'bo', markersize=10)
axes[0, 0].set_ylabel('QG Effect Size')
axes[0, 0].set_title('Quantum Optics QG Effects')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Precision comparison
axes[0, 1].loglog(experiment_names, current_precisions, 'ro', markersize=10)
axes[0, 1].set_ylabel('Current Precision')
axes[0, 1].set_title('Experimental Precision')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Detectability
colors = ['green' if d else 'red' for d in detectable]
axes[1, 0].bar(experiment_names, [1 if d else 0 for d in detectable], color=colors)
axes[1, 0].set_ylabel('Detectable (1=Yes, 0=No)')
axes[1, 0].set_title('Detection Prospects')
axes[1, 0].tick_params(axis='x', rotation=45)

# Effect vs precision
axes[1, 1].loglog(qg_effects, current_precisions, 'go', markersize=10)
axes[1, 1].set_xlabel('QG Effect Size')
axes[1, 1].set_ylabel('Current Precision')
axes[1, 1].set_title('Effect Size vs Precision')
axes[1, 1].grid(True, alpha=0.3)

# Add experiment labels
for i, exp in enumerate(experiment_names):
    axes[1, 1].annotate(exp, (qg_effects[i], current_precisions[i]), 
                         xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('quantum_optics_qg_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print("\n=== Quantum Optics QG Detection Summary ===")
detectable_count = sum(detectable)
total_count = len(detectable)
print(f"Detectable experiments: {detectable_count}/{total_count}")
print(f"Success rate: {detectable_count/total_count*100:.1f}%")
```

### **Example 5: Multi-Force Correlation Analysis**

**Goal**: Analyze combined effects of multiple forces for QG detection.

```python
from quantum_gravity_framework.multi_force_correlation import MultiForceCorrelationQG

# Initialize multi-force correlation analysis
mfc_qg = MultiForceCorrelationQG()

# Combined force effects
combined_results = mfc_qg.combined_force_effects()
print("=== Combined Force Effects ===")
print(f"Combined QG effect: {combined_results['combined_effect']:.2e}")
print(f"Strong force contribution: {combined_results['strong_contribution']:.2e}")
print(f"EM force contribution: {combined_results['em_contribution']:.2e}")
print(f"Weak force contribution: {combined_results['weak_contribution']:.2e}")

# Cross-correlation experiments
correlation_results = mfc_qg.cross_correlation_experiments()
print("\n=== Cross-Correlation Experiments ===")
print(f"Cross-correlation enhancement: {correlation_results['enhancement_factor']:.2e}")
print(f"Statistical significance: {correlation_results['statistical_significance']:.3f}σ")
print(f"Detectable: {correlation_results['detectable']}")

# Unified force detection
unified_results = mfc_qg.unified_force_detection()
print("\n=== Unified Force Detection ===")
print(f"Unified effect size: {unified_results['unified_effect']:.2e}")
print(f"Detection threshold: {unified_results['detection_threshold']:.2e}")
print(f"Detectable: {unified_results['detectable']}")

# Multi-observable analysis
multi_results = mfc_qg.multi_observable_analysis()
print("\n=== Multi-Observable Analysis ===")
print(f"Multi-observable effect: {multi_results['multi_observable_effect']:.2e}")
print(f"Observable correlation: {multi_results['observable_correlation']:.3f}")
print(f"Combined significance: {multi_results['combined_significance']:.3f}σ")

# Create comprehensive analysis
all_analyses = {
    'Combined Forces': combined_results,
    'Cross-Correlation': correlation_results,
    'Unified Detection': unified_results,
    'Multi-Observable': multi_results
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

analysis_names = list(all_analyses.keys())
effect_sizes = []
detectable_flags = []

for analysis_name in analysis_names:
    results = all_analyses[analysis_name]
    if 'combined_effect' in results:
        effect_sizes.append(results['combined_effect'])
    elif 'unified_effect' in results:
        effect_sizes.append(results['unified_effect'])
    elif 'multi_observable_effect' in results:
        effect_sizes.append(results['multi_observable_effect'])
    else:
        effect_sizes.append(results['enhancement_factor'])
    
    detectable_flags.append(results['detectable'])

# Effect size comparison
axes[0, 0].loglog(analysis_names, effect_sizes, 'bo', markersize=10)
axes[0, 0].set_ylabel('Effect Size')
axes[0, 0].set_title('Multi-Force QG Effects')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Detectability
colors = ['green' if d else 'red' for d in detectable_flags]
axes[0, 1].bar(analysis_names, [1 if d else 0 for d in detectable_flags], color=colors)
axes[0, 1].set_ylabel('Detectable (1=Yes, 0=No)')
axes[0, 1].set_title('Detection Prospects')
axes[0, 1].tick_params(axis='x', rotation=45)

# Statistical significance
significances = []
for analysis_name in analysis_names:
    results = all_analyses[analysis_name]
    if 'statistical_significance' in results:
        significances.append(results['statistical_significance'])
    elif 'combined_significance' in results:
        significances.append(results['combined_significance'])
    else:
        significances.append(0.0)

axes[1, 0].bar(analysis_names, significances, color='orange')
axes[1, 0].set_ylabel('Significance (σ)')
axes[1, 0].set_title('Statistical Significance')
axes[1, 0].tick_params(axis='x', rotation=45)

# Correlation analysis
correlations = []
for analysis_name in analysis_names:
    results = all_analyses[analysis_name]
    if 'observable_correlation' in results:
        correlations.append(results['observable_correlation'])
    else:
        correlations.append(0.0)

axes[1, 1].bar(analysis_names, correlations, color='purple')
axes[1, 1].set_ylabel('Correlation Coefficient')
axes[1, 1].set_title('Observable Correlations')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('multi_force_correlation_tutorial.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print("\n=== Multi-Force Correlation Summary ===")
detectable_count = sum(detectable_flags)
total_count = len(detectable_flags)
print(f"Detectable analyses: {detectable_count}/{total_count}")
print(f"Success rate: {detectable_count/total_count*100:.1f}%")
print(f"Average effect size: {np.mean(effect_sizes):.2e}")
print(f"Average significance: {np.mean(significances):.3f}σ")
```

## 🔗 Next Steps

1. **Explore the [API Reference](../reference/API_REFERENCE.md)** for complete function documentation
2. **Read the [Research Background](../research/RESEARCH_BACKGROUND.md)** for theoretical context
3. **Check the [Experimental Results](../results/)** for detection findings
4. **Review the [Implementation Summary](../research/HONEST_QG_DETECTION_IMPLEMENTATION_SUMMARY.md)** for research outcomes

## 📚 Additional Resources

- **Framework Code**: [`quantum_gravity_framework/`](../../quantum_gravity_framework/)
- **QFT Components**: [`qft/`](../../qft/)
- **Requirements**: [`requirements.txt`](../../requirements.txt)
- **Main Repository**: [README.md](../../README.md)

---

*For technical details, see the [API Reference](../reference/API_REFERENCE.md)*
*For research context, see the [Research Background](../research/RESEARCH_BACKGROUND.md)* 