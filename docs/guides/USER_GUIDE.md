# Quantum Gravity Framework - User Guide

## 🚀 Getting Started

This guide will help you get started with the Quantum Gravity Framework, whether you're a new researcher or an experienced physicist.

### **Prerequisites**
- Python 3.8 or higher
- Basic understanding of quantum field theory
- Familiarity with category theory (helpful but not required)
- Knowledge of general relativity concepts

### **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/qft_qg_other.git
cd qft_qg_other

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from quantum_gravity_framework import QuantumSpacetimeAxioms; print('Installation successful!')"
```

## 📚 Quick Start Tutorial

### **Step 1: Basic Framework Usage**

```python
# Import core components
from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG

# Initialize the framework
qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10)
rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Calculate spectral dimension
dimension = qst.compute_spectral_dimension(1.0)
print(f"Spectral dimension: {dimension}")
```

### **Step 2: Experimental Predictions**

```python
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions

# Create experimental predictions
exp_pred = ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Calculate Higgs pT corrections
higgs_corrections = exp_pred.higgs_pt_corrections(energy=13.6e3, pT_range=(0, 1000))
print(f"Higgs pT corrections: {higgs_corrections}")
```

### **Step 3: Quantum Gravity Detection**

```python
from quantum_gravity_framework.quantum_optics_qg import QuantumOpticsQG

# Analyze quantum optics experiments
qo_qg = QuantumOpticsQG()
interference_results = qo_qg.single_photon_interference()
print(f"QG phase shift: {interference_results['qg_phase_shift']:.2e}")
```

## 🔬 Common Use Cases

### **Case 1: Analyzing Spectral Dimension**

```python
import numpy as np
import matplotlib.pyplot as plt

# Create energy range
energies = np.logspace(-3, 3, 100)

# Calculate spectral dimensions
dimensions = [qst.compute_spectral_dimension(energy) for energy in energies]

# Plot results
plt.figure(figsize=(10, 6))
plt.loglog(energies, dimensions)
plt.xlabel('Energy Scale')
plt.ylabel('Spectral Dimension')
plt.title('Spectral Dimension vs Energy Scale')
plt.grid(True)
plt.savefig('spectral_dimension.png')
plt.show()
```

### **Case 2: Computing RG Flow**

```python
# Compute renormalization group flow
rg.compute_rg_flow(scale_range=(1e-6, 1e3), num_points=50)

# Access results
scales = rg.scales
couplings = rg.couplings
dimensions = rg.dimensions

# Plot coupling evolution
plt.figure(figsize=(12, 8))
for force, coupling in couplings.items():
    plt.loglog(scales, coupling, label=force)
plt.xlabel('Energy Scale (GeV)')
plt.ylabel('Coupling Strength')
plt.title('Renormalization Group Flow')
plt.legend()
plt.grid(True)
plt.show()
```

### **Case 3: Experimental Validation**

```python
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions

# Create experimental predictions
exp_pred = ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Analyze LHC predictions
lhc_results = exp_pred.lhc_predictions(energy=13.6e3)
print(f"LHC significance: {lhc_results['significance']:.3f}σ")

# Analyze gravitational wave predictions
gw_results = exp_pred.gravitational_wave_predictions(frequency=100)
print(f"GW strain: {gw_results['strain']:.2e}")
```

## 🔧 Advanced Usage

### **Custom Dimension Profiles**

```python
def custom_dimension_profile(energy):
    """Custom dimension profile function."""
    dim_ir = 4.0
    dim_uv = 2.0
    transition_scale = 1.0
    
    # Smooth transition with different functional form
    return dim_ir - (dim_ir - dim_uv) * np.tanh(energy / transition_scale)

# Use custom profile
qst_custom = QuantumSpacetimeAxioms(
    dim=4, 
    planck_length=1.0, 
    spectral_cutoff=10,
    dimension_profile=custom_dimension_profile
)
```

### **Multi-Force Analysis**

```python
from quantum_gravity_framework.multi_force_correlation import MultiForceCorrelationQG

# Analyze combined force effects
mfc_qg = MultiForceCorrelationQG()
combined_results = mfc_qg.combined_force_effects()
print(f"Combined QG effect: {combined_results['combined_effect']:.2e}")
```

### **Precision Measurements**

```python
from quantum_gravity_framework.precision_em_qg import PrecisionElectromagneticQG

# Analyze atomic clock experiments
pem_qg = PrecisionElectromagneticQG()
clock_results = pem_qg.atomic_clock_frequency_shifts()
print(f"Clock frequency shift: {clock_results['frequency_shift']:.2e} Hz")
```

## 🐛 Troubleshooting

### **Common Issues**

#### **Import Errors**
```python
# If you get import errors, check your installation
python -c "import quantum_gravity_framework; print('Import successful')"
```

#### **Numerical Issues**
```python
# For numerical stability, use appropriate energy ranges
# Avoid extremely small or large values
energies = np.logspace(-2, 2, 100)  # Safe range
```

#### **Memory Issues**
```python
# For large calculations, use chunking
def chunked_calculation(energy_range, chunk_size=1000):
    results = []
    for i in range(0, len(energy_range), chunk_size):
        chunk = energy_range[i:i+chunk_size]
        chunk_results = [qst.compute_spectral_dimension(e) for e in chunk]
        results.extend(chunk_results)
    return results
```

### **Performance Optimization**

```python
# Use vectorized operations when possible
energies = np.logspace(-3, 3, 1000)
dimensions = np.vectorize(qst.compute_spectral_dimension)(energies)

# Cache expensive calculations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_spectral_dimension(energy):
    return qst.compute_spectral_dimension(energy)
```

## 📖 Best Practices

### **Code Organization**
```python
# Separate theoretical calculations from visualization
def compute_theory(parameters):
    """Pure theoretical calculations."""
    # ... calculations ...
    return results

def visualize_results(results):
    """Visualization and plotting."""
    # ... plotting ...
    return figure

# Use this pattern
theory_results = compute_theory(my_parameters)
plot = visualize_results(theory_results)
```

### **Documentation**
```python
def calculate_qg_effect(energy, coupling_strength):
    """
    Calculate quantum gravity effect at given energy.
    
    Parameters:
    -----------
    energy : float
        Energy scale in GeV
    coupling_strength : float
        Coupling strength (dimensionless)
    
    Returns:
    --------
    float
        QG effect magnitude
    """
    # ... calculation ...
    return effect
```

### **Testing**
```python
# Include unit tests for your calculations
def test_spectral_dimension():
    qst = QuantumSpacetimeAxioms(dim=4)
    
    # Test low energy limit
    low_energy_dim = qst.compute_spectral_dimension(1e-6)
    assert abs(low_energy_dim - 4.0) < 0.1
    
    # Test high energy limit
    high_energy_dim = qst.compute_spectral_dimension(1e3)
    assert abs(high_energy_dim - 2.0) < 0.1
```

## 🔗 Next Steps

1. **Read the [Tutorial Examples](TUTORIAL_EXAMPLES.md)** for detailed examples
2. **Check the [API Reference](../reference/API_REFERENCE.md)** for complete documentation
3. **Review the [Research Background](../research/RESEARCH_BACKGROUND.md)** for theoretical context
4. **Examine the [Experimental Results](../results/)** for detection methods

## 📚 Additional Resources

- **Framework Code**: [`quantum_gravity_framework/`](../../quantum_gravity_framework/)
- **QFT Components**: [`qft/`](../../qft/)
- **Requirements**: [`requirements.txt`](../../requirements.txt)
- **Main Repository**: [README.md](../../README.md)

---

*For technical details, see the [API Reference](../reference/API_REFERENCE.md)*
*For research context, see the [Research Background](../research/RESEARCH_BACKGROUND.md)* 