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

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.loglog(scales, couplings)
plt.xlabel('Energy Scale')
plt.ylabel('Coupling Strength')
plt.title('RG Flow - Couplings')

plt.subplot(1, 3, 2)
plt.loglog(scales, dimensions)
plt.xlabel('Energy Scale')
plt.ylabel('Dimension')
plt.title('RG Flow - Dimensions')

plt.subplot(1, 3, 3)
plt.semilogx(scales, dimensions)
plt.xlabel('Energy Scale')
plt.ylabel('Dimension')
plt.title('Dimension Flow (Linear)')

plt.tight_layout()
plt.savefig('rg_flow.png')
plt.show()
```

### **Case 3: Experimental Validation**

```python
from quantum_gravity_framework.experimental_validation import ExperimentalValidator

# Run comprehensive experimental validation
validator = ExperimentalValidator()
results = validator.run_comprehensive_validation()

# Access specific results
lhc_results = results['lhc_run3']
hl_lhc_results = results['hl_lhc']
fcc_results = results['fcc']

print(f"LHC significance: {lhc_results['significance']:.2e}")
print(f"HL-LHC significance: {hl_lhc_results['significance']:.2e}")
print(f"FCC significance: {fcc_results['significance']:.2e}")
```

### **Case 4: Black Hole Analysis**

```python
from quantum_gravity_framework.black_hole_microstates import BlackHoleMicrostates

# Analyze black hole microstates
bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Compute entropy for different masses
masses = np.logspace(0, 3, 50)
entropies = [bh.compute_entropy(mass, use_dimension_flow=True) for mass in masses]

# Plot entropy scaling
plt.figure(figsize=(10, 6))
plt.loglog(masses, entropies)
plt.xlabel('Black Hole Mass (M_Pl)')
plt.ylabel('Entropy')
plt.title('Black Hole Entropy Scaling')
plt.grid(True)
plt.savefig('black_hole_entropy.png')
plt.show()
```

## 🛠️ Advanced Usage

### **Custom Dimension Profiles**

```python
# Define custom dimension profile
def custom_dimension_profile(energy):
    """Custom dimension profile that varies with energy."""
    base_dim = 4.0
    quantum_correction = 2.0 / (1.0 + (energy * 0.1)**2)
    return base_dim - quantum_correction

# Use custom profile
qst_custom = QuantumSpacetimeAxioms(
    dim=4, 
    planck_length=1.0, 
    spectral_cutoff=10,
    dimension_profile=custom_dimension_profile
)
```

### **Multi-Force Correlation Analysis**

```python
from quantum_gravity_framework.multi_force_correlation import MultiForceCorrelationQG

# Analyze multi-force correlations
mfc_qg = MultiForceCorrelationQG()
correlation_results = mfc_qg.combined_force_effects(energy_scale=100.0)

print(f"Strong force effect: {correlation_results['strong_effect']:.2e}")
print(f"EM force effect: {correlation_results['em_effect']:.2e}")
print(f"Combined effect: {correlation_results['total_effect']:.2e}")
```

### **Precision Electromagnetic Measurements**

```python
from quantum_gravity_framework.precision_em_qg import PrecisionElectromagneticQG

# Analyze precision EM measurements
pem_qg = PrecisionElectromagneticQG()
atomic_clock_results = pem_qg.atomic_clock_frequency_shifts()

print(f"QG frequency shift: {atomic_clock_results['qg_frequency_shift']:.2e} Hz")
print(f"Detectable: {atomic_clock_results['detectable']}")
```

## 🔧 Troubleshooting

### **Common Issues and Solutions**

#### **Issue 1: Import Errors**
```python
# Error: ModuleNotFoundError: No module named 'quantum_gravity_framework'
# Solution: Install in development mode
pip install -e .
```

#### **Issue 2: Numerical Instability**
```python
# Error: NaN or infinite values in calculations
# Solution: Check parameter ranges and use smaller energy scales
qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=5)  # Reduced cutoff
```

#### **Issue 3: Memory Issues**
```python
# Error: MemoryError during large calculations
# Solution: Use smaller arrays and batch processing
energies = np.logspace(-3, 3, 50)  # Reduced from 100 to 50 points
```

#### **Issue 4: Convergence Problems**
```python
# Error: RG flow not converging
# Solution: Adjust transition scale and iteration parameters
rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=0.1)  # Smaller transition
```

### **Performance Optimization**

```python
# For large-scale calculations, use vectorized operations
import numpy as np

# Vectorized spectral dimension calculation
energies = np.logspace(-3, 3, 1000)
dimensions = np.array([qst.compute_spectral_dimension(e) for e in energies])

# Use caching for expensive calculations
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_spectral_dimension(energy):
    return qst.compute_spectral_dimension(energy)
```

## 📊 Best Practices

### **1. Parameter Selection**
- Start with default parameters
- Adjust based on your specific physics question
- Document parameter choices and their rationale

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

## 🎯 Next Steps

After mastering the basics:

1. **Explore the examples directory** for more complex use cases
2. **Read the API reference** for detailed function documentation
3. **Check the research background** for theoretical context
4. **Run the comprehensive tests** to verify your installation
5. **Contribute to the framework** by reporting issues or suggesting improvements

## 📞 Getting Help

- **Check the API reference** for detailed function documentation
- **Look at the examples** in the `examples/` directory
- **Run the test suite** to verify your setup
- **Report issues** on the GitHub repository

---

*This user guide provides the essential information to get started with the Quantum Gravity Framework. For advanced usage and detailed API documentation, see the other documentation files.* 