# Quantum Gravity Framework - API Reference

## 📚 Complete Function and Class Documentation

This document provides comprehensive API documentation for all components of the Quantum Gravity Framework.

## 🏗️ Core Framework Components

### **QuantumSpacetimeAxioms**

**Purpose**: Implements spectral dimension calculations and quantum spacetime properties.

**Constructor**:
```python
QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10, dimension_profile=None)
```

**Parameters**:
- `dim` (float): Base spacetime dimension (default: 4)
- `planck_length` (float): Planck length in natural units (default: 1.0)
- `spectral_cutoff` (int): Spectral cutoff for calculations (default: 10)
- `dimension_profile` (callable): Custom dimension profile function (optional)

**Key Methods**:

#### `compute_spectral_dimension(diffusion_time)`
Computes the spectral dimension at a given diffusion time.

**Parameters**:
- `diffusion_time` (float): Diffusion time in natural units

**Returns**:
- `float`: Spectral dimension value

**Example**:
```python
qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10)
dimension = qst.compute_spectral_dimension(1.0)
print(f"Spectral dimension: {dimension}")
```

#### `compute_spectral_properties(energy_range, num_points=100)`
Computes spectral properties over an energy range.

**Parameters**:
- `energy_range` (tuple): (min_energy, max_energy) in GeV
- `num_points` (int): Number of calculation points

**Returns**:
- `dict`: Dictionary containing energies, dimensions, and properties

### **DimensionalFlowRG**

**Purpose**: Implements renormalization group flow with dimensional effects.

**Constructor**:
```python
DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0, coupling_constants=None)
```

**Parameters**:
- `dim_uv` (float): Ultraviolet dimension (default: 2.0)
- `dim_ir` (float): Infrared dimension (default: 4.0)
- `transition_scale` (float): Transition scale in GeV (default: 1.0)
- `coupling_constants` (dict): Initial coupling constants (optional)

**Key Methods**:

#### `compute_rg_flow(scale_range, num_points=50)`
Computes renormalization group flow over energy scales.

**Parameters**:
- `scale_range` (tuple): (min_scale, max_scale) in GeV
- `num_points` (int): Number of calculation points

**Returns**:
- `dict`: RG flow results with scales, couplings, and dimensions

**Example**:
```python
rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
results = rg.compute_rg_flow(scale_range=(1e-6, 1e3), num_points=50)
```

#### `compute_unification_scale()`
Computes the gauge unification scale.

**Returns**:
- `float`: Unification scale in GeV

### **ExperimentalPredictions**

**Purpose**: Generates experimental predictions for quantum gravity effects.

**Constructor**:
```python
ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
```

**Key Methods**:

#### `higgs_pt_corrections(energy, pT_range)`
Computes Higgs pT corrections due to quantum gravity.

**Parameters**:
- `energy` (float): Collision energy in GeV
- `pT_range` (tuple): (min_pT, max_pT) in GeV

**Returns**:
- `dict`: Higgs pT correction results

**Example**:
```python
exp_pred = ExperimentalPredictions(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
corrections = exp_pred.higgs_pt_corrections(energy=13.6e3, pT_range=(0, 1000))
```

#### `gauge_coupling_unification()`
Computes gauge coupling unification with quantum gravity effects.

**Returns**:
- `dict`: Unification scale and coupling values

## 🔬 Detection Framework Components

### **QuantumOpticsQG**

**Purpose**: Analyzes quantum optics experiments for QG detection.

**Constructor**:
```python
QuantumOpticsQG()
```

**Key Methods**:

#### `single_photon_interference(wavelength=633e-9, path_length=1.0, gravitational_field=9.81)`
Analyzes single photon interference in curved spacetime.

**Parameters**:
- `wavelength` (float): Photon wavelength in meters
- `path_length` (float): Interferometer path length in meters
- `gravitational_field` (float): Gravitational field strength in m/s²

**Returns**:
- `dict`: Interference analysis results

**Example**:
```python
qo_qg = QuantumOpticsQG()
results = qo_qg.single_photon_interference()
print(f"QG phase shift: {results['qg_phase_shift']:.2e}")
```

#### `quantum_state_evolution(mass=1e-27, superposition_distance=1e-9, evolution_time=1e-3)`
Analyzes quantum state evolution in gravitational field.

**Parameters**:
- `mass` (float): Mass of quantum system in kg
- `superposition_distance` (float): Spatial superposition distance in meters
- `evolution_time` (float): Evolution time in seconds

**Returns**:
- `dict`: Quantum state evolution analysis

#### `run_comprehensive_analysis()`
Runs comprehensive quantum optics analysis.

**Returns**:
- `dict`: Complete analysis results

### **PrecisionElectromagneticQG**

**Purpose**: Analyzes precision electromagnetic measurements for QG detection.

**Constructor**:
```python
PrecisionElectromagneticQG()
```

**Key Methods**:

#### `atomic_clock_frequency_shifts(clock_frequency=9.192631770e9, gravitational_potential=6.26e7, measurement_time=1.0)`
Analyzes atomic clock frequency shifts from QG effects.

**Parameters**:
- `clock_frequency` (float): Atomic clock frequency in Hz
- `gravitational_potential` (float): Gravitational potential in m²/s²
- `measurement_time` (float): Measurement time in seconds

**Returns**:
- `dict`: Atomic clock frequency analysis

**Example**:
```python
pem_qg = PrecisionElectromagneticQG()
results = pem_qg.atomic_clock_frequency_shifts()
print(f"QG frequency shift: {results['qg_frequency_shift']:.2e} Hz")
```

#### `laser_interferometry_quantum(wavelength=1064e-9, arm_length=4000.0, laser_power=100.0)`
Analyzes quantum-enhanced laser interferometry.

**Parameters**:
- `wavelength` (float): Laser wavelength in meters
- `arm_length` (float): Interferometer arm length in meters
- `laser_power` (float): Laser power in watts

**Returns**:
- `dict`: Laser interferometry analysis

#### `run_comprehensive_analysis()`
Runs comprehensive precision EM analysis.

**Returns**:
- `dict`: Complete analysis results

### **MultiForceCorrelationQG**

**Purpose**: Analyzes multi-force correlations for QG detection.

**Constructor**:
```python
MultiForceCorrelationQG()
```

**Key Methods**:

#### `combined_force_effects(energy_scale=100.0, correlation_factor=1.0)`
Analyzes correlations between different forces.

**Parameters**:
- `energy_scale` (float): Energy scale in GeV
- `correlation_factor` (float): Correlation factor between forces

**Returns**:
- `dict`: Combined force effects analysis

**Example**:
```python
mfc_qg = MultiForceCorrelationQG()
results = mfc_qg.combined_force_effects(energy_scale=100.0)
print(f"Combined effect: {results['total_effect']:.2e}")
```

#### `cross_correlation_experiments(n_measurements=1000, measurement_time=1.0, noise_level=1e-18)`
Analyzes cross-correlation between different observables.

**Parameters**:
- `n_measurements` (int): Number of measurements
- `measurement_time` (float): Measurement time in seconds
- `noise_level` (float): Noise level in arbitrary units

**Returns**:
- `dict`: Cross-correlation analysis

#### `run_comprehensive_analysis()`
Runs comprehensive multi-force correlation analysis.

**Returns**:
- `dict`: Complete analysis results

## 🧪 Experimental Validation Components

### **ExperimentalValidator**

**Purpose**: Validates QG predictions against experimental constraints.

**Constructor**:
```python
ExperimentalValidator()
```

**Key Methods**:

#### `run_comprehensive_validation()`
Runs comprehensive experimental validation.

**Returns**:
- `dict`: Complete validation results

**Example**:
```python
validator = ExperimentalValidator()
results = validator.run_comprehensive_validation()
print(f"LHC significance: {results['lhc_run3']['significance']:.2e}")
```

### **ComprehensiveQGDetection**

**Purpose**: Integrates all detection approaches for comprehensive analysis.

**Constructor**:
```python
ComprehensiveQGDetection()
```

**Key Methods**:

#### `run_all_detection_methods()`
Runs all detection methods and compiles results.

**Returns**:
- `dict`: Comprehensive detection results

**Example**:
```python
cqg = ComprehensiveQGDetection()
results = cqg.run_all_detection_methods()
print(f"Total methods analyzed: {results['comprehensive']['total_methods']}")
```

#### `generate_honest_report()`
Generates honest assessment report.

**Returns**:
- `str`: Honest assessment report

#### `save_results(filename="comprehensive_qg_detection_results.txt")`
Saves comprehensive results to file.

**Parameters**:
- `filename` (str): Output filename

## 🖥️ Utility Components

### **BlackHoleMicrostates**

**Purpose**: Analyzes black hole microstates and entropy.

**Constructor**:
```python
BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
```

**Key Methods**:

#### `compute_entropy(mass, use_dimension_flow=True)`
Computes black hole entropy with quantum gravity effects.

**Parameters**:
- `mass` (float): Black hole mass in Planck units
- `use_dimension_flow` (bool): Whether to include dimensional flow effects

**Returns**:
- `float`: Black hole entropy

**Example**:
```python
bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
entropy = bh.compute_entropy(10.0, use_dimension_flow=True)
print(f"Black hole entropy: {entropy}")
```

### **CategoryTheory**

**Purpose**: Implements category theory foundations for quantum gravity.

**Constructor**:
```python
CategoryTheory(num_objects=25, num_morphisms=158)
```

**Key Methods**:

#### `evaluate_topos_logic(statement)`
Evaluates logical statements in topos logic.

**Parameters**:
- `statement` (dict): Logical statement to evaluate

**Returns**:
- `str`: Evaluation result ('true', 'false', 'superposition', 'entangled')

## 📊 Data Structures

### **Common Return Types**

#### **Analysis Results Dictionary**
```python
{
    'method_name': str,
    'parameters': dict,
    'results': dict,
    'uncertainties': dict,
    'metadata': dict
}
```

#### **Experimental Validation Results**
```python
{
    'facility_name': {
        'energy': float,
        'luminosity': float,
        'significance': float,
        'observables': dict,
        'constraints': dict
    }
}
```

#### **Detection Analysis Results**
```python
{
    'effect_size': float,
    'current_precision': float,
    'detectable': bool,
    'improvement_needed': float,
    'method_details': dict
}
```

## 🔧 Configuration

### **Default Parameters**

The framework uses sensible defaults for most parameters:

```python
# Core framework defaults
DEFAULT_DIM = 4.0
DEFAULT_PLANCK_LENGTH = 1.0
DEFAULT_SPECTRAL_CUTOFF = 10

# RG flow defaults
DEFAULT_DIM_UV = 2.0
DEFAULT_DIM_IR = 4.0
DEFAULT_TRANSITION_SCALE = 1.0

# Experimental defaults
DEFAULT_ENERGY = 13.6e3  # GeV
DEFAULT_LUMINOSITY = 300.0  # fb^-1
DEFAULT_PRECISION = 1e-18
```

### **Parameter Validation**

All parameters are validated for physical consistency:

- Energy scales must be positive
- Dimensions must be between 1 and 4
- Precision values must be positive
- Masses must be positive

## 🚨 Error Handling

### **Common Exceptions**

#### **PhysicalConsistencyError**
Raised when parameters violate physical constraints.

#### **NumericalInstabilityError**
Raised when calculations become numerically unstable.

#### **ConvergenceError**
Raised when iterative methods fail to converge.

### **Error Recovery**

```python
try:
    result = qst.compute_spectral_dimension(energy)
except NumericalInstabilityError:
    # Try with reduced precision
    result = qst.compute_spectral_dimension(energy, precision=1e-6)
except PhysicalConsistencyError as e:
    print(f"Physical constraint violated: {e}")
```

## 📈 Performance Considerations

### **Memory Usage**
- Large calculations can use significant memory
- Use batch processing for large datasets
- Consider using generators for memory-intensive operations

### **Computation Time**
- Spectral dimension calculations: O(n²)
- RG flow computations: O(n³)
- Experimental validation: O(n⁴)
- Use caching for repeated calculations

### **Optimization Tips**
```python
# Use vectorized operations
import numpy as np
energies = np.logspace(-3, 3, 1000)
dimensions = np.array([qst.compute_spectral_dimension(e) for e in energies])

# Cache expensive calculations
from functools import lru_cache
@lru_cache(maxsize=128)
def cached_calculation(energy):
    return qst.compute_spectral_dimension(energy)
```

---

*This API reference provides comprehensive documentation for all components of the Quantum Gravity Framework. For usage examples and tutorials, see the User Guide.* 