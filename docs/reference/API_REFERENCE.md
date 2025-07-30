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

#### `lhc_predictions(energy=13.6e3)`
Generates LHC predictions for quantum gravity effects.

**Parameters**:
- `energy` (float): Collision energy in GeV (default: 13.6e3)

**Returns**:
- `dict`: Dictionary containing significance, cross section enhancements, and corrections

#### `gravitational_wave_predictions(frequency=100)`
Generates gravitational wave predictions.

**Parameters**:
- `frequency` (float): Gravitational wave frequency in Hz (default: 100)

**Returns**:
- `dict`: Dictionary containing strain modifications and phase shifts

### **QuantumOpticsQG**

**Purpose**: Analyzes quantum optics experiments for quantum gravity detection.

**Constructor**:
```python
QuantumOpticsQG(planck_energy=1.22e19, qg_base_effect=1e-40)
```

**Parameters**:
- `planck_energy` (float): Planck energy in eV (default: 1.22e19)
- `qg_base_effect` (float): Base quantum gravity effect size (default: 1e-40)

**Key Methods**:

#### `single_photon_interference()`
Analyzes single photon interference experiments.

**Returns**:
- `dict`: Dictionary containing QG phase shift, current precision, and detectability

#### `quantum_state_evolution()`
Analyzes quantum state evolution in curved spacetime.

**Returns**:
- `dict`: Dictionary containing QG decoherence effects and detectability

#### `precision_phase_measurements()`
Analyzes precision phase measurement experiments.

**Returns**:
- `dict`: Dictionary containing QG phase shifts and experimental precision

#### `quantum_entanglement_gravity()`
Analyzes quantum entanglement with gravitational effects.

**Returns**:
- `dict`: Dictionary containing QG entanglement measures and detectability

### **PrecisionElectromagneticQG**

**Purpose**: Analyzes precision electromagnetic measurements for quantum gravity detection.

**Constructor**:
```python
PrecisionElectromagneticQG(planck_energy=1.22e19, qg_base_effect=1e-40)
```

**Key Methods**:

#### `atomic_clock_frequency_shifts()`
Analyzes atomic clock frequency shifts due to quantum gravity.

**Returns**:
- `dict`: Dictionary containing frequency shifts and detectability

#### `laser_interferometry_quantum()`
Analyzes quantum-enhanced laser interferometry.

**Returns**:
- `dict`: Dictionary containing length changes and precision

#### `cavity_qed_precision()`
Analyzes cavity QED precision measurements.

**Returns**:
- `dict`: Dictionary containing energy shifts and detectability

#### `quantum_sensor_field_variations()`
Analyzes quantum sensor field variations.

**Returns**:
- `dict`: Dictionary containing field variations and precision

### **MultiForceCorrelationQG**

**Purpose**: Analyzes combined effects of multiple forces for quantum gravity detection.

**Constructor**:
```python
MultiForceCorrelationQG(planck_energy=1.22e19, qg_base_effect=1e-40)
```

**Key Methods**:

#### `combined_force_effects()`
Analyzes combined effects of strong, electromagnetic, and weak forces.

**Returns**:
- `dict`: Dictionary containing combined effects and individual contributions

#### `cross_correlation_experiments()`
Analyzes cross-correlation experiments for enhanced detection.

**Returns**:
- `dict`: Dictionary containing enhancement factors and statistical significance

#### `unified_force_detection()`
Analyzes unified force detection strategies.

**Returns**:
- `dict`: Dictionary containing unified effects and detection thresholds

#### `multi_observable_analysis()`
Analyzes multi-observable correlation analysis.

**Returns**:
- `dict`: Dictionary containing multi-observable effects and correlations

### **ComprehensiveQGDetection**

**Purpose**: Integrates all quantum gravity detection approaches for comprehensive analysis.

**Constructor**:
```python
ComprehensiveQGDetection(planck_energy=1.22e19, qg_base_effect=1e-40)
```

**Key Methods**:

#### `run_comprehensive_analysis()`
Runs comprehensive analysis of all detection methods.

**Returns**:
- `dict`: Dictionary containing results from all detection approaches

#### `generate_honest_report()`
Generates an honest assessment report of detection prospects.

**Returns**:
- `str`: Comprehensive text report of findings

#### `save_results(filename_prefix='comprehensive_qg_detection')`
Saves analysis results to files.

**Parameters**:
- `filename_prefix` (str): Prefix for output files (default: 'comprehensive_qg_detection')

**Returns**:
- `tuple`: (text_file_path, numpy_file_path)

## 🔧 Utility Functions

### **Mathematical Functions**

#### `spectral_dimension_formula(diffusion_time, dim_ir=4.0, dim_uv=2.0, transition_scale=1.0)`
Computes spectral dimension using the standard formula.

**Parameters**:
- `diffusion_time` (float): Diffusion time
- `dim_ir` (float): Infrared dimension (default: 4.0)
- `dim_uv` (float): Ultraviolet dimension (default: 2.0)
- `transition_scale` (float): Transition scale (default: 1.0)

**Returns**:
- `float`: Spectral dimension value

#### `rg_beta_function(coupling, dimension, force_type)`
Computes renormalization group beta function.

**Parameters**:
- `coupling` (float): Coupling strength
- `dimension` (float): Spacetime dimension
- `force_type` (str): Type of force ('strong', 'electromagnetic', 'weak')

**Returns**:
- `float`: Beta function value

### **Experimental Functions**

#### `calculate_qg_effect(energy, coupling_strength, effect_type='phase_shift')`
Calculates quantum gravity effect for given parameters.

**Parameters**:
- `energy` (float): Energy scale in GeV
- `coupling_strength` (float): Coupling strength
- `effect_type` (str): Type of effect (default: 'phase_shift')

**Returns**:
- `float`: Quantum gravity effect magnitude

#### `compare_with_precision(effect_size, current_precision)`
Compares effect size with experimental precision.

**Parameters**:
- `effect_size` (float): Calculated effect size
- `current_precision` (float): Current experimental precision

**Returns**:
- `dict`: Dictionary containing detectability and improvement needed

## 📊 Data Structures

### **Analysis Results**

All analysis methods return dictionaries with consistent structure:

```python
{
    'qg_effect': float,           # Quantum gravity effect size
    'current_precision': float,    # Current experimental precision
    'detectable': bool,           # Whether effect is detectable
    'improvement_needed': float,  # Required precision improvement
    'significance': float,        # Statistical significance (σ)
    'method': str,               # Analysis method name
    'parameters': dict           # Input parameters used
}
```

### **Experimental Parameters**

Standard experimental parameters used throughout the framework:

```python
{
    'planck_energy': 1.22e19,     # Planck energy in eV
    'qg_base_effect': 1e-40,      # Base QG effect size
    'current_precision': {
        'phase_shift': 1e-18,     # Phase shift precision (radians)
        'frequency': 1e-18,       # Frequency precision (Hz)
        'length': 1e-18,          # Length precision (meters)
        'energy': 1e-18,          # Energy precision (eV)
        'field': 1e-15            # Field precision (Tesla)
    }
}
```

## 🔗 Integration Examples

### **Complete Analysis Workflow**

```python
from quantum_gravity_framework.comprehensive_qg_detection import ComprehensiveQGDetection

# Initialize comprehensive analysis
comprehensive = ComprehensiveQGDetection()

# Run complete analysis
results = comprehensive.run_comprehensive_analysis()

# Generate honest report
report = comprehensive.generate_honest_report()
print(report)

# Save results
text_file, numpy_file = comprehensive.save_results()
print(f"Results saved to: {text_file}, {numpy_file}")
```

### **Custom Analysis Pipeline**

```python
from quantum_gravity_framework.quantum_optics_qg import QuantumOpticsQG
from quantum_gravity_framework.precision_em_qg import PrecisionElectromagneticQG
from quantum_gravity_framework.multi_force_correlation import MultiForceCorrelationQG

# Initialize all detection methods
qo_qg = QuantumOpticsQG()
pem_qg = PrecisionElectromagneticQG()
mfc_qg = MultiForceCorrelationQG()

# Run individual analyses
interference_results = qo_qg.single_photon_interference()
clock_results = pem_qg.atomic_clock_frequency_shifts()
correlation_results = mfc_qg.combined_force_effects()

# Compile results
all_results = {
    'quantum_optics': interference_results,
    'precision_em': clock_results,
    'multi_force': correlation_results
}

# Analyze detectability
detectable_methods = [name for name, results in all_results.items() 
                     if results['detectable']]
print(f"Detectable methods: {detectable_methods}")
```

## 📝 Best Practices

### **Parameter Selection**
- Use realistic energy scales (1e-3 to 1e3 GeV for most analyses)
- Choose appropriate precision levels based on current technology
- Consider physical constraints when setting parameters

### **Numerical Stability**
- Avoid extremely small or large parameter values
- Use appropriate numerical precision for calculations
- Check for convergence in iterative methods

### **Documentation**
- Always document parameter choices and their physical motivation
- Include units and dimensional analysis in calculations
- Reference relevant theoretical frameworks

## 🔗 Related Documentation

- **[User Guide](../guides/USER_GUIDE.md)** - Getting started guide
- **[Tutorial Examples](../guides/TUTORIAL_EXAMPLES.md)** - Detailed examples
- **[Research Background](../research/RESEARCH_BACKGROUND.md)** - Theoretical context
- **[Implementation Summary](../research/HONEST_QG_DETECTION_IMPLEMENTATION_SUMMARY.md)** - Research outcomes

---

*For getting started, see the [User Guide](../guides/USER_GUIDE.md)*
*For examples, see the [Tutorial Examples](../guides/TUTORIAL_EXAMPLES.md)* 