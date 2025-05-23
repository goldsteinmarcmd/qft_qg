# Quantum Gravity Framework

A comprehensive framework for quantum gravity research with dimensional flow and variable-dimension spacetime.

## Overview

This framework implements mathematical structures needed for quantum gravity research, focusing on dimensional flow effects across energy scales. It provides tools for computing quantum corrections to standard quantum field theory processes and investigating non-perturbative quantum gravity regimes.

Key components:

- **Path Integral Formulation**: Numerical techniques for computing quantum gravity corrections to standard QFT processes
- **Unified Coupling Framework**: Tools for tracking how fundamental forces unify at high energies with dimensional flow effects
- **Non-Perturbative Methods**: Lattice techniques and other tools for strong coupling regimes
- **Cosmological Applications**: Modules for applying quantum gravity to inflation, dark energy, and early universe physics
- **Black Hole Microstate Accounting**: Tools for analyzing black hole entropy, evaporation, and information paradox resolution with dimensional flow effects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-gravity-framework.git
cd quantum-gravity-framework

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_gravity_framework import PathIntegral, UnifiedCouplingFramework, QuantumCosmology, BlackHoleMicrostates

# Define a dimension profile (how spacetime dimension varies with energy)
# At high energies (E→∞): dimension → 2
# At low energies (E→0): dimension → 4
dim_profile = lambda E: 2.0 + 2.0 / (1.0 + (100.0 * E)**2)

# Create path integral calculator
discreteness_profile = lambda E: 1.0 / (1.0 + (E * 0.1)**(-2))
pi = PathIntegral(dim_profile, discreteness_profile)

# Compute quantum gravity corrections to scalar propagator
corrections = pi.compute_quantum_corrections('scalar_propagator', 
                                          energy_range=(1e-3, 1.0),
                                          num_points=10)

# Visualize corrections
fig = pi.visualize_corrections(corrections)
plt.savefig("quantum_gravity_corrections.png")

# Create unified coupling framework
ucf = UnifiedCouplingFramework(dim_profile)

# Compute coupling unification
unif_results = ucf.compute_unification_scale()

# Analyze cosmological implications
qc = QuantumCosmology(dim_profile)
inflation_results = qc.inflation_quantum_effects()

# Analyze black hole microstate accounting
bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
entropy = bh.compute_entropy(10.0, use_dimension_flow=True)  # Entropy of a 10 Planck mass black hole
```

## Framework Components

### Path Integral Formulation

The `PathIntegral` class provides methods to:
- Calculate quantum corrections to propagators and vertices
- Compute scattering amplitudes with quantum gravity effects
- Monte Carlo evaluation of path integrals in varying dimensions

```python
from quantum_gravity_framework import PathIntegral

# Create path integral calculator
pi = PathIntegral(dim_profile, discreteness_profile)

# Compute scattering amplitude
result = pi.compute_scattering_amplitude('2to2', 0.1, 
                                       particles=['graviton', 'scalar', 'scalar', 'graviton'])
```

### Unified Coupling Framework

The `UnifiedCouplingFramework` implements:
- Running coupling constants across energy scales
- Analysis of force unification with dimensional flow effects
- Computation of critical energy scales for unification

```python
from quantum_gravity_framework import UnifiedCouplingFramework

# Create unified coupling framework
ucf = UnifiedCouplingFramework(dim_profile)

# Analyze how dimensional flow affects unification
flow_results = ucf.compute_dimensional_flow_effects()

# Visualize results
fig = ucf.visualize_coupling_flow(flow_results)
```

### Non-Perturbative Methods

Classes for non-perturbative quantum gravity:
- `LatticeDynamicTriangulation`: Causal dynamical triangulation techniques
- `SpinFoamModel`: Loop quantum gravity inspired calculations
- `AsymptoticallyFreeMethods`: Tools for asymptotically free theories

```python
from quantum_gravity_framework import LatticeDynamicTriangulation, SpinFoamModel

# Create CDT simulation
cdt = LatticeDynamicTriangulation(dim=4, lattice_size=8)
results = cdt.run_simulation()

# Create spin foam model
foam = SpinFoamModel(dim=4, truncation=5)
amp = foam.compute_amplitude()
```

### Cosmological Applications

The `QuantumCosmology` class provides:
- Modified Friedmann equations with dimensional flow
- Inflation with quantum gravity corrections
- Dark energy evolution with running coupling constants

```python
from quantum_gravity_framework import QuantumCosmology

# Create quantum cosmology model
qc = QuantumCosmology(dim_profile)

# Analyze inflation
inflation_results = qc.inflation_quantum_effects(num_efolds=60)

# Examine dark energy equation of state
de_results = qc.dark_energy_equation_of_state()
```

### Black Hole Microstate Accounting

The `BlackHoleMicrostates` class provides:
- Computation of black hole entropy with dimension-dependent corrections
- Microstate counting and analysis of the black hole information paradox
- Modeling of black hole evaporation process with dimensional flow effects
- Analysis of black hole remnant properties

```python
from quantum_gravity_framework import BlackHoleMicrostates

# Create black hole microstate accounting module
bh = BlackHoleMicrostates(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)

# Compute entropy with dimensional flow effects
entropy = bh.compute_entropy(10.0, use_dimension_flow=True)

# Analyze black hole evaporation 
evap_time = bh.compute_evaporation_time(10.0)

# Analyze information paradox resolution
paradox_results = bh.analyze_information_paradox()

# Generate visualizations
bh.plot_microstate_accounting("black_hole_entropy.png")
```

## Developer Notes

This framework is designed for theoretical physics research and numerical experimentation. Many components contain placeholder implementations for theoretical structures that remain active research problems.

## License

MIT License

## Testing and Validation

The framework includes a comprehensive test suite to ensure mathematical consistency and physical validity:

```bash
# Run all tests
./tests/run_all_tests.py

# Run specific test modules
python -m unittest tests.test_qft_qg_integration
python -m unittest tests.test_numerical_validations
```

The test suite verifies:
- Consistency between QFT and QG calculations
- Correct limiting behaviors in different energy regimes
- Numerical implementations against known analytic solutions
- Recovery of standard QFT in the low-energy limit

## Benchmark Examples

Several end-to-end examples demonstrate the framework's capabilities:

```bash
# Black hole formation and evaporation with dimensional flow
python examples/black_hole_evolution.py

# Unification of coupling constants across energy scales
python examples/unified_framework_example.py
```

The examples provide visualizations and detailed output showing how quantum gravity effects modify standard physics at different energy scales. 