# Quantum Field Theory Calculations

This module provides tools for performing quantum field theory calculations, including:

- Path integrals
- Feynman diagrams
- Propagators
- Scattering amplitudes
- Renormalization
- Gauge theories
- Effective field theories
- Non-perturbative methods
- Quantum gravity approaches
- Lattice field theory
- Tensor network methods

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Path integrals and numerical calculations
from qft.path_integral import calculate_path_integral, generate_field_configurations

# Feynman diagrams and perturbation theory
from qft.feynman import FeynmanDiagram, electron_positron_scattering

# Scattering amplitudes and cross sections
from qft.scattering import mandelstam_variables, compton_scattering_amplitude

# Renormalization and running couplings
from qft.renormalization import running_coupling, beta_function

# Gauge theories (QED, QCD)
from qft.gauge_theory import U1GaugeField, SUNGaugeField

# Effective field theories
from qft.effective_field_theory import EffectiveTheory, create_sm_eft

# Non-perturbative methods
from qft.non_perturbative import SchwingerDysonSolver, FunctionalRG, instanton_action

# Quantum gravity
from qft.quantum_gravity import CausalTriangulation, EmergentSpacetime, SpinNetwork

# Lattice field theory
from qft.lattice_field_theory import LatticeScalarField, HybridMonteCarlo

# Tensor networks
from qft.tensor_networks import MatrixProductState, TensorRenormalizationGroup
```

## Features

### Path Integrals

The `path_integral` module provides tools for calculating path integrals in quantum field theory, including generating field configurations and evaluating actions.

### Feynman Diagrams

The `feynman` module implements classes for representing and calculating Feynman diagrams, with support for visualization.

### Scattering Calculations

The `scattering` module provides functions for calculating scattering amplitudes, cross sections, and related quantities.

### Renormalization

The `renormalization` module implements tools for handling divergences, including regularization schemes, renormalization group equations, and running couplings.

### Gauge Theories

The `gauge_theory` module provides implementations of U(1) and SU(N) gauge theories, including field strength tensors, gauge-covariant derivatives, and lattice formulations.

### Effective Field Theories

The `effective_field_theory` module implements tools for constructing and analyzing effective field theories, including operator dimension counting, power counting, and matching.

### Non-Perturbative Methods

The `non_perturbative` module implements techniques beyond perturbation theory, including Schwinger-Dyson equations, instantons, resummation, and functional renormalization group approaches.

### Quantum Gravity

The `quantum_gravity` module provides implementations of various approaches to quantum gravity, including asymptotic safety, causal dynamical triangulations, and emergent spacetime models.

### Lattice Field Theory

The `lattice_field_theory` module implements numerical simulations of quantum field theories on a lattice, including Monte Carlo methods, Hybrid Monte Carlo algorithm, and tools for critical phenomena and continuum limit extrapolation.

### Tensor Networks

The `tensor_networks` module implements tensor network methods for quantum field theory, including Matrix Product States (MPS) and the Tensor Renormalization Group (TRG) algorithm. These techniques provide efficient representations of quantum states and enable calculations in strongly correlated systems.

## Examples

See the `examples.py` file for demonstrations of how to use each module. 