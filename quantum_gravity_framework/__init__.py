"""
Quantum Gravity Framework

A comprehensive framework for quantum gravity research with dimensional flow.
This framework integrates path integral techniques, coupling unification,
non-perturbative methods, and cosmological applications.
"""

# Import core classes from modules
from .path_integral import PathIntegral
from .unified_coupling import UnifiedCouplingFramework
from .non_perturbative import (
    LatticeDynamicTriangulation, 
    SpinFoamModel,
    AsymptoticallyFreeMethods
)
from .cosmology import QuantumCosmology
from .black_hole_microstates import BlackHoleMicrostates
from .unified_framework import UnifiedFramework

__all__ = [
    'PathIntegral',
    'UnifiedCouplingFramework',
    'LatticeDynamicTriangulation',
    'SpinFoamModel',
    'AsymptoticallyFreeMethods',
    'QuantumCosmology',
    'BlackHoleMicrostates',
    'UnifiedFramework'
]

# Version information
__version__ = '0.1.0'
__author__ = 'Quantum Gravity Research Team' 