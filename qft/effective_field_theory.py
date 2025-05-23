"""Effective field theory module for QFT.

This module implements tools for working with effective field theories (EFTs),
including operator dimension counting, power counting, and matching.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Set, Optional, Union, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Field:
    """Class representing a field in an effective field theory."""
    name: str
    spin: float
    mass_dimension: float
    statistics: str  # 'fermion' or 'boson'
    
    def __post_init__(self):
        # Validate statistics
        if self.statistics not in ['fermion', 'boson']:
            raise ValueError("Statistics must be 'fermion' or 'boson'")
        
        # Default mass dimension for common field types if not specified
        if self.mass_dimension == 0:
            if self.statistics == 'boson':
                if self.spin == 0:  # Scalar
                    self.mass_dimension = 1
                elif self.spin == 1:  # Vector
                    self.mass_dimension = 1
                elif self.spin == 2:  # Graviton
                    self.mass_dimension = 0
            elif self.statistics == 'fermion':
                if self.spin == 0.5:  # Dirac/Weyl fermion
                    self.mass_dimension = 1.5

@dataclass
class Operator:
    """Class representing an operator in an effective field theory."""
    terms: List[List[Field]]  # List of products of fields
    derivatives: int = 0
    coefficient: float = 1.0
    name: str = ""
    
    def dimension(self) -> float:
        """Calculate the mass dimension of the operator."""
        dim = self.derivatives
        
        for term in self.terms:
            for field in term:
                dim += field.mass_dimension
        
        return dim
    
    def __str__(self) -> str:
        if self.name:
            return self.name
        
        # Generate a string representation of the operator
        term_strs = []
        for term in self.terms:
            fields_str = " ".join([field.name for field in term])
            if self.derivatives > 0:
                fields_str = f"∂^{self.derivatives} ({fields_str})"
            term_strs.append(fields_str)
        
        return " + ".join(term_strs)

class EffectiveTheory:
    """Class representing an effective field theory."""
    
    def __init__(self, name: str, fields: List[Field], cutoff_scale: float):
        self.name = name
        self.fields = {field.name: field for field in fields}
        self.cutoff_scale = cutoff_scale  # EFT cutoff scale Λ
        self.operators: Dict[str, Operator] = {}
    
    def add_operator(self, operator: Operator, name: Optional[str] = None) -> str:
        """Add an operator to the theory.
        
        Args:
            operator: The operator to add
            name: Optional name for the operator
            
        Returns:
            The name of the operator
        """
        if name is None:
            name = f"O_{len(self.operators) + 1}"
        
        operator.name = name
        self.operators[name] = operator
        return name
    
    def lagrangian(self, max_dimension: int = 4, include_names: bool = False) -> str:
        """Generate the EFT Lagrangian up to a maximum operator dimension.
        
        Args:
            max_dimension: Maximum operator dimension to include
            include_names: Whether to include operator names in output
            
        Returns:
            String representation of the Lagrangian
        """
        terms = []
        
        for name, op in self.operators.items():
            dim = op.dimension()
            if dim <= max_dimension:
                power = int(max(0, dim - 4))  # Canonical normalization
                
                if power == 0:
                    coef = f"{op.coefficient:.4g}"
                else:
                    coef = f"{op.coefficient:.4g} / Λ^{power}"
                
                if include_names:
                    terms.append(f"{coef} {name}")
                else:
                    terms.append(f"{coef} ({op})")
        
        return "L = " + " + ".join(terms)

    def power_counting(self, operator_name: str, energy_scale: float) -> float:
        """Perform power counting for an operator at a given energy scale.
        
        Args:
            operator_name: Name of the operator
            energy_scale: Energy scale E for power counting
            
        Returns:
            The power counting estimate
        """
        if operator_name not in self.operators:
            raise ValueError(f"Operator {operator_name} not found")
        
        operator = self.operators[operator_name]
        dim = operator.dimension()
        
        # Power counting: (E/Λ)^(d-4) where d is the operator dimension
        power = dim - 4
        result = operator.coefficient * (energy_scale / self.cutoff_scale) ** power
        
        return result
    
    def matching_condition(self, low_energy_op: str, high_energy_ops: List[str],
                          matching_scale: float) -> float:
        """Calculate a matching condition between EFTs.
        
        Args:
            low_energy_op: Name of the low-energy EFT operator
            high_energy_ops: List of high-energy EFT operators involved in matching
            matching_scale: Energy scale where the theories are matched
            
        Returns:
            The matching coefficient
        """
        # Simplified matching - in reality would involve loop calculations
        if low_energy_op not in self.operators:
            raise ValueError(f"Low energy operator {low_energy_op} not found")
        
        low_op = self.operators[low_energy_op]
        
        # Simply demonstrate the concept with a basic formula
        # In real calculations, this would be much more complex
        coefficient = 0.0
        for high_op_name in high_energy_ops:
            if high_op_name not in self.operators:
                raise ValueError(f"High energy operator {high_op_name} not found")
            
            high_op = self.operators[high_op_name]
            dim_diff = high_op.dimension() - low_op.dimension()
            
            # Simplified matching formula
            coefficient += high_op.coefficient * (matching_scale / self.cutoff_scale) ** dim_diff
        
        return coefficient

def create_sm_eft():
    """Create the Standard Model Effective Field Theory (SMEFT)."""
    # Define fields
    h = Field(name="h", spin=0, mass_dimension=1, statistics="boson")  # Higgs
    W = Field(name="W", spin=1, mass_dimension=1, statistics="boson")  # W boson
    B = Field(name="B", spin=1, mass_dimension=1, statistics="boson")  # B boson
    q = Field(name="q", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Left quark
    u = Field(name="u", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Right up quark
    d = Field(name="d", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Right down quark
    l = Field(name="l", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Left lepton
    e = Field(name="e", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Right electron
    
    # Create SMEFT with cutoff Λ = 1 TeV
    smeft = EffectiveTheory(
        name="Standard Model Effective Field Theory",
        fields=[h, W, B, q, u, d, l, e],
        cutoff_scale=1000  # 1 TeV
    )
    
    # Add some dimension-4 operators (SM)
    smeft.add_operator(
        Operator(terms=[[h, h]], derivatives=2),
        name="O_h"
    )
    
    smeft.add_operator(
        Operator(terms=[[q, q, h]], derivatives=0),
        name="O_q"
    )
    
    # Add some dimension-6 operators (BSM)
    smeft.add_operator(
        Operator(terms=[[h, h, h, h, h, h]], derivatives=0),
        name="O_h6"
    )
    
    smeft.add_operator(
        Operator(terms=[[l, l, e, e]], derivatives=0),
        name="O_le"
    )
    
    smeft.add_operator(
        Operator(terms=[[q, q, q, q]], derivatives=0),
        name="O_4q"
    )
    
    return smeft

def create_heavy_quark_eft():
    """Create the Heavy Quark Effective Theory (HQET)."""
    # Define fields
    Q = Field(name="Q", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Heavy quark
    q = Field(name="q", spin=0.5, mass_dimension=1.5, statistics="fermion")  # Light quark
    g = Field(name="g", spin=1, mass_dimension=1, statistics="boson")  # Gluon
    
    # Create HQET with cutoff Λ = m_b (bottom quark mass)
    hqet = EffectiveTheory(
        name="Heavy Quark Effective Theory",
        fields=[Q, q, g],
        cutoff_scale=4.2  # b-quark mass in GeV
    )
    
    # Add HQET operators
    hqet.add_operator(
        Operator(terms=[[Q, Q]], derivatives=1),
        name="O_kin"  # Kinetic energy operator
    )
    
    hqet.add_operator(
        Operator(terms=[[Q, Q, g]], derivatives=0),
        name="O_mag"  # Chromomagnetic operator
    )
    
    hqet.add_operator(
        Operator(terms=[[Q, q, q, Q]], derivatives=0),
        name="O_4q"  # Four-quark operator
    )
    
    return hqet

def plot_operator_scaling(eft, operator_names, energy_range):
    """Plot how operator contributions scale with energy.
    
    Args:
        eft: EffectiveTheory instance
        operator_names: List of operator names to plot
        energy_range: Array of energy values to evaluate at
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for op_name in operator_names:
        if op_name not in eft.operators:
            continue
            
        contributions = []
        for energy in energy_range:
            contributions.append(eft.power_counting(op_name, energy))
        
        ax.plot(energy_range, contributions, label=f"{op_name} (d={eft.operators[op_name].dimension()})")
    
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Relative Contribution")
    ax.set_title(f"Operator Scaling in {eft.name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Example 1: Create the SMEFT
    print("Example 1: SMEFT Lagrangian")
    smeft = create_sm_eft()
    print(smeft.lagrangian(max_dimension=6))
    
    # Example 2: Power counting
    print("\nExample 2: Power counting at different energy scales")
    for op_name in ["O_h", "O_h6"]:
        dim = smeft.operators[op_name].dimension()
        print(f"\nOperator {op_name} has dimension {dim}")
        
        for energy in [10, 100, 500, 1000]:  # GeV
            contribution = smeft.power_counting(op_name, energy)
            print(f"  Contribution at {energy} GeV: {contribution:.6g}")
    
    # Example 3: HQET
    print("\nExample 3: Heavy Quark Effective Theory")
    hqet = create_heavy_quark_eft()
    print(hqet.lagrangian(max_dimension=5))
    
    # Example 4: Plot operator scaling
    print("\nExample 4: Generating scaling plot for SMEFT operators")
    energy_range = np.logspace(1, 3, 100)  # 10 GeV to 1 TeV
    fig = plot_operator_scaling(smeft, ["O_h", "O_h6", "O_le", "O_4q"], energy_range)
    # Uncomment to display: plt.show() 