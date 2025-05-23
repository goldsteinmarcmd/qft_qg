"""Gauge theory module for QFT.

This module implements tools for working with gauge theories
in quantum field theory, including SU(N) and U(1) gauge theories.
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass

# SU(N) generators and structure constants
def su2_generators():
    """Return the generators of SU(2) (Pauli matrices)."""
    sigma1 = np.array([[0, 1], [1, 0]])
    sigma2 = np.array([[0, -1j], [1j, 0]])
    sigma3 = np.array([[1, 0], [0, -1]])
    
    # SU(2) generators are sigma/2
    return [sigma1/2, sigma2/2, sigma3/2]

def su3_generators():
    """Return the generators of SU(3) (Gell-Mann matrices)."""
    # Gell-Mann matrices
    lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    lambda8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
    
    # SU(3) generators are lambda/2
    return [lambda1/2, lambda2/2, lambda3/2, lambda4/2, 
            lambda5/2, lambda6/2, lambda7/2, lambda8/2]

def structure_constants(generators):
    """Calculate structure constants from the generators.
    
    Args:
        generators: List of generators for the gauge group
        
    Returns:
        Structure constants tensor f^{abc}
    """
    n = len(generators)
    f = np.zeros((n, n, n), dtype=complex)
    
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # f^{abc} = -2i Tr([T^a, T^b] T^c)
                commutator = generators[a] @ generators[b] - generators[b] @ generators[a]
                f[a, b, c] = -2j * np.trace(commutator @ generators[c])
    
    # Check if imaginary parts are negligible and return real array if so
    if np.allclose(f.imag, 0):
        return f.real
    return f

# Gauge field classes
@dataclass
class GaugeField:
    """Base class for gauge fields."""
    name: str
    group: str  # 'U(1)', 'SU(2)', 'SU(3)', etc.
    coupling: float
    
    def field_strength(self, A_mu, partial_mu_A_nu):
        """Calculate the field strength tensor F_μν.
        
        For an abelian gauge theory like QED, the field strength is:
        F_μν = ∂_μ A_ν - ∂_ν A_μ
        
        Args:
            A_mu: Gauge field components
            partial_mu_A_nu: Partial derivatives of gauge field
            
        Returns:
            Field strength tensor
        """
        raise NotImplementedError("Subclasses must implement field_strength")
    
    def action(self, F_munu):
        """Calculate the gauge field action.
        
        The action is -1/4 ∫ F_μν F^μν d^4x
        
        Args:
            F_munu: Field strength tensor
            
        Returns:
            Action value
        """
        raise NotImplementedError("Subclasses must implement action")

class U1GaugeField(GaugeField):
    """U(1) gauge field (e.g., photon in QED)."""
    
    def __init__(self, name="Photon", coupling=1/137.036):
        super().__init__(name=name, group="U(1)", coupling=coupling)
    
    def field_strength(self, A_mu, partial_mu_A_nu):
        """Calculate the U(1) field strength tensor.
        
        Args:
            A_mu: Gauge field 4-vector
            partial_mu_A_nu: 4x4 matrix of partial derivatives of A_mu
            
        Returns:
            Field strength tensor F_μν
        """
        # F_μν = ∂_μ A_ν - ∂_ν A_μ
        F_munu = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                F_munu[mu, nu] = partial_mu_A_nu[mu, nu] - partial_mu_A_nu[nu, mu]
        
        return F_munu
    
    def action(self, F_munu):
        """Calculate the U(1) gauge field action.
        
        Args:
            F_munu: Field strength tensor
            
        Returns:
            Action value
        """
        # S = -1/4 ∫ F_μν F^μν d^4x
        # We're calculating just the integrand F_μν F^μν
        F_squared = 0
        for mu in range(4):
            for nu in range(4):
                F_squared += F_munu[mu, nu] * F_munu[mu, nu]
        
        return -0.25 * F_squared

class SUNGaugeField(GaugeField):
    """SU(N) gauge field (e.g., gluon in QCD)."""
    
    def __init__(self, name="Gluon", group="SU(3)", coupling=0.1, generators=None):
        super().__init__(name=name, group=group, coupling=coupling)
        
        if generators is None:
            if group == "SU(2)":
                self.generators = su2_generators()
            elif group == "SU(3)":
                self.generators = su3_generators()
            else:
                raise ValueError(f"No default generators for {group}")
        else:
            self.generators = generators
        
        self.structure_constants = structure_constants(self.generators)
        self.dim = len(self.generators)
    
    def field_strength(self, A_mu, partial_mu_A_nu):
        """Calculate the SU(N) field strength tensor.
        
        Args:
            A_mu: Gauge field components, shape (4, dim)
            partial_mu_A_nu: Derivatives, shape (4, 4, dim)
            
        Returns:
            Field strength tensor F_μν^a
        """
        # F_μν^a = ∂_μ A_ν^a - ∂_ν A_μ^a + g f^{abc} A_μ^b A_ν^c
        F_munu = np.zeros((4, 4, self.dim))
        
        for a in range(self.dim):
            for mu in range(4):
                for nu in range(4):
                    # Linear terms
                    F_munu[mu, nu, a] = partial_mu_A_nu[mu, nu, a] - partial_mu_A_nu[nu, mu, a]
                    
                    # Nonlinear term from gauge self-interaction
                    for b in range(self.dim):
                        for c in range(self.dim):
                            F_munu[mu, nu, a] += self.coupling * self.structure_constants[a, b, c] * \
                                               A_mu[mu, b] * A_nu[nu, c]
        
        return F_munu
    
    def action(self, F_munu):
        """Calculate the SU(N) gauge field action.
        
        Args:
            F_munu: Field strength tensor, shape (4, 4, dim)
            
        Returns:
            Action value
        """
        # S = -1/4 ∫ F_μν^a F^{μν}_a d^4x
        F_squared = 0
        for a in range(self.dim):
            for mu in range(4):
                for nu in range(4):
                    F_squared += F_munu[mu, nu, a] * F_munu[mu, nu, a]
        
        return -0.25 * F_squared

# Gauge covariant derivative
def covariant_derivative(partial_mu, A_mu, generators, coupling):
    """Calculate the gauge covariant derivative.
    
    The covariant derivative is:
    D_μ = ∂_μ - ig A_μ^a T^a
    
    Args:
        partial_mu: Partial derivative operator
        A_mu: Gauge field components
        generators: Generators of the gauge group
        coupling: Gauge coupling constant
        
    Returns:
        Covariant derivative operator
    """
    # This is a function that will apply the covariant derivative to a field
    def D_mu(field):
        # Apply partial derivative
        result = partial_mu(field)
        
        # Add gauge field contribution
        if len(generators) == 1:  # U(1) case
            result -= 1j * coupling * A_mu * field
        else:  # SU(N) case
            for a, generator in enumerate(generators):
                result -= 1j * coupling * A_mu[a] * (generator @ field)
        
        return result
    
    return D_mu

# Wilson loop for non-perturbative gauge theory
def wilson_loop(gauge_links, loop_path):
    """Calculate Wilson loop for a given path.
    
    The Wilson loop is the trace of the product of gauge links
    around a closed loop, which is gauge invariant.
    
    Args:
        gauge_links: Dictionary mapping (x,y,z,t,mu) to gauge link matrices
        loop_path: List of (position, direction) tuples defining the loop
        
    Returns:
        Complex value of the Wilson loop
    """
    # Start with identity matrix
    if isinstance(next(iter(gauge_links.values())), np.ndarray):
        # SU(N) case
        dim = next(iter(gauge_links.values())).shape[0]
        product = np.eye(dim, dtype=complex)
    else:
        # U(1) case
        product = complex(1, 0)
    
    # Multiply gauge links around the loop
    current_pos = loop_path[0][0]
    
    for pos, direction in loop_path:
        if not np.array_equal(pos, current_pos):
            raise ValueError(f"Path is not continuous: {current_pos} to {pos}")
        
        # Get gauge link
        link = gauge_links.get((pos[0], pos[1], pos[2], pos[3], direction))
        
        if link is None:
            raise ValueError(f"No gauge link at {pos} in direction {direction}")
        
        # Update the product
        if isinstance(link, np.ndarray):
            product = product @ link
        else:
            product *= link
        
        # Move to the next position
        current_pos = current_pos.copy()
        current_pos[direction] += 1
    
    # Take trace if matrix, or return value if U(1)
    if isinstance(product, np.ndarray):
        return np.trace(product)
    return product

# Lattice gauge theory utilities
def generate_random_su2_matrix():
    """Generate a random SU(2) matrix."""
    # Parameterize SU(2) as a0 + i a.σ where a0^2 + a.a = 1
    a0 = np.random.uniform(-1, 1)
    ax = np.random.uniform(-1, 1)
    ay = np.random.uniform(-1, 1)
    az = np.random.uniform(-1, 1)
    
    # Normalize
    norm = np.sqrt(a0**2 + ax**2 + ay**2 + az**2)
    a0 /= norm
    ax /= norm
    ay /= norm
    az /= norm
    
    # Construct matrix
    return np.array([[a0 + 1j*az, ax + 1j*ay],
                     [-ax + 1j*ay, a0 - 1j*az]])

def generate_random_u1_phase():
    """Generate a random U(1) phase."""
    theta = np.random.uniform(0, 2*np.pi)
    return np.exp(1j * theta)

def initialize_lattice_gauge_field(lattice_size, gauge_group="U(1)"):
    """Initialize a lattice gauge field configuration.
    
    Args:
        lattice_size: Tuple of (nx, ny, nz, nt) lattice dimensions
        gauge_group: "U(1)" or "SU(2)" or "SU(3)"
        
    Returns:
        Dictionary of gauge links
    """
    nx, ny, nz, nt = lattice_size
    gauge_links = {}
    
    # Loop over all lattice sites and directions
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for t in range(nt):
                    for mu in range(4):  # 4 directions
                        if gauge_group == "U(1)":
                            gauge_links[(x, y, z, t, mu)] = generate_random_u1_phase()
                        elif gauge_group == "SU(2)":
                            gauge_links[(x, y, z, t, mu)] = generate_random_su2_matrix()
                        else:
                            raise ValueError(f"Gauge group {gauge_group} not implemented")
    
    return gauge_links

def plaquette(gauge_links, pos, mu, nu):
    """Calculate plaquette at a given position and plane.
    
    The plaquette is a 1x1 Wilson loop.
    
    Args:
        gauge_links: Dictionary of gauge links
        pos: (x,y,z,t) position
        mu, nu: Directions defining the plane
        
    Returns:
        Plaquette value
    """
    x, y, z, t = pos
    
    # Define the path of the 1x1 loop (plaquette)
    pos_array = np.array([x, y, z, t])
    
    loop_path = [
        (pos_array, mu),
        (pos_array + np.array([1 if i == mu else 0 for i in range(4)]), nu),
        (pos_array + np.array([1 if i == nu else 0 for i in range(4)]), mu),
        (pos_array, nu)
    ]
    
    # Calculate the Wilson loop
    return wilson_loop(gauge_links, loop_path)

def lattice_action(gauge_links, lattice_size, beta, gauge_group="U(1)"):
    """Calculate the Wilson gauge action for a lattice configuration.
    
    S = β ∑_p (1 - 1/N Re(Tr(U_p)))
    
    Args:
        gauge_links: Dictionary of gauge links
        lattice_size: Tuple of (nx, ny, nz, nt) lattice dimensions
        beta: β = 2N/g²
        gauge_group: Gauge group
        
    Returns:
        Action value
    """
    nx, ny, nz, nt = lattice_size
    action = 0.0
    
    # Determine normalization factor
    if gauge_group == "U(1)":
        norm = 1.0
    elif gauge_group == "SU(2)":
        norm = 2.0
    elif gauge_group == "SU(3)":
        norm = 3.0
    else:
        raise ValueError(f"Gauge group {gauge_group} not implemented")
    
    # Sum over all plaquettes
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for t in range(nt):
                    pos = (x, y, z, t)
                    
                    # Sum over all planes
                    for mu in range(4):
                        for nu in range(mu+1, 4):
                            p = plaquette(gauge_links, pos, mu, nu)
                            
                            if gauge_group == "U(1)":
                                action += 1.0 - np.real(p)
                            else:  # SU(N)
                                action += 1.0 - np.real(p) / norm
    
    return beta * action

# Example usage
if __name__ == "__main__":
    # Example 1: SU(3) gluon field strength
    print("Example 1: SU(3) field strength")
    gluon = SUNGaugeField(name="Gluon", group="SU(3)", coupling=0.1)
    
    # Random gauge field configuration (simplified)
    A_mu = np.random.rand(4, 8)  # 4 spacetime components, 8 SU(3) generators
    partial_mu_A_nu = np.random.rand(4, 4, 8)  # Derivatives
    
    # Calculate field strength
    F_munu = gluon.field_strength(A_mu, partial_mu_A_nu)
    action_value = gluon.action(F_munu)
    
    print(f"SU(3) action: {action_value}")
    
    # Example 2: U(1) field strength
    print("\nExample 2: U(1) field strength")
    photon = U1GaugeField(name="Photon", coupling=1/137.036)
    
    # Random electromagnetic field
    A_mu = np.random.rand(4)  # 4-vector potential
    partial_mu_A_nu = np.random.rand(4, 4)  # Derivatives
    
    # Calculate field strength
    F_munu = photon.field_strength(A_mu, partial_mu_A_nu)
    action_value = photon.action(F_munu)
    
    print(f"U(1) action: {action_value}")
    
    # Example 3: Lattice gauge theory
    print("\nExample 3: Lattice gauge theory")
    lattice_size = (4, 4, 4, 4)  # Small 4⁴ lattice
    gauge_links = initialize_lattice_gauge_field(lattice_size, gauge_group="U(1)")
    
    # Calculate action
    beta = 2.0
    action_value = lattice_action(gauge_links, lattice_size, beta, gauge_group="U(1)")
    
    print(f"U(1) lattice action with β={beta}: {action_value}") 