"""Renormalization module for QFT.

This module implements tools for handling renormalization in
quantum field theory calculations.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt

def dimensional_regularization_pole(epsilon):
    """Calculate the dimensional regularization pole.
    
    In dimensional regularization, we work in D = 4 - 2ε dimensions
    and extract the pole in ε as it approaches 0.
    
    Args:
        epsilon: The regularization parameter
        
    Returns:
        The regularization pole
    """
    # For divergent integrals, the result often contains terms like 1/ε
    return 1.0 / epsilon

def ms_subtraction(divergent_part, finite_part):
    """Apply MS (minimal subtraction) scheme.
    
    The MS scheme subtracts only the pole term 1/ε.
    
    Args:
        divergent_part: Coefficient of the 1/ε pole
        finite_part: Finite part of the expression
        
    Returns:
        Renormalized result
    """
    return finite_part

def msbar_subtraction(divergent_part, finite_part):
    """Apply MSbar (modified minimal subtraction) scheme.
    
    The MSbar scheme subtracts the pole term 1/ε and some 
    universal constants.
    
    Args:
        divergent_part: Coefficient of the 1/ε pole
        finite_part: Finite part of the expression
        
    Returns:
        Renormalized result
    """
    # In MSbar, we also subtract log(4π) - γ_E along with the 1/ε pole
    # where γ_E is the Euler-Mascheroni constant
    euler_gamma = 0.5772156649
    universal_constants = np.log(4 * np.pi) - euler_gamma
    
    return finite_part - divergent_part * universal_constants

def one_loop_correction(external_momentum, mass, coupling, cutoff=None, epsilon=None):
    """Calculate one-loop correction for a scalar field theory (φ⁴).
    
    Implements both cutoff and dimensional regularization.
    
    Args:
        external_momentum: External momentum
        mass: Physical mass
        coupling: Coupling constant
        cutoff: Momentum cutoff (for cutoff regularization)
        epsilon: Dimensional regularization parameter
        
    Returns:
        Tuple of (divergent_part, finite_part)
    """
    p_squared = np.sum(external_momentum**2)
    
    if cutoff is not None:
        # Cutoff regularization
        divergent_part = coupling**2 * (cutoff**2) / (16 * np.pi**2)
        finite_part = coupling**2 * (np.log(mass**2 / p_squared)) / (16 * np.pi**2)
    elif epsilon is not None:
        # Dimensional regularization
        divergent_part = coupling**2 / (16 * np.pi**2 * epsilon)
        finite_part = coupling**2 * (np.log(mass**2 / p_squared)) / (16 * np.pi**2)
    else:
        raise ValueError("Either cutoff or epsilon must be provided")
    
    return divergent_part, finite_part

def beta_function(coupling, order=1):
    """Calculate the beta function for φ⁴ theory.
    
    The beta function describes how the coupling constant
    changes with energy scale.
    
    Args:
        coupling: The coupling constant
        order: Order of the calculation (1 for one-loop)
        
    Returns:
        Value of the beta function
    """
    if order == 1:
        # One-loop beta function for φ⁴ theory
        return 3 * coupling**2 / (16 * np.pi**2)
    elif order == 2:
        # Two-loop beta function (simplified)
        return 3 * coupling**2 / (16 * np.pi**2) - 17 * coupling**3 / (256 * np.pi**4)
    else:
        raise ValueError("Order not implemented")

def running_coupling(coupling_at_mu, mu, energy, order=1):
    """Calculate the running coupling constant.
    
    Args:
        coupling_at_mu: Coupling at reference scale μ
        mu: Reference scale
        energy: Energy scale to calculate coupling at
        order: Order of the calculation (1 for one-loop)
        
    Returns:
        Coupling constant at the specified energy
    """
    if order == 1:
        # One-loop running coupling
        beta_0 = 3 / (16 * np.pi**2)
        return coupling_at_mu / (1 - beta_0 * coupling_at_mu * np.log(energy / mu))
    else:
        raise ValueError("Order not implemented")

def plot_running_coupling(coupling_at_mu, mu, energy_range, order=1):
    """Plot the running coupling as a function of energy.
    
    Args:
        coupling_at_mu: Coupling at reference scale μ
        mu: Reference scale
        energy_range: Array of energy values
        order: Order of the calculation
        
    Returns:
        Tuple of (figure, axis)
    """
    couplings = []
    
    for energy in energy_range:
        try:
            coupling = running_coupling(coupling_at_mu, mu, energy, order)
            couplings.append(coupling)
        except:
            # Landau pole or other issue
            couplings.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(energy_range, couplings)
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('Coupling constant')
    ax.set_title('Running coupling constant')
    ax.set_xscale('log')
    ax.grid(True)
    
    return fig, ax

def renormalized_propagator(momentum, bare_mass, bare_coupling, renormalized_mass, 
                           renormalization_scale):
    """Calculate a renormalized propagator.
    
    Args:
        momentum: 4-momentum
        bare_mass: Bare mass parameter
        bare_coupling: Bare coupling constant
        renormalized_mass: Physical (renormalized) mass
        renormalization_scale: Renormalization scale μ
        
    Returns:
        Renormalized propagator
    """
    p_squared = np.sum(momentum**2)
    
    # One-loop self-energy contribution (simplified)
    one_loop = bare_coupling**2 * np.log(renormalized_mass**2 / renormalization_scale**2) / (16 * np.pi**2)
    
    # Renormalized propagator
    return 1 / (p_squared - renormalized_mass**2 - one_loop)

# Example usage
if __name__ == "__main__":
    # Example parameters
    coupling = 0.1  # Coupling constant at 1 GeV
    mu = 1.0  # Reference scale (1 GeV)
    
    # Calculate beta function
    beta = beta_function(coupling)
    print(f"Beta function at coupling {coupling}: {beta}")
    
    # Calculate running coupling at different energies
    energies = [0.1, 1.0, 10.0, 100.0]  # GeV
    for energy in energies:
        running_coup = running_coupling(coupling, mu, energy)
        print(f"Coupling at {energy} GeV: {running_coup}")
    
    # Plot running coupling
    energy_range = np.logspace(-1, 3, 100)  # 0.1 to 1000 GeV
    plot_running_coupling(coupling, mu, energy_range)
    
    # Calculate one-loop correction with cutoff regularization
    external_momentum = np.array([10.0, 0, 0, 0])  # 10 GeV
    mass = 1.0  # 1 GeV
    cutoff = 1000.0  # 1 TeV cutoff
    
    div_part, fin_part = one_loop_correction(external_momentum, mass, coupling, cutoff=cutoff)
    renormalized = ms_subtraction(div_part, fin_part)
    
    print(f"One-loop correction (cutoff reg.): Divergent part = {div_part}, Finite part = {fin_part}")
    print(f"Renormalized result (MS scheme): {renormalized}")
    
    # Also calculate with dimensional regularization
    epsilon = 0.01  # Small epsilon for dimensional regularization
    div_part_dim, fin_part_dim = one_loop_correction(external_momentum, mass, coupling, epsilon=epsilon)
    renormalized_dim = msbar_subtraction(div_part_dim, fin_part_dim)
    
    print(f"One-loop correction (dim. reg.): Divergent part = {div_part_dim}, Finite part = {fin_part_dim}")
    print(f"Renormalized result (MSbar scheme): {renormalized_dim}") 