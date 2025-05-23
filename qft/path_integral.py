"""Path integral calculation module for QFT.

This module implements numerical and analytical methods for computing path integrals
in quantum field theory.
"""

import numpy as np
from scipy import integrate
import sympy as sp

def action_free_scalar(phi, mass, d_phi, potential=None):
    """Calculate the action for a free scalar field.
    
    Args:
        phi: Field configuration
        mass: Mass parameter
        d_phi: Derivative of the field
        potential: Optional additional potential term
        
    Returns:
        The calculated action
    """
    kinetic_term = 0.5 * np.sum(d_phi**2)
    mass_term = 0.5 * mass**2 * np.sum(phi**2)
    
    if potential is None:
        return kinetic_term + mass_term
    else:
        return kinetic_term + mass_term + potential(phi)

def calculate_path_integral(action, configurations, beta=1.0):
    """Calculate a path integral using Monte Carlo methods.
    
    Args:
        action: Action function to use
        configurations: Field configurations to sum over
        beta: Inverse temperature parameter
        
    Returns:
        Approximated path integral result
    """
    # Calculate action for each configuration
    actions = np.array([action(config) for config in configurations])
    
    # Calculate Boltzmann weights
    weights = np.exp(-beta * actions)
    
    # Normalize and sum
    return np.sum(weights) / len(configurations)

def generate_field_configurations(grid_size, num_configs, sigma=1.0):
    """Generate random field configurations for path integral calculation.
    
    Args:
        grid_size: Size of the spatial grid
        num_configs: Number of configurations to generate
        sigma: Standard deviation for the random fields
        
    Returns:
        Array of field configurations
    """
    return np.random.normal(0, sigma, (num_configs, grid_size))

def propagator_momentum_space(momentum, mass):
    """Calculate the momentum-space propagator for a scalar field.
    
    Args:
        momentum: Momentum 4-vector
        mass: Mass of the particle
        
    Returns:
        Scalar field propagator in momentum space
    """
    p_squared = np.sum(momentum**2)
    return 1 / (p_squared - mass**2 + 1e-10j)  # Small imaginary part for pole

def propagator_position_space(position, mass):
    """Calculate the position-space propagator for a scalar field.
    
    Args:
        position: Position 4-vector
        mass: Mass of the particle
        
    Returns:
        Scalar field propagator in position space
    """
    r = np.sqrt(np.sum(position**2))
    if r == 0:
        return 0
    return np.exp(-mass * r) / (4 * np.pi * r)

# Example usage
if __name__ == "__main__":
    # Generate field configurations
    configs = generate_field_configurations(grid_size=100, num_configs=1000)
    
    # Define an action (free scalar field)
    def example_action(phi):
        # Approximate derivative using finite differences
        d_phi = np.gradient(phi)
        return action_free_scalar(phi, mass=1.0, d_phi=d_phi)
    
    # Calculate path integral
    result = calculate_path_integral(example_action, configs)
    print(f"Path integral result: {result}") 