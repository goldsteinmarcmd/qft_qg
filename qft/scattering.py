"""Scattering amplitude calculation module for QFT.

This module implements tools for calculating scattering amplitudes
in quantum field theory.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
import sympy as sp

def mandelstam_variables(p1, p2, p3, p4):
    """Calculate Mandelstam variables for a 2→2 scattering process.
    
    Args:
        p1, p2: 4-momenta of incoming particles
        p3, p4: 4-momenta of outgoing particles
        
    Returns:
        Tuple (s, t, u) of Mandelstam variables
    """
    # s = (p1 + p2)²
    s = np.sum((p1 + p2)**2)
    
    # t = (p1 - p3)²
    t = np.sum((p1 - p3)**2)
    
    # u = (p1 - p4)²
    u = np.sum((p1 - p4)**2)
    
    return s, t, u

def klein_gordon_propagator(momentum, mass):
    """Calculate the Klein-Gordon propagator.
    
    Args:
        momentum: 4-momentum
        mass: Mass of the scalar particle
        
    Returns:
        Propagator value
    """
    p_squared = np.sum(momentum**2)
    return 1.0 / (p_squared - mass**2 + 1e-10j)

def dirac_propagator(momentum, mass):
    """Calculate the Dirac propagator for a fermion.
    
    This is a simplified version that returns the scalar part.
    For a complete implementation, the gamma matrices would be needed.
    
    Args:
        momentum: 4-momentum
        mass: Mass of the fermion
        
    Returns:
        Propagator value (scalar approximation)
    """
    p_squared = np.sum(momentum**2)
    return 1.0 / (p_squared - mass**2 + 1e-10j)

def photon_propagator(momentum):
    """Calculate the photon propagator in Feynman gauge.
    
    This is a simplified version that returns the scalar part.
    For a complete implementation, the tensor structure would be needed.
    
    Args:
        momentum: 4-momentum
        
    Returns:
        Propagator value (scalar approximation)
    """
    p_squared = np.sum(momentum**2)
    return -1.0 / (p_squared + 1e-10j)

def phase_space_2to2(s, m1, m2, m3, m4):
    """Calculate the 2→2 phase space factor.
    
    Args:
        s: Mandelstam s variable
        m1, m2: Masses of incoming particles
        m3, m4: Masses of outgoing particles
        
    Returns:
        Phase space factor
    """
    if s <= (m3 + m4)**2:
        return 0  # Below threshold
    
    lambda_func = lambda a, b, c: a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c
    
    # Kallen lambda function
    lambda_s = lambda_func(s, m3**2, m4**2)
    
    # Phase space factor
    return np.sqrt(lambda_s) / (8 * np.pi * s)

def cross_section(matrix_element_squared, s, m1, m2, m3, m4):
    """Calculate the cross section for a 2→2 scattering process.
    
    Args:
        matrix_element_squared: Square of the matrix element |M|²
        s: Mandelstam s variable
        m1, m2: Masses of incoming particles
        m3, m4: Masses of outgoing particles
        
    Returns:
        Differential cross section
    """
    # Calculate phase space factor
    dPS = phase_space_2to2(s, m1, m2, m3, m4)
    
    # Initial state momentum flux factor
    lambda_s = s**2 - 2*s*(m1**2 + m2**2) + (m1**2 - m2**2)**2
    p_initial = np.sqrt(lambda_s) / (2 * np.sqrt(s))
    
    # Cross section
    return (1 / (4 * p_initial * np.sqrt(s))) * matrix_element_squared * dPS

def compton_scattering_amplitude(p_electron_in, p_photon_in, p_electron_out, p_photon_out, 
                                alpha=1/137):
    """Calculate the Compton scattering amplitude (e⁻γ → e⁻γ).
    
    This is a simplified calculation that returns the squared amplitude
    summed over final spins and averaged over initial spins.
    
    Args:
        p_electron_in: 4-momentum of incoming electron
        p_photon_in: 4-momentum of incoming photon
        p_electron_out: 4-momentum of outgoing electron
        p_photon_out: 4-momentum of outgoing photon
        alpha: Fine structure constant
        
    Returns:
        Squared amplitude |M|²
    """
    # Calculate Mandelstam variables
    s, t, u = mandelstam_variables(p_electron_in, p_photon_in, 
                                  p_electron_out, p_photon_out)
    
    # Electron mass
    m_e = 0.511  # MeV
    
    # Calculate |M|² for Compton scattering
    # This is the Klein-Nishina formula
    prefactor = 2 * np.pi * alpha**2 / m_e**2
    
    term1 = m_e**2 * (1/t + 1/u)
    term2 = (s - m_e**2) * (1/t**2 + 1/u**2)
    term3 = 2 * (1/t + 1/u)
    
    return prefactor * (term1 + term2 - term3)

# Example usage
if __name__ == "__main__":
    # Example: Compton scattering
    # Define momenta in the lab frame (MeV)
    electron_mass = 0.511  # MeV
    
    # Incoming electron at rest
    p_e_in = np.array([electron_mass, 0, 0, 0])
    
    # Incoming photon with energy 10 MeV in z-direction
    photon_energy = 10.0  # MeV
    p_gamma_in = np.array([photon_energy, 0, 0, photon_energy])
    
    # Outgoing electron (this would be determined by conservation laws in reality)
    p_e_out = np.array([5.0, 1.0, 0, 3.0])
    
    # Outgoing photon (from momentum conservation)
    p_gamma_out = p_e_in + p_gamma_in - p_e_out
    
    # Calculate amplitude squared
    amplitude_squared = compton_scattering_amplitude(p_e_in, p_gamma_in, p_e_out, p_gamma_out)
    
    print(f"Compton scattering amplitude squared: {amplitude_squared} MeV⁻²")
    
    # Calculate Mandelstam variables
    s, t, u = mandelstam_variables(p_e_in, p_gamma_in, p_e_out, p_gamma_out)
    print(f"Mandelstam s: {s} MeV²")
    print(f"Mandelstam t: {t} MeV²")
    print(f"Mandelstam u: {u} MeV²") 