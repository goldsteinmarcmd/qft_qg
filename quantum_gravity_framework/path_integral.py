"""
Path Integral Formulation

This module implements path integral techniques for quantum gravity with
support for varying dimensions and numerical computation of quantum corrections.
"""

import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import networkx as nx
from scipy.special import gamma
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba
from numba import jit

from .quantum_spacetime_foundations import SpectralGeometry, CausalSet, DiscreteToContinum

class PathIntegral:
    """
    Implements path integral techniques for quantum gravity.
    
    This class provides methods to compute path integrals in spacetimes with
    varying dimensions, incorporating dimensional flow effects.
    """
    
    def __init__(self, dim_profile, discreteness_profile, beta=1.0, action_type='einstein_hilbert'):
        """
        Initialize path integral calculator.
        
        Parameters:
        -----------
        dim_profile : callable
            Function that returns dimension as a function of energy scale
        discreteness_profile : callable
            Function that returns discreteness parameter as a function of energy scale
        beta : float
            Inverse temperature parameter (1/kT)
        action_type : str
            Type of action: 'einstein_hilbert', 'higher_derivative', or 'asymptotic_safety'
        """
        self.dim_profile = dim_profile
        self.discreteness_profile = discreteness_profile
        self.beta = beta
        self.action_type = action_type
        
        # Monte Carlo configuration
        self.num_samples = 10000
        self.thermalization_steps = 1000
        self.correlation_steps = 10
        
        # Discretization parameters
        self.lattice_size = 16
        self.lattice_spacing = 0.1
        
        # Initialize action components
        self._initialize_action()
    
    def _initialize_action(self):
        """Initialize the gravitational action based on the chosen type."""
        # Action coupling constants (dimensionless)
        self.G_newton = 1.0  # Newton's constant in Planck units
        
        if self.action_type == 'einstein_hilbert':
            # Standard Einstein-Hilbert action with cosmological constant
            self.lambda_cosmological = 0.1
            self.higher_order_couplings = {}
            
        elif self.action_type == 'higher_derivative':
            # Higher derivative terms
            self.lambda_cosmological = 0.1
            self.higher_order_couplings = {
                'R_squared': 0.01,  # R² term
                'Ricci_squared': 0.005,  # R_μν R^μν term
                'Weyl_squared': 0.002  # C_μνρσ C^μνρσ term
            }
            
        elif self.action_type == 'asymptotic_safety':
            # Running couplings based on asymptotic safety
            self.lambda_cosmological = 0.1
            self.higher_order_couplings = {
                'R_squared': 0.01,
                'Ricci_squared': 0.005
            }
            # Define beta functions for running couplings
            self.beta_G = lambda g, λ, k: 2*g - g**2*(2 + self.dim_profile(k))
            self.beta_lambda = lambda g, λ, k: -2*λ + g*λ*(2 - self.dim_profile(k))
            
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def action_density(self, metric, energy_scale):
        """
        Compute the action density at a specific energy scale.
        
        Parameters:
        -----------
        metric : ndarray
            Metric tensor components
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Action density
        """
        # Get dimension at this scale
        dim = self.dim_profile(energy_scale)
        
        # Compute metric determinant (simplified for demonstration)
        if len(metric.shape) == 2:  # Full metric
            g_det = np.linalg.det(metric)
        else:  # Diagonal metric
            g_det = np.prod(metric)
        
        # Compute Ricci scalar (simplified)
        R = self._compute_ricci_scalar(metric, dim)
        
        # Base Einstein-Hilbert term
        action_val = (R - 2*self.lambda_cosmological) * np.sqrt(abs(g_det))
        
        # Add higher order terms if needed
        if self.action_type in ['higher_derivative', 'asymptotic_safety']:
            if 'R_squared' in self.higher_order_couplings:
                action_val += self.higher_order_couplings['R_squared'] * R**2 * np.sqrt(abs(g_det))
                
            # Other terms would be added similarly...
        
        # Apply energy scale dependence for asymptotic safety
        if self.action_type == 'asymptotic_safety':
            # Scale-dependent Newton's constant
            G_k = self.G_newton / (1 + self.G_newton * np.log(energy_scale + 1e-10))
            
            # Scale-dependent cosmological constant
            lambda_k = self.lambda_cosmological - self.G_newton * energy_scale**2 / (1 + self.G_newton * np.log(energy_scale + 1e-10))
            
            # Update action with running couplings
            action_val = (R/G_k - 2*lambda_k) * np.sqrt(abs(g_det))
        
        return action_val
    
    def _compute_ricci_scalar(self, metric, dim):
        """
        Compute the Ricci scalar from the metric (simplified).
        
        Parameters:
        -----------
        metric : ndarray
            Metric tensor components
        dim : float
            Dimension at current scale
            
        Returns:
        --------
        float
            Approximate Ricci scalar
        """
        # This is a simplified placeholder implementation
        # In a real implementation, we would compute derivatives and contractions
        
        # For demonstration, assume a simple case like a constant curvature space
        # R = k*d*(d-1) where k is the curvature constant
        if len(metric.shape) == 2:  # Full metric
            # Extract diagonal
            diag_metric = np.diag(metric)
            metric_trace = np.sum(diag_metric)
            
            # Simple approximation based on metric trace
            # (this is not physically accurate, just for demonstration)
            curvature_approx = 1.0 - metric_trace / dim
            R = curvature_approx * dim * (dim - 1)
        else:
            # Simple approximation based on diagonal metric
            metric_avg = np.mean(metric)
            curvature_approx = 1.0 - metric_avg
            R = curvature_approx * dim * (dim - 1)
        
        return R
    
    @jit(nopython=True)
    def _metropolis_step(self, config, beta, dim):
        """
        Perform a Metropolis update step for Monte Carlo integration.
        
        Parameters:
        -----------
        config : ndarray
            Current field configuration
        beta : float
            Inverse temperature
        dim : float
            Current dimension
            
        Returns:
        --------
        ndarray
            Updated configuration
        """
        # Size of lattice
        N = config.shape[0]
        
        # Choose a random site
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        # Compute current action
        S_old = self._local_action(config, i, j, dim)
        
        # Propose a new configuration by changing one value
        config_new = config.copy()
        config_new[i, j] += np.random.normal(0, 0.1)
        
        # Compute new action
        S_new = self._local_action(config_new, i, j, dim)
        
        # Accept or reject based on Metropolis criterion
        dS = S_new - S_old
        if dS < 0 or np.random.random() < np.exp(-beta * dS):
            return config_new
        else:
            return config
    
    @jit(nopython=True)
    def _local_action(self, config, i, j, dim):
        """
        Compute local contribution to the action.
        
        Parameters:
        -----------
        config : ndarray
            Field configuration
        i, j : int
            Lattice site indices
        dim : float
            Current dimension
            
        Returns:
        --------
        float
            Local action
        """
        # This is a simplified action for demonstration
        # In a real implementation, we would use the actual discretized action
        
        # Get lattice size
        N = config.shape[0]
        
        # Calculate discrete Laplacian
        laplacian = -4 * config[i, j]
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = (i + di) % N, (j + dj) % N
            laplacian += config[ni, nj]
        
        # Simple scalar field action (kinetic + mass terms)
        mass_sq = 1.0
        action = 0.5 * laplacian * config[i, j] + 0.5 * mass_sq * config[i, j]**2
        
        # Add dimension-dependent scaling
        action *= dim / 4.0
        
        return action
    
    def monte_carlo_path_integral(self, energy_scale):
        """
        Compute path integral using Monte Carlo integration.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Results of the path integral calculation
        """
        # Get dimension and discreteness at this scale
        dim = self.dim_profile(energy_scale)
        discreteness = self.discreteness_profile(energy_scale)
        
        print(f"Computing path integral at energy scale {energy_scale:.2e}")
        print(f"  Dimension: {dim:.3f}, Discreteness: {discreteness:.3f}")
        
        # Initialize lattice configuration
        config = np.random.randn(self.lattice_size, self.lattice_size) * 0.1
        
        # Thermalization steps
        for _ in range(self.thermalization_steps):
            config = self._metropolis_step(config, self.beta, dim)
        
        # Sampling steps
        samples = []
        observables = []
        
        for i in range(self.num_samples):
            # Update configuration with several steps to reduce correlations
            for _ in range(self.correlation_steps):
                config = self._metropolis_step(config, self.beta, dim)
            
            # Store sample
            samples.append(config.copy())
            
            # Compute observables
            obs = {
                'action': np.mean(self._local_action(config, 
                                               np.arange(self.lattice_size), 
                                               np.arange(self.lattice_size), 
                                               dim)),
                'field_sq': np.mean(config**2)
            }
            observables.append(obs)
            
            # Print progress
            if (i+1) % (self.num_samples // 10) == 0:
                print(f"  Progress: {(i+1) / self.num_samples * 100:.1f}%")
        
        # Compute average observables
        avg_observables = {
            'action': np.mean([obs['action'] for obs in observables]),
            'field_sq': np.mean([obs['field_sq'] for obs in observables])
        }
        
        # Compute correlation functions
        correlation = self._compute_correlation(samples)
        
        # Compute effective action
        effective_action = self._compute_effective_action(avg_observables, correlation, dim)
        
        return {
            'energy_scale': energy_scale,
            'dimension': dim,
            'discreteness': discreteness,
            'observables': avg_observables,
            'correlation': correlation,
            'effective_action': effective_action
        }
    
    def _compute_correlation(self, samples):
        """
        Compute correlation functions from MC samples.
        
        Parameters:
        -----------
        samples : list
            List of field configurations
            
        Returns:
        --------
        dict
            Correlation functions
        """
        # Convert samples to array
        samples_array = np.array(samples)
        
        # Two-point function (position space)
        N = self.lattice_size
        center = N // 2
        
        # Compute average field value
        avg_field = np.mean(samples_array)
        
        # Compute two-point function for different separations
        correlations = []
        distances = np.arange(0, N // 2)
        
        for r in distances:
            # Circular average of two-point function at distance r
            corr_sum = 0
            count = 0
            
            for i in range(N):
                for j in range(N):
                    x = (i - center) % N
                    y = (j - center) % N
                    dist = np.sqrt(x**2 + y**2)
                    
                    if abs(dist - r) < 0.5:
                        # Correlation <φ(0)φ(r)>
                        corr_sum += np.mean(samples_array[:, center, center] * samples_array[:, i, j])
                        count += 1
            
            if count > 0:
                correlations.append(corr_sum / count)
            else:
                correlations.append(0)
        
        # Normalize by <φ>²
        correlations = np.array(correlations) / avg_field**2
        
        return {
            'distances': distances,
            'two_point': correlations
        }
    
    def _compute_effective_action(self, observables, correlation, dim):
        """
        Compute effective action from observables and correlation functions.
        
        Parameters:
        -----------
        observables : dict
            Average observables
        correlation : dict
            Correlation functions
        dim : float
            Current dimension
            
        Returns:
        --------
        dict
            Effective action parameters
        """
        # Extract two-point function
        distances = correlation['distances']
        two_point = correlation['two_point']
        
        # Fit to expected form: G(r) ~ r^(-(d-2))
        # For small r, excluding r=0
        valid_idx = (distances > 0) & (distances < self.lattice_size // 4)
        if np.sum(valid_idx) > 3:
            log_r = np.log(distances[valid_idx])
            log_G = np.log(two_point[valid_idx])
            
            # Linear regression
            coeffs = np.polyfit(log_r, log_G, 1)
            slope = coeffs[0]
            
            # Extract effective dimension: slope = -(d_eff - 2)
            d_eff = -(slope) + 2
        else:
            d_eff = dim
        
        # Compute effective mass from correlation decay
        if np.sum(valid_idx) > 3:
            # Fit to exponential decay: G(r) ~ exp(-m_eff * r) / r^(d-2)
            log_G_corrected = log_G + (d_eff - 2) * log_r
            m_eff = -np.polyfit(distances[valid_idx], log_G_corrected, 1)[0]
            m_eff = max(0, m_eff)  # Ensure non-negative
        else:
            m_eff = 1.0
        
        # Compute quantum corrections to couplings
        # Simplistic model: G_eff = G * (1 + c * <φ²>)
        G_correction = 1.0 + 0.1 * observables['field_sq']
        
        return {
            'effective_dimension': d_eff,
            'effective_mass': m_eff,
            'G_correction': G_correction
        }
    
    def compute_quantum_corrections(self, process_name, energy_range=None, num_points=10):
        """
        Compute quantum gravity corrections to standard QFT processes.
        
        Parameters:
        -----------
        process_name : str
            Name of the process: 'scalar_propagator', 'graviton_propagator', 
            'vertex_correction', etc.
        energy_range : tuple, optional
            (min_scale, max_scale) in Planck units
        num_points : int
            Number of energy points to compute
            
        Returns:
        --------
        dict
            Quantum corrections results
        """
        if energy_range is None:
            # Default range: well below to near Planck scale
            energy_range = (1e-4, 2.0)
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(
            np.log10(energy_range[0]), 
            np.log10(energy_range[1]), 
            num_points
        )
        
        # Initialize result containers
        corrections = []
        dimensions = []
        
        # Compute corrections at each energy scale
        for scale in energy_scales:
            dim = self.dim_profile(scale)
            dimensions.append(dim)
            
            if process_name == 'scalar_propagator':
                # Compute correction to scalar propagator
                correction = self._scalar_propagator_correction(scale, dim)
            elif process_name == 'graviton_propagator':
                # Compute correction to graviton propagator
                correction = self._graviton_propagator_correction(scale, dim)
            elif process_name == 'vertex_correction':
                # Compute vertex correction
                correction = self._vertex_correction(scale, dim)
            else:
                raise ValueError(f"Unknown process: {process_name}")
            
            corrections.append(correction)
            print(f"Computed {process_name} correction at E={scale:.2e}, dimension={dim:.2f}: {correction:.6f}")
        
        return {
            'process': process_name,
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'corrections': corrections
        }
    
    def _scalar_propagator_correction(self, energy_scale, dim):
        """
        Compute quantum gravity correction to scalar propagator.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale
        dim : float
            Dimension at this scale
            
        Returns:
        --------
        float
            Propagator correction factor
        """
        # Leading quantum gravity correction to scalar propagator
        # Simplified model based on dimensional regularization
        
        # Momentum scale (p/M_Pl)
        p = energy_scale
        
        # Newton's constant dependence
        G_factor = self.G_newton * p**2
        
        # Dimension-dependent factor
        if abs(dim - 4) < 1e-6:
            # Special case for dim = 4 (avoids division by zero)
            dim_factor = np.log(p + 1e-10)
        else:
            # General case
            dim_factor = p**(dim - 4) / (dim - 4)
        
        # Combine factors with appropriate sign
        correction = 1.0 + G_factor * dim_factor * (dim - 2) / (4 * np.pi)**(dim/2)
        
        return correction
    
    def _graviton_propagator_correction(self, energy_scale, dim):
        """
        Compute quantum correction to graviton propagator.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale
        dim : float
            Dimension at this scale
            
        Returns:
        --------
        float
            Propagator correction factor
        """
        # Leading quantum correction to graviton propagator
        # More complex than scalar case due to gauge issues
        
        # Momentum scale
        p = energy_scale
        
        # Base correction (similar structure to scalar but different coefficients)
        if abs(dim - 4) < 1e-6:
            # Special case for dim = 4
            dim_factor = np.log(p + 1e-10)
        else:
            # General case with dimensional regularization
            dim_factor = p**(dim - 4) / (dim - 4)
        
        # Different coefficient for graviton
        grav_coeff = (dim**2 - dim - 4) / (dim - 2)
        
        # Combine factors
        correction = 1.0 + self.G_newton * p**2 * dim_factor * grav_coeff / (4 * np.pi)**(dim/2)
        
        return correction
    
    def _vertex_correction(self, energy_scale, dim):
        """
        Compute quantum gravity correction to a vertex.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale
        dim : float
            Dimension at this scale
            
        Returns:
        --------
        float
            Vertex correction factor
        """
        # Vertex correction from graviton exchange
        # Simplified model
        
        # Momentum scale
        p = energy_scale
        
        # Dimension-dependent coupling
        if abs(dim - 4) < 1e-6:
            # 4D case
            coupling = self.G_newton * np.log(p + 1e-10)
        else:
            # Higher-dimensional case
            coupling = self.G_newton * p**(dim - 4) / (dim - 4)
        
        # Vertex correction proportional to G * p^2 with dimension-dependent factor
        vertex_factor = (dim - 1) / (dim - 2)
        
        # Combine factors
        correction = 1.0 + coupling * p**2 * vertex_factor / (4 * np.pi)**(dim/2)
        
        return correction
    
    def compute_scattering_amplitude(self, process, energy, particles=None):
        """
        Compute scattering amplitude with quantum gravity corrections.
        
        Parameters:
        -----------
        process : str
            Process type: '2to2', '2to3', etc.
        energy : float
            Center of mass energy in Planck units
        particles : list, optional
            List of particle types involved
            
        Returns:
        --------
        dict
            Scattering amplitude results
        """
        # Default to scalar particles if not specified
        if particles is None:
            particles = ['scalar'] * 4  # For 2→2 process
        
        # Get dimension at this energy
        dim = self.dim_profile(energy)
        
        print(f"Computing {process} scattering for {particles} at E={energy:.2e}")
        print(f"  Dimension: {dim:.3f}")
        
        # Different handling based on process
        if process == '2to2':
            # 2→2 scattering
            if all(p == 'scalar' for p in particles[:4]):
                # Scalar-scalar scattering
                result = self._scalar_scalar_scattering(energy, dim)
            elif 'graviton' in particles[:4]:
                # Process involving gravitons
                result = self._graviton_scattering(energy, dim, particles[:4])
            else:
                # Default implementation
                result = self._default_scattering(energy, dim, particles[:4])
        else:
            # Other processes
            result = self._default_scattering(energy, dim, particles)
        
        return result
    
    def _scalar_scalar_scattering(self, energy, dim):
        """
        Compute scalar-scalar scattering with quantum gravity corrections.
        
        Parameters:
        -----------
        energy : float
            Energy scale
        dim : float
            Dimension at this energy
            
        Returns:
        --------
        dict
            Scattering results
        """
        # Simplified model for scalar-scalar scattering
        
        # Standard QFT contribution (λφ⁴ interaction)
        lambda_phi4 = 0.1  # Coupling constant
        standard_amp = lambda_phi4
        
        # Quantum gravity correction
        # Leading correction proportional to G_N * s
        G_correction = self.G_newton * energy**2
        
        # Dimension-dependent factor
        if abs(dim - 4) < 1e-6:
            # 4D case
            dim_factor = np.log(energy + 1e-10)
        else:
            # General dimension case
            dim_factor = energy**(dim - 4) / (dim - 4)
        
        # Combine corrections
        qg_correction = G_correction * dim_factor * (dim - 2) / (4 * np.pi)**(dim/2)
        
        # Full amplitude
        total_amp = standard_amp * (1 + qg_correction)
        
        # Scattering cross-section (simplified)
        cross_section = abs(total_amp)**2 / (8 * np.pi * energy**2)
        
        return {
            'energy': energy,
            'dimension': dim,
            'standard_amplitude': standard_amp,
            'qg_correction': qg_correction,
            'total_amplitude': total_amp,
            'cross_section': cross_section
        }
    
    def _graviton_scattering(self, energy, dim, particles):
        """
        Compute scattering involving gravitons.
        
        Parameters:
        -----------
        energy : float
            Energy scale
        dim : float
            Dimension at this energy
        particles : list
            List of particle types
            
        Returns:
        --------
        dict
            Scattering results
        """
        # Simplified model for graviton scattering
        
        # Count gravitons
        num_gravitons = sum(1 for p in particles if p == 'graviton')
        
        # Amplitude scales with G_N^(num_gravitons/2-1) * E^(num_gravitons-2)
        power_G = num_gravitons/2 - 1
        power_E = num_gravitons - 2
        
        # Base amplitude
        base_amp = (self.G_newton**power_G) * (energy**power_E)
        
        # Dimension-dependent factor
        dim_factor = (4*np.pi)**(2-dim/2) * gamma(dim/2-1)
        
        # Quantum corrections (simplified)
        qg_correction = self.G_newton * energy**2 * np.log(energy + 1e-10)
        
        # Full amplitude
        total_amp = base_amp * dim_factor * (1 + qg_correction)
        
        # Scattering cross-section (simplified)
        cross_section = abs(total_amp)**2 / (8 * np.pi * energy**(dim-2))
        
        return {
            'energy': energy,
            'dimension': dim,
            'particles': particles,
            'base_amplitude': base_amp,
            'qg_correction': qg_correction,
            'total_amplitude': total_amp,
            'cross_section': cross_section
        }
    
    def _default_scattering(self, energy, dim, particles):
        """
        Default scattering computation for other combinations.
        
        Parameters:
        -----------
        energy : float
            Energy scale
        dim : float
            Dimension at this energy
        particles : list
            List of particle types
            
        Returns:
        --------
        dict
            Scattering results
        """
        # Simple placeholder implementation
        # In a real implementation, different particle combinations would have specific treatments
        
        # Base amplitude estimation
        base_amp = 0.1 * energy**(4-dim)
        
        # Quantum gravity correction
        qg_correction = self.G_newton * energy**2 * np.log(energy + 1e-10)
        
        # Full amplitude
        total_amp = base_amp * (1 + qg_correction)
        
        # Scattering cross-section
        cross_section = abs(total_amp)**2 / (8 * np.pi * energy**(dim-2))
        
        return {
            'energy': energy,
            'dimension': dim,
            'particles': particles,
            'base_amplitude': base_amp,
            'qg_correction': qg_correction,
            'total_amplitude': total_amp,
            'cross_section': cross_section
        }
    
    def visualize_corrections(self, correction_results):
        """
        Visualize quantum gravity corrections.
        
        Parameters:
        -----------
        correction_results : dict
            Results from compute_quantum_corrections
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Extract results
        energy_scales = correction_results['energy_scales']
        dimensions = correction_results['dimensions']
        corrections = correction_results['corrections']
        process = correction_results['process']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot dimension vs energy
        ax1.semilogx(energy_scales, dimensions, 'k-', linewidth=2)
        ax1.set_xlabel('Energy Scale (Planck units)')
        ax1.set_ylabel('Effective Dimension')
        ax1.set_title('Dimensional Flow')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot correction vs energy
        ax2.loglog(energy_scales, np.abs(corrections), 'r-', linewidth=2)
        ax2.set_xlabel('Energy Scale (Planck units)')
        ax2.set_ylabel('Correction Magnitude')
        ax2.set_title(f'Quantum Gravity Correction to {process}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Create a simple dimension profile for testing
    dim_profile = lambda E: 4.0 - 2.0 / (1 + (E * 0.1)**(-2))
    discreteness_profile = lambda E: 1.0 / (1 + (E * 0.1)**(-2))
    
    # Create path integral calculator
    pi = PathIntegral(dim_profile, discreteness_profile)
    
    # Test quantum corrections
    corrections = pi.compute_quantum_corrections('scalar_propagator', 
                                              energy_range=(1e-3, 1.0),
                                              num_points=5)
    
    # Print results
    print("\nQuantum Gravity Corrections to Scalar Propagator:")
    for i, scale in enumerate(corrections['energy_scales']):
        print(f"E = {scale:.3e}, dimension = {corrections['dimensions'][i]:.3f}, "
              f"correction = {corrections['corrections'][i]:.6f}")
    
    # Test scattering calculation
    scattering = pi.compute_scattering_amplitude('2to2', 0.1)
    
    print("\nScattering Amplitude at E = 0.1 M_Pl:")
    print(f"Standard amplitude: {scattering['standard_amplitude']:.6f}")
    print(f"QG correction: {scattering['qg_correction']:.6f}")
    print(f"Total amplitude: {scattering['total_amplitude']:.6f}")
    print(f"Cross-section: {scattering['cross_section']:.6e}")
    
    # Plot corrections
    fig = pi.visualize_corrections(corrections)
    plt.savefig("quantum_gravity_corrections.png")
    print("\nVisualization saved to quantum_gravity_corrections.png") 