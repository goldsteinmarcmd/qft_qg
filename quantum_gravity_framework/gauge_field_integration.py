"""
Gauge Field Integration with Quantum Gravity

This module implements gauge field integration with dimensional flow effects,
allowing for consistent treatment of gauge theories across energy scales where
quantum gravity effects become important.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, linalg as splinalg
import networkx as nx

from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG
from quantum_gravity_framework.non_perturbative_path_integral import NonPerturbativePathIntegral
from quantum_gravity_framework.unified_approach import UnifiedQGApproach


class GaugeFieldIntegration:
    """
    Implements gauge field integration with dimensional flow effects.
    """
    
    def __init__(self, gauge_group="U1", dim_uv=2.0, dim_ir=4.0, 
                 transition_scale=1.0, lattice_size=10, beta=1.0):
        """
        Initialize gauge field integration.
        
        Parameters:
        -----------
        gauge_group : str
            Gauge group ("U1", "SU2", "SU3")
        dim_uv : float
            UV (high energy) spectral dimension
        dim_ir : float
            IR (low energy) spectral dimension
        transition_scale : float
            Scale of dimension transition (in Planck units)
        lattice_size : int
            Size of the lattice for discretization
        beta : float
            Inverse coupling parameter (1/g²)
        """
        self.gauge_group = gauge_group
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        self.lattice_size = lattice_size
        self.beta = beta
        
        # Initialize dimensional flow RG
        self.rg = DimensionalFlowRG(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize unified approach
        self.unified = UnifiedQGApproach(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale,
            lattice_size=lattice_size,
            beta=beta
        )
        
        # Initialize gauge field configuration
        self.links = self._initialize_gauge_fields()
        
        # Setup gauge group parameters
        self._setup_gauge_group()
        
        # Store results
        self.results = {}
        
    def _setup_gauge_group(self):
        """Set up parameters specific to the gauge group."""
        if self.gauge_group == "U1":
            # U(1) gauge group (electromagnetism/QED)
            self.group_dim = 1
            self.casimir = 1.0
            # U(1) has no structure constants (abelian)
            self.structure_constants = None
            
        elif self.gauge_group == "SU2":
            # SU(2) gauge group (weak isospin)
            self.group_dim = 3
            self.casimir = 2.0
            # SU(2) structure constants = Levi-Civita symbol
            self.structure_constants = np.zeros((3, 3, 3))
            for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
                self.structure_constants[i,j,k] = 1.0
            for i, j, k in [(2,1,0), (0,2,1), (1,0,2)]:
                self.structure_constants[i,j,k] = -1.0
                
        elif self.gauge_group == "SU3":
            # SU(3) gauge group (QCD)
            self.group_dim = 8
            self.casimir = 3.0
            # SU(3) has more complex structure constants
            # This is a simplified implementation
            self.structure_constants = np.zeros((8, 8, 8))
            # Usually filled with actual SU(3) structure constants
            # Placeholder for demonstration
            
        else:
            raise ValueError(f"Unsupported gauge group: {self.gauge_group}")
    
    def _initialize_gauge_fields(self):
        """
        Initialize gauge field configuration on links.
        
        Returns:
        --------
        dict
            Gauge field configuration on lattice links
        """
        # For simplicity, we'll use a regular cubic lattice
        lattice_shape = (self.lattice_size,) * int(self.dim_ir)
        
        # Initialize links (gauge fields live on links)
        links = {}
        
        # For each lattice site and direction
        # In a gauge theory, we associate gauge fields with links between sites
        for site_idx in np.ndindex(lattice_shape):
            for mu in range(int(self.dim_ir)):
                # Link is identified by (site, direction)
                # For U(1), link variables are complex phases e^{i*theta}
                # For SU(N), they are SU(N) matrices
                
                if self.gauge_group == "U1":
                    # U(1) gauge theory uses complex phases
                    theta = np.random.uniform(0, 2*np.pi)
                    link_var = np.exp(1j * theta)
                    
                elif self.gauge_group == "SU2":
                    # SU(2) gauge theory uses 2x2 special unitary matrices
                    # A convenient parameterization: U = a0*I + i*sum(aj*sigma_j)
                    # where sigma_j are Pauli matrices and a0^2 + a1^2 + a2^2 + a3^2 = 1
                    a = np.random.randn(4)
                    a = a / np.sqrt(np.sum(a**2))  # Normalize
                    
                    # Convert to 2x2 matrix
                    a0, a1, a2, a3 = a
                    link_var = np.array([
                        [a0 + 1j*a3, a2 + 1j*a1],
                        [-a2 + 1j*a1, a0 - 1j*a3]
                    ])
                    
                elif self.gauge_group == "SU3":
                    # SU(3) gauge theory uses 3x3 special unitary matrices
                    # This is just a placeholder - actual implementation would use
                    # proper SU(3) parameterization
                    H = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
                    H = (H + H.conj().T) / 2  # Make Hermitian
                    U, _ = np.linalg.eigh(H)
                    Udet = np.exp(-1j * np.angle(np.linalg.det(U)))  # Ensure det(U)=1
                    link_var = U * Udet
                
                # Store link variable
                links[(site_idx, mu)] = link_var
                
        return links
    
    def compute_wilson_loop(self, loop_size=1, energy_scale=1.0, num_samples=100):
        """
        Compute Wilson loop at a given energy scale.
        
        Parameters:
        -----------
        loop_size : int
            Size of square Wilson loop
        energy_scale : float
            Energy scale in Planck units
        num_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict
            Wilson loop results
        """
        print(f"Computing {loop_size}x{loop_size} Wilson loop at scale {energy_scale:.2e}...")
        
        # Get spectral dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Dimension-dependent coupling
        # Compute RG flow if not already computed
        if not self.rg.flow_results:
            self.rg.compute_rg_flow(scale_range=(energy_scale*0.1, energy_scale*10))
        
        # Extract coupling at this scale
        scales = self.rg.flow_results['scales']
        idx = np.abs(scales - energy_scale).argmin()
        
        if 'g' in self.rg.flow_results['coupling_trajectories']:
            g_coupling = self.rg.flow_results['coupling_trajectories']['g'][idx]
            # Convert to beta = 1/g² for lattice gauge theory
            beta_scale = 1.0 / (g_coupling**2)
        else:
            # Default scaling behavior if RG doesn't have 'g'
            beta_scale = self.beta * (energy_scale / self.transition_scale)**(dimension - self.dim_ir)
            
        print(f"  Effective beta at dimension {dimension:.3f}: {beta_scale:.4f}")
        
        # Dimension-dependent Wilson loop calculation
        # For non-integer dimensions, we use fractional calculus concepts
        dim_int = int(np.round(dimension))
        
        # Correction factor for non-integer dimensions
        dim_correction = np.exp((dimension - dim_int) * (loop_size / self.lattice_size))
        
        # Monte Carlo sampling of Wilson loops
        loop_values = []
        
        # In a real implementation, we would use a proper Metropolis algorithm
        # to update gauge configurations and measure Wilson loops
        # This is a simplified demonstration
        
        for _ in range(num_samples):
            # Generate a new gauge configuration or update existing one
            # (In a real implementation, this would be done with proper Metropolis)
            self._update_gauge_configuration(beta_scale)
            
            # Measure Wilson loop
            loop_value = self._measure_wilson_loop(loop_size, dim_int)
            
            # Apply dimension correction
            loop_value *= dim_correction
            
            loop_values.append(loop_value)
        
        # Compute mean and error
        loop_mean = np.mean(loop_values)
        loop_error = np.std(loop_values) / np.sqrt(num_samples)
        
        print(f"  Wilson loop expectation value: {loop_mean:.6f} ± {loop_error:.6f}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'beta': beta_scale,
            'loop_size': loop_size,
            'loop_mean': loop_mean,
            'loop_error': loop_error,
            'loop_values': loop_values,
            'dimension_correction': dim_correction
        }
        
        return result
    
    def _update_gauge_configuration(self, beta):
        """
        Update gauge configuration using Metropolis algorithm.
        
        Parameters:
        -----------
        beta : float
            Inverse coupling parameter (1/g²)
        """
        # In a real implementation, this would perform Metropolis updates
        # For simplicity, we'll just generate new random links with a bias toward
        # ordered configuration for higher beta (lower coupling)
        
        # Order parameter: higher beta -> more ordered configuration
        order = 1.0 - np.exp(-beta)
        
        for link_key in self.links:
            if self.gauge_group == "U1":
                # U(1) case: generate phases biased toward zero for high beta
                theta = np.random.normal(0, 1.0 / (beta + 0.1))
                self.links[link_key] = np.exp(1j * theta)
                
            elif self.gauge_group == "SU2":
                # SU(2) case: generate matrices biased toward identity for high beta
                a = np.random.randn(4)
                a = a / np.sqrt(np.sum(a**2))  # Normalize
                
                # Bias toward identity (a0=1, a1=a2=a3=0)
                a[0] = a[0] * (1-order) + order
                a[1:] = a[1:] * (1-order)
                a = a / np.sqrt(np.sum(a**2))  # Renormalize
                
                # Convert to 2x2 matrix
                a0, a1, a2, a3 = a
                self.links[link_key] = np.array([
                    [a0 + 1j*a3, a2 + 1j*a1],
                    [-a2 + 1j*a1, a0 - 1j*a3]
                ])
                
            elif self.gauge_group == "SU3":
                # SU(3) case: similar idea but more complex
                # This is a simplified placeholder
                H = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
                H = (H + H.conj().T) / 2  # Make Hermitian
                
                # Bias toward identity
                if np.random.random() < order:
                    H = H * 0.1  # Closer to identity
                    
                U, _ = np.linalg.eigh(H)
                Udet = np.exp(-1j * np.angle(np.linalg.det(U)))  # Ensure det(U)=1
                self.links[link_key] = U * Udet
    
    def _measure_wilson_loop(self, loop_size, dimension):
        """
        Measure a Wilson loop in the current gauge configuration.
        
        Parameters:
        -----------
        loop_size : int
            Size of square Wilson loop
        dimension : int
            Spacetime dimension
            
        Returns:
        --------
        float or complex
            Wilson loop value
        """
        # For simplicity, we'll measure a square loop in the x-y plane
        # starting from the origin
        
        # Origin site
        origin = (0,) * dimension
        
        # Wilson loop is the trace of the product of link variables around a loop
        
        # Links for the path: 
        # 1. Move loop_size steps in x direction
        # 2. Move loop_size steps in y direction
        # 3. Move loop_size steps in -x direction
        # 4. Move loop_size steps in -y direction
        
        # Define the path
        path = []
        
        # Current position
        current = list(origin)
        
        # Direction 0 (x) for loop_size steps
        for i in range(loop_size):
            path.append((tuple(current), 0))
            current[0] = (current[0] + 1) % self.lattice_size
            
        # Direction 1 (y) for loop_size steps
        for i in range(loop_size):
            path.append((tuple(current), 1))
            current[1] = (current[1] + 1) % self.lattice_size
            
        # Direction 0 (x) backward for loop_size steps
        for i in range(loop_size):
            # Move backward: need the link from previous site
            current[0] = (current[0] - 1) % self.lattice_size
            path.append((tuple(current), 0))
            
        # Direction 1 (y) backward for loop_size steps
        for i in range(loop_size):
            # Move backward: need the link from previous site
            current[1] = (current[1] - 1) % self.lattice_size
            path.append((tuple(current), 1))
        
        # Compute path ordered product
        if self.gauge_group == "U1":
            # For U(1), just multiply the complex phases
            product = 1.0
            for site, direction in path:
                # For backward links, use complex conjugate
                if direction == 0 and path.index((site, direction)) >= loop_size*2:
                    product *= np.conj(self.links.get((site, direction), 1.0))
                elif direction == 1 and path.index((site, direction)) >= loop_size*3:
                    product *= np.conj(self.links.get((site, direction), 1.0))
                else:
                    product *= self.links.get((site, direction), 1.0)
                    
            # For U(1), the trace is just the complex number itself
            return np.real(product)  # Should be real up to numerical precision
        
        elif self.gauge_group == "SU2":
            # For SU(2), multiply 2x2 matrices
            product = np.eye(2, dtype=complex)
            for site, direction in path:
                if direction == 0 and path.index((site, direction)) >= loop_size*2:
                    # Backward: use Hermitian conjugate
                    link = self.links.get((site, direction), np.eye(2))
                    product = product @ link.conj().T
                elif direction == 1 and path.index((site, direction)) >= loop_size*3:
                    # Backward: use Hermitian conjugate
                    link = self.links.get((site, direction), np.eye(2))
                    product = product @ link.conj().T
                else:
                    link = self.links.get((site, direction), np.eye(2))
                    product = product @ link
            
            # Trace is the sum of diagonal elements
            return np.real(np.trace(product))
            
        elif self.gauge_group == "SU3":
            # For SU(3), multiply 3x3 matrices
            product = np.eye(3, dtype=complex)
            for site, direction in path:
                if direction == 0 and path.index((site, direction)) >= loop_size*2:
                    # Backward: use Hermitian conjugate
                    link = self.links.get((site, direction), np.eye(3))
                    product = product @ link.conj().T
                elif direction == 1 and path.index((site, direction)) >= loop_size*3:
                    # Backward: use Hermitian conjugate
                    link = self.links.get((site, direction), np.eye(3))
                    product = product @ link.conj().T
                else:
                    link = self.links.get((site, direction), np.eye(3))
                    product = product @ link
            
            # Trace is the sum of diagonal elements
            return np.real(np.trace(product))
    
    def compute_multiscale_wilson_loops(self, loop_sizes=None, num_scales=5, num_samples=100):
        """
        Compute Wilson loops across multiple energy scales.
        
        Parameters:
        -----------
        loop_sizes : list, optional
            Sizes of Wilson loops to compute
        num_scales : int
            Number of energy scales to compute
        num_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict
            Wilson loop results at different scales and sizes
        """
        if loop_sizes is None:
            loop_sizes = [1, 2, 3]
            
        print(f"Computing Wilson loops across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-3, 3, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'loop_sizes': loop_sizes,
            'wilson_loops': {size: [] for size in loop_sizes},
            'wilson_errors': {size: [] for size in loop_sizes}
        }
        
        # Compute Wilson loops at each scale and size
        for scale in energy_scales:
            # Get dimension at this scale
            dimension = self.rg.compute_spectral_dimension(scale)
            results['dimensions'].append(dimension)
            
            # Compute Wilson loops of each size
            for size in loop_sizes:
                loop_result = self.compute_wilson_loop(
                    loop_size=size,
                    energy_scale=scale,
                    num_samples=num_samples
                )
                
                # Store results
                results['wilson_loops'][size].append(loop_result['loop_mean'])
                results['wilson_errors'][size].append(loop_result['loop_error'])
        
        # Store complete results
        self.results['multiscale_wilson_loops'] = results
        return results
    
    def compute_string_tension(self, energy_scale=1.0, max_loop_size=4, num_samples=100):
        """
        Compute the string tension at a given energy scale.
        
        String tension is extracted from the area-law behavior of Wilson loops:
        <W(R,T)> ~ exp(-σ·R·T) for large loops
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        max_loop_size : int
            Maximum loop size for fitting
        num_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict
            String tension results
        """
        print(f"Computing string tension at scale {energy_scale:.2e}...")
        
        # Compute Wilson loops of different sizes
        loop_sizes = list(range(1, max_loop_size + 1))
        wilson_values = []
        wilson_errors = []
        
        for size in loop_sizes:
            result = self.compute_wilson_loop(
                loop_size=size,
                energy_scale=energy_scale,
                num_samples=num_samples
            )
            wilson_values.append(result['loop_mean'])
            wilson_errors.append(result['loop_error'])
        
        # Convert to numpy arrays
        loop_sizes = np.array(loop_sizes)
        wilson_values = np.array(wilson_values)
        wilson_errors = np.array(wilson_errors)
        
        # Compute areas
        areas = loop_sizes**2
        
        # Fit to area law: log(<W>) = -σ·A + c
        # Take log of Wilson loop values
        log_wilson = np.log(np.maximum(wilson_values, 1e-10))  # Avoid log(0)
        
        # Simple linear fit: log(<W>) = -σ·A + c
        # weights from errors (error propagation for logarithm)
        weights = 1.0 / (wilson_errors / np.maximum(wilson_values, 1e-10))
        coeffs = np.polyfit(areas, log_wilson, 1, w=weights)
        
        # String tension is the negative slope
        string_tension = -coeffs[0]
        # y-intercept
        intercept = coeffs[1]
        
        # Compute fit quality
        fit_line = coeffs[0] * areas + coeffs[1]
        residuals = log_wilson - fit_line
        chi_squared = np.sum((residuals * weights)**2)
        
        print(f"  String tension: σ = {string_tension:.6f}")
        print(f"  Fit quality: χ² = {chi_squared:.2f}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': self.rg.compute_spectral_dimension(energy_scale),
            'string_tension': string_tension,
            'intercept': intercept,
            'loop_sizes': loop_sizes,
            'wilson_values': wilson_values,
            'wilson_errors': wilson_errors,
            'chi_squared': chi_squared
        }
        
        return result
    
    def compute_multiscale_string_tension(self, num_scales=5, max_loop_size=4, num_samples=100):
        """
        Compute string tension across multiple energy scales.
        
        Parameters:
        -----------
        num_scales : int
            Number of energy scales to compute
        max_loop_size : int
            Maximum loop size for fitting
        num_samples : int
            Number of Monte Carlo samples per loop
            
        Returns:
        --------
        dict
            String tension results at different scales
        """
        print(f"Computing string tension across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-3, 3, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'string_tensions': []
        }
        
        # Compute string tension at each scale
        for scale in energy_scales:
            tension_result = self.compute_string_tension(
                energy_scale=scale,
                max_loop_size=max_loop_size,
                num_samples=num_samples
            )
            
            # Store results
            results['dimensions'].append(tension_result['dimension'])
            results['string_tensions'].append(tension_result['string_tension'])
        
        # Store complete results
        self.results['multiscale_string_tension'] = results
        return results
    
    def compute_glueball_mass(self, energy_scale=1.0, max_time=10, num_samples=100):
        """
        Compute glueball mass at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        max_time : int
            Maximum time separation for correlator
        num_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict
            Glueball mass results
        """
        print(f"Computing glueball mass at scale {energy_scale:.2e}...")
        
        # Get spectral dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Set up effective coupling at this scale
        scales = self.rg.flow_results['scales']
        idx = np.abs(scales - energy_scale).argmin()
        
        if 'g' in self.rg.flow_results['coupling_trajectories']:
            g_coupling = self.rg.flow_results['coupling_trajectories']['g'][idx]
            beta_scale = 1.0 / (g_coupling**2)
        else:
            beta_scale = self.beta * (energy_scale / self.transition_scale)**(dimension - self.dim_ir)
        
        # Create plaquette operators
        # (In a full implementation, we would compute plaquette-plaquette correlators)
        correlator_values = []
        correlator_errors = []
        
        # Time separations
        time_seps = np.arange(1, max_time + 1)
        
        # In a real implementation, we would:
        # 1. Generate gauge configurations
        # 2. Compute plaquette-plaquette correlators at different time separations
        # 3. Extract glueball mass from exponential decay
        
        # Simplified demonstration
        for t in time_seps:
            # Simulate correlator with expected exponential decay
            # C(t) ~ exp(-m·t)
            # Use dimension-dependent mass
            mass = 1.0 + (dimension - self.dim_ir) * 0.5
            
            # Add fluctuations
            corr_values = []
            for _ in range(num_samples):
                # Generate a value with expected exponential decay plus noise
                value = np.exp(-mass * t) * (1.0 + np.random.normal(0, 0.1))
                corr_values.append(value)
            
            # Compute mean and error
            corr_mean = np.mean(corr_values)
            corr_error = np.std(corr_values) / np.sqrt(num_samples)
            
            correlator_values.append(corr_mean)
            correlator_errors.append(corr_error)
        
        # Convert to numpy arrays
        correlator_values = np.array(correlator_values)
        correlator_errors = np.array(correlator_errors)
        
        # Fit to exponential decay: log(C(t)) = -m·t + c
        log_corr = np.log(np.maximum(correlator_values, 1e-10))  # Avoid log(0)
        
        # Weights from errors (error propagation for logarithm)
        weights = 1.0 / (correlator_errors / np.maximum(correlator_values, 1e-10))
        
        # Linear fit
        coeffs = np.polyfit(time_seps, log_corr, 1, w=weights)
        
        # Glueball mass is the negative slope
        glueball_mass = -coeffs[0]
        intercept = coeffs[1]
        
        # Compute fit quality
        fit_line = coeffs[0] * time_seps + coeffs[1]
        residuals = log_corr - fit_line
        chi_squared = np.sum((residuals * weights)**2)
        
        print(f"  Glueball mass: m = {glueball_mass:.6f}")
        print(f"  Fit quality: χ² = {chi_squared:.2f}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'glueball_mass': glueball_mass,
            'intercept': intercept,
            'time_seps': time_seps,
            'correlator_values': correlator_values,
            'correlator_errors': correlator_errors,
            'chi_squared': chi_squared
        }
        
        return result
    
    def plot_multiscale_wilson_loops(self, save_path=None):
        """
        Plot Wilson loops across energy scales.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'multiscale_wilson_loops' not in self.results:
            raise ValueError("No Wilson loop results available. Run compute_multiscale_wilson_loops first.")
            
        results = self.results['multiscale_wilson_loops']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Wilson loops vs energy scale
        for size in results['loop_sizes']:
            axs[0].errorbar(
                results['energy_scales'],
                results['wilson_loops'][size],
                yerr=results['wilson_errors'][size],
                fmt='o-',
                label=f'Size = {size}'
            )
        
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Energy Scale (Planck units)')
        axs[0].set_ylabel('Wilson Loop Value')
        axs[0].set_title('Wilson Loops across Energy Scales')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # Plot 2: Wilson loops vs spectral dimension
        for size in results['loop_sizes']:
            axs[1].errorbar(
                results['dimensions'],
                results['wilson_loops'][size],
                yerr=results['wilson_errors'][size],
                fmt='o-',
                label=f'Size = {size}'
            )
        
        axs[1].set_xlabel('Spectral Dimension')
        axs[1].set_ylabel('Wilson Loop Value')
        axs[1].set_title('Wilson Loops vs Spectral Dimension')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_string_tension(self, save_path=None):
        """
        Plot string tension across energy scales.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'multiscale_string_tension' not in self.results:
            raise ValueError("No string tension results available. Run compute_multiscale_string_tension first.")
            
        results = self.results['multiscale_string_tension']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot string tension vs energy scale
        ax.plot(results['energy_scales'], results['string_tensions'], 'ro-', linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Energy Scale (Planck units)')
        ax.set_ylabel('String Tension σ')
        ax.set_title(f'String Tension vs Energy Scale ({self.gauge_group} Gauge Theory)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference line for asymptotic freedom
        x_vals = np.logspace(-3, 3, 100)
        if self.gauge_group in ["SU2", "SU3"]:
            # For non-Abelian gauge theories, asymptotic freedom predicts σ ~ 1/log(E)
            y_vals = 1.0 / np.log(1.0 + x_vals * 10)
            ax.plot(x_vals, y_vals, 'b--', alpha=0.7, label='Asymptotic Freedom')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def derive_dimension_dependent_gauge_scaling(self):
        """
        Derive how gauge theory parameters scale with spectral dimension.
        
        Returns:
        --------
        dict
            Scaling relations
        """
        print("Deriving dimension-dependent gauge scaling laws...")
        
        # Need both string tension and Wilson loop results
        if 'multiscale_string_tension' not in self.results or 'multiscale_wilson_loops' not in self.results:
            raise ValueError("Need both string tension and Wilson loop results. Run computations first.")
        
        st_results = self.results['multiscale_string_tension']
        wl_results = self.results['multiscale_wilson_loops']
        
        # Extract data
        dimensions = st_results['dimensions']
        string_tensions = st_results['string_tensions']
        
        # Find how string tension scales with dimension
        # σ(d) ~ (d - 2)^α · (4 - d)^β
        
        # Define fit function
        def scaling_func(d, alpha, beta, gamma):
            return gamma * np.abs(d - 2.0)**alpha * np.abs(4.0 - d)**beta
        
        # Fit parameters using curve_fit
        from scipy.optimize import curve_fit
        
        try:
            popt, pcov = curve_fit(
                scaling_func,
                dimensions,
                string_tensions,
                p0=[1.0, 1.0, 1.0],
                bounds=([0, 0, 0], [10, 10, 10])
            )
            
            alpha, beta, gamma = popt
            
            # Compute fit quality
            fit_values = scaling_func(np.array(dimensions), *popt)
            residuals = np.array(string_tensions) - fit_values
            chi_squared = np.sum(residuals**2)
            
            # Formula string
            formula = f"σ(d) = {gamma:.4f} · |d - 2|^{alpha:.4f} · |4 - d|^{beta:.4f}"
            
            print(f"  String tension scaling: {formula}")
            print(f"  Fit quality: χ² = {chi_squared:.4f}")
            
            # Also examine confinement-deconfinement transition
            # In QCD, this happens when σ → 0
            # Based on our formula, this happens when d → 2 or d → 4
            
            # Store scaling law
            scaling_law = {
                'formula': formula,
                'parameters': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma
                },
                'chi_squared': chi_squared,
                'type': 'string_tension'
            }
            
            # Store in results
            self.results['gauge_scaling_law'] = scaling_law
            return scaling_law
            
        except:
            print("  Failed to fit scaling function. May need more data points.")
            return None


if __name__ == "__main__":
    # Test the gauge field integration
    
    # Create a gauge field integration instance
    gauge_integration = GaugeFieldIntegration(
        gauge_group="SU3",  # QCD
        dim_uv=2.0,
        dim_ir=4.0,
        transition_scale=1.0,
        lattice_size=8  # Small for testing
    )
    
    # Compute Wilson loops across energy scales
    wl_results = gauge_integration.compute_multiscale_wilson_loops(
        loop_sizes=[1, 2],
        num_scales=3,
        num_samples=50
    )
    
    # Compute string tension across energy scales
    st_results = gauge_integration.compute_multiscale_string_tension(
        num_scales=3,
        max_loop_size=3,
        num_samples=50
    )
    
    # Derive dimension-dependent gauge scaling
    scaling_law = gauge_integration.derive_dimension_dependent_gauge_scaling()
    
    # Plot results
    gauge_integration.plot_multiscale_wilson_loops(save_path="wilson_loops.png")
    gauge_integration.plot_string_tension(save_path="string_tension.png")
    
    print("\nGauge field integration test complete.") 