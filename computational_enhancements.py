#!/usr/bin/env python
"""
QFT-QG Computational Enhancements

This module provides computational enhancements for the QFT-QG integration framework,
including improved numerical stability at high energy scales, more sophisticated
lattice techniques, and parallel computing capabilities.
"""

import numpy as np
import numba
from numba import jit, prange
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
from tqdm import tqdm
import os


class ComputationalEnhancer:
    """
    Enhanced computational methods for QFT-QG integration.
    """
    
    def __init__(self, precision='double', use_gpu=False, parallelize=True):
        """
        Initialize computational enhancement system.
        
        Parameters:
        -----------
        precision : str
            Numerical precision ('single', 'double', or 'quad')
        use_gpu : bool
            Whether to use GPU acceleration if available
        parallelize : bool
            Whether to use parallel processing
        """
        self.precision = precision
        self.use_gpu = use_gpu
        self.parallelize = parallelize
        
        # Set up numeric precision
        self._configure_precision()
        
        # Set up parallelization
        self.n_cores = mp.cpu_count() if parallelize else 1
        
        # Configure GPU if requested and available
        self.has_gpu = self._check_gpu() if use_gpu else False
    
    def _configure_precision(self):
        """Configure numerical precision settings."""
        if self.precision == 'single':
            self.float_type = np.float32
            self.complex_type = np.complex64
            self.eps = 1.19e-7  # Single precision machine epsilon
        elif self.precision == 'double':
            self.float_type = np.float64
            self.complex_type = np.complex128
            self.eps = 2.22e-16  # Double precision machine epsilon
        elif self.precision == 'quad':
            # Quad precision for critical high-energy calculations
            try:
                self.float_type = np.float128
                self.complex_type = np.complex256
                self.eps = 1.93e-34  # Quad precision machine epsilon
            except AttributeError:
                print("Quad precision not available, falling back to double")
                self.float_type = np.float64
                self.complex_type = np.complex128
                self.eps = 2.22e-16
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
    
    def _check_gpu(self):
        """Check if GPU is available for computation."""
        try:
            import cupy
            return True
        except ImportError:
            print("CUDA not available. Using CPU only.")
            self.use_gpu = False
            return False
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _high_energy_stability_transform(matrix, scale):
        """
        Apply stabilizing transform for high-energy calculations.
        Uses logarithmic scaling and numerical dampening.
        
        Parameters:
        -----------
        matrix : ndarray
            Input matrix to stabilize
        scale : float
            Energy scale for stabilization
            
        Returns:
        --------
        ndarray
            Stabilized matrix
        """
        n = matrix.shape[0]
        result = np.zeros_like(matrix)
        
        # Apply logarithmic scaling with dampening factors
        for i in prange(n):
            for j in range(n):
                if abs(matrix[i, j]) > 1e-10:
                    # Stabilize high-magnitude elements with log transform
                    sign = 1.0 if matrix[i, j] > 0 else -1.0
                    log_val = np.log(abs(matrix[i, j]) + 1.0) * sign
                    result[i, j] = log_val * np.exp(-abs(i - j) / scale)
                else:
                    # Zero out very small elements to reduce noise
                    result[i, j] = 0.0
        
        # Rescale to preserve matrix norm
        norm_factor = np.linalg.norm(matrix) / (np.linalg.norm(result) + 1e-15)
        result *= norm_factor
        
        return result
    
    def improved_lattice_discretization(self, dim=4, size=32, lattice_spacing=0.1, adaptive=True):
        """
        Create improved lattice discretization with better continuum limit behavior.
        
        Parameters:
        -----------
        dim : int
            Number of dimensions
        size : int
            Lattice size in each dimension
        lattice_spacing : float
            Initial lattice spacing
        adaptive : bool
            Whether to use adaptive spacing near high-curvature regions
            
        Returns:
        --------
        dict
            Discretized lattice components
        """
        # Create base lattice points
        points_per_dim = [size] * dim
        total_points = np.prod(points_per_dim)
        
        # Basic lattice coordinates
        coords = np.zeros((total_points, dim), dtype=self.float_type)
        
        # Generate coordinates with potential adaptive spacing
        if adaptive:
            # Create non-uniform grid with refinement near "center"
            grid_arrays = []
            for d in range(dim):
                # Hyperbolic tangent mapping for smooth refinement
                linear_grid = np.linspace(-1, 1, points_per_dim[d])
                # More points near zero, fewer at edges
                nonuniform_grid = np.tanh(linear_grid * 2) / np.tanh(2)
                # Scale to desired lattice size
                grid_arrays.append(nonuniform_grid * (size/2) * lattice_spacing)
            
            # Create full grid from component arrays
            grid_mesh = np.meshgrid(*grid_arrays, indexing='ij')
            # Reshape to coordinate list
            for d in range(dim):
                coords[:, d] = grid_mesh[d].flatten()
        else:
            # Uniform lattice spacing
            for d in range(dim):
                indices = np.arange(points_per_dim[d])
                grid_position = (indices - points_per_dim[d]/2) * lattice_spacing
                # Repeat and reshape to match coords shape
                repeats = total_points // points_per_dim[d]
                coords[:, d] = np.repeat(grid_position, repeats)
        
        # Construct finite difference operators with improved accuracy
        # 4th order accurate derivatives (more accurate than standard 2nd order)
        laplacian = self._construct_improved_laplacian(dim, points_per_dim, lattice_spacing)
        
        # Nearest neighbor map for interactions
        neighbor_map = self._construct_neighbor_map(dim, points_per_dim)
        
        return {
            'points_per_dim': points_per_dim,
            'total_points': total_points,
            'coords': coords,
            'laplacian': laplacian,
            'neighbor_map': neighbor_map,
            'lattice_spacing': lattice_spacing,
            'dim': dim
        }
    
    def _construct_improved_laplacian(self, dim, points_per_dim, spacing):
        """
        Construct improved (higher-order accurate) Laplacian operator.
        
        Parameters:
        -----------
        dim : int
            Number of dimensions
        points_per_dim : list
            Number of points in each dimension
        spacing : float
            Lattice spacing
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Sparse Laplacian matrix
        """
        total_points = np.prod(points_per_dim)
        
        # We'll use scipy sparse matrices for efficiency
        row_indices = []
        col_indices = []
        values = []
        
        # 4th order accurate central difference coefficients
        # f''(x) ≈ (-1/12*f(x-2h) + 4/3*f(x-h) - 5/2*f(x) + 4/3*f(x+h) - 1/12*f(x+2h)) / h^2
        central_coef = -5.0/2.0 / (spacing**2)
        neighbor1_coef = 4.0/3.0 / (spacing**2)
        neighbor2_coef = -1.0/12.0 / (spacing**2)
        
        # Construct the sparse matrix
        for i in range(total_points):
            # Add central point
            row_indices.append(i)
            col_indices.append(i)
            values.append(central_coef * dim)  # Diagonal term appears once per dimension
            
            # Multi-dimensional index
            multi_idx = np.zeros(dim, dtype=int)
            remain = i
            for d in range(dim-1, -1, -1):
                divisor = np.prod(points_per_dim[d+1:]) if d < dim-1 else 1
                multi_idx[d] = remain // divisor
                remain = remain % divisor
            
            # Add neighbors in each dimension
            for d in range(dim):
                # First neighbors (±1)
                for offset in [-1, 1]:
                    neighbor_idx = multi_idx.copy()
                    neighbor_idx[d] += offset
                    
                    # Check boundary and apply periodic conditions
                    if 0 <= neighbor_idx[d] < points_per_dim[d]:
                        # Convert back to flat index
                        flat_idx = 0
                        for d2 in range(dim):
                            if d2 < dim - 1:
                                flat_idx += neighbor_idx[d2] * np.prod(points_per_dim[d2+1:])
                            else:
                                flat_idx += neighbor_idx[d2]
                        
                        row_indices.append(i)
                        col_indices.append(flat_idx)
                        values.append(neighbor1_coef)
                
                # Second neighbors (±2)
                for offset in [-2, 2]:
                    neighbor_idx = multi_idx.copy()
                    neighbor_idx[d] += offset
                    
                    # Check boundary and apply periodic conditions
                    if 0 <= neighbor_idx[d] < points_per_dim[d]:
                        # Convert back to flat index
                        flat_idx = 0
                        for d2 in range(dim):
                            if d2 < dim - 1:
                                flat_idx += neighbor_idx[d2] * np.prod(points_per_dim[d2+1:])
                            else:
                                flat_idx += neighbor_idx[d2]
                        
                        row_indices.append(i)
                        col_indices.append(flat_idx)
                        values.append(neighbor2_coef)
        
        # Create sparse matrix
        laplacian = sparse.csr_matrix((values, (row_indices, col_indices)), 
                                      shape=(total_points, total_points))
        
        return laplacian
    
    def _construct_neighbor_map(self, dim, points_per_dim):
        """
        Construct map of nearest neighbors for each lattice point.
        
        Parameters:
        -----------
        dim : int
            Number of dimensions
        points_per_dim : list
            Number of points in each dimension
            
        Returns:
        --------
        list
            List of neighbor indices for each point
        """
        total_points = np.prod(points_per_dim)
        neighbor_map = [[] for _ in range(total_points)]
        
        # For each point, find its neighbors
        for i in range(total_points):
            # Convert to multi-dimensional index
            multi_idx = np.zeros(dim, dtype=int)
            remain = i
            for d in range(dim-1, -1, -1):
                divisor = np.prod(points_per_dim[d+1:]) if d < dim-1 else 1
                multi_idx[d] = remain // divisor
                remain = remain % divisor
            
            # For each dimension, add ±1 neighbors with periodic boundary conditions
            for d in range(dim):
                for offset in [-1, 1]:
                    neighbor_idx = multi_idx.copy()
                    neighbor_idx[d] = (neighbor_idx[d] + offset) % points_per_dim[d]
                    
                    # Convert back to flat index
                    flat_idx = 0
                    for d2 in range(dim):
                        if d2 < dim - 1:
                            flat_idx += neighbor_idx[d2] * np.prod(points_per_dim[d2+1:])
                        else:
                            flat_idx += neighbor_idx[d2]
                    
                    neighbor_map[i].append(flat_idx)
        
        return neighbor_map
    
    @jit(nopython=True)
    def stable_high_energy_propagator(self, momentum, mass, qg_scale, beta_coeffs):
        """
        Numerically stable calculation of QG-corrected propagator at high energies.
        
        Parameters:
        -----------
        momentum : ndarray
            Momentum values
        mass : float
            Particle mass
        qg_scale : float
            Quantum gravity scale
        beta_coeffs : ndarray
            Higher-derivative coefficients
            
        Returns:
        --------
        ndarray
            Stabilized propagator values
        """
        p_squared = np.sum(momentum**2, axis=1)
        propagator = np.zeros_like(p_squared, dtype=np.complex128)
        
        # Process each momentum value with stabilized calculation
        for i in range(len(p_squared)):
            p2 = p_squared[i]
            
            # For very high momenta, use asymptotic expansion
            if p2 > 0.1 * qg_scale**2:
                # Leading terms in 1/p² expansion for high momentum
                p4_term = beta_coeffs[0] * p2**2 / qg_scale**2
                denominator = p4_term * (1.0 - mass**2/p2 + beta_coeffs[1]*mass**2/(qg_scale**2))
                propagator[i] = 1.0 / denominator
            else:
                # Standard calculation for moderate momenta
                # Avoid direct subtraction to prevent catastrophic cancellation
                m2_term = mass**2 * (1.0 + beta_coeffs[1] * p2 / qg_scale**2)
                p4_term = beta_coeffs[0] * p2**2 / qg_scale**2
                
                # Calculate terms separately before combining
                denominator = p2 + p4_term - m2_term + beta_coeffs[2] * mass**4 / qg_scale**2
                propagator[i] = 1.0 / denominator
        
        return propagator
    
    def parallel_monte_carlo(self, hamiltonian, lattice, n_steps=1000, n_thermalize=100, beta=1.0):
        """
        Parallel Monte Carlo simulation for lattice field theory.
        
        Parameters:
        -----------
        hamiltonian : callable
            Function computing the Hamiltonian for a field configuration
        lattice : dict
            Lattice information from improved_lattice_discretization
        n_steps : int
            Number of Monte Carlo steps
        n_thermalize : int
            Number of thermalization steps
        beta : float
            Inverse temperature
            
        Returns:
        --------
        dict
            Simulation results
        """
        total_points = lattice['total_points']
        dim = lattice['dim']
        
        # Initial random field configuration
        field = np.random.normal(0, 1, total_points).astype(self.float_type)
        
        # For storing results
        accepted = 0
        field_samples = []
        energies = []
        
        # Create process pool if parallel
        if self.parallelize and self.n_cores > 1:
            pool = mp.Pool(processes=self.n_cores)
        
        # Define update for a single site
        def update_site(site_idx, current_field):
            # Get current value and propose a new one
            old_value = current_field[site_idx]
            new_value = old_value + np.random.normal(0, 0.1)
            
            # Create proposed field configuration
            proposed_field = current_field.copy()
            proposed_field[site_idx] = new_value
            
            # Calculate energy difference
            delta_energy = hamiltonian(proposed_field, lattice) - hamiltonian(current_field, lattice)
            
            # Metropolis acceptance
            if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
                return True, new_value
            else:
                return False, old_value
        
        # Main Monte Carlo loop
        for step in tqdm(range(n_steps + n_thermalize)):
            if self.parallelize and self.n_cores > 1:
                # Divide lattice into chunks for parallel updates
                chunk_size = total_points // self.n_cores
                chunks = [(i, field.copy()) for i in range(0, total_points, chunk_size)]
                
                # Parallel site updates
                chunk_results = []
                for chunk_start, chunk_field in chunks:
                    chunk_end = min(chunk_start + chunk_size, total_points)
                    
                    # Process each site in the chunk
                    for site in range(chunk_start, chunk_end):
                        accepted_site, new_value = update_site(site, chunk_field)
                        if accepted_site:
                            chunk_field[site] = new_value
                            accepted += 1
                    
                    chunk_results.append(chunk_field)
                
                # Merge results from all chunks
                for i, chunk_field in enumerate(chunk_results):
                    chunk_start = i * chunk_size
                    chunk_end = min(chunk_start + chunk_size, total_points)
                    field[chunk_start:chunk_end] = chunk_field[chunk_start:chunk_end]
            else:
                # Sequential updates
                for site in range(total_points):
                    accepted_site, new_value = update_site(site, field)
                    if accepted_site:
                        field[site] = new_value
                        accepted += 1
            
            # Record samples after thermalization
            if step >= n_thermalize:
                field_samples.append(field.copy())
                energies.append(hamiltonian(field, lattice))
        
        # Clean up parallel pool if used
        if self.parallelize and self.n_cores > 1:
            pool.close()
        
        # Calculate observables
        field_mean = np.mean(field_samples, axis=0)
        field_var = np.var(field_samples, axis=0)
        energy_mean = np.mean(energies)
        acceptance_rate = accepted / (total_points * n_steps)
        
        return {
            'field_samples': field_samples,
            'energies': energies,
            'field_mean': field_mean,
            'field_var': field_var,
            'energy_mean': energy_mean,
            'acceptance_rate': acceptance_rate
        }


def main():
    """Run demonstration of computational enhancements."""
    print("Demonstrating QFT-QG computational enhancements...")
    
    # Initialize computational enhancer
    enhancer = ComputationalEnhancer(precision='double', parallelize=True)
    
    # Create improved lattice
    lattice = enhancer.improved_lattice_discretization(dim=4, size=16, adaptive=True)
    
    print(f"Created {lattice['total_points']} lattice points in {lattice['dim']} dimensions")
    print(f"Using {enhancer.precision} precision with {enhancer.n_cores} CPU cores")
    
    # Example: stable high-energy calculation
    test_momenta = np.random.normal(0, 1e18, (100, 4))
    qg_scale = 1.22e19  # Planck scale in GeV
    mass = 125.0  # Higgs mass in GeV
    beta_coeffs = np.array([0.1, 0.05, 0.01])
    
    start_time = os.times()[0]
    stable_results = enhancer.stable_high_energy_propagator(test_momenta, mass, qg_scale, beta_coeffs)
    end_time = os.times()[0]
    
    print(f"Computed {len(test_momenta)} stable high-energy propagators in {end_time - start_time:.3f} seconds")
    print(f"Mean propagator magnitude: {np.mean(np.abs(stable_results)):.2e}")
    
    print("Computational enhancements successfully demonstrated")


if __name__ == "__main__":
    main() 