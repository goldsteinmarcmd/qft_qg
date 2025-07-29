#!/usr/bin/env python
"""
GPU Acceleration for Quantum Gravity Framework

This module provides GPU acceleration for lattice field theory simulations
using JAX, building on the existing framework for faster parameter exploration.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random
from jax.lax import scan
from typing import Dict, Tuple, Optional, List, Callable
import warnings

# Import existing framework components
from qft.lattice_field_theory import LatticeScalarField
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class GPUAcceleratedLattice:
    """
    GPU-accelerated lattice field theory using JAX.
    
    This class provides GPU acceleration for the existing lattice field theory
    framework, enabling faster Monte Carlo simulations and parameter exploration.
    """
    
    def __init__(self, lattice_shape: Tuple[int, ...], 
                 mass_squared: float = 0.1,
                 coupling: float = 0.1,
                 dimension: int = 4,
                 use_gpu: bool = True):
        """
        Initialize GPU-accelerated lattice simulation.
        
        Parameters:
        -----------
        lattice_shape : tuple
            Shape of the lattice (e.g., (32, 32) for 2D)
        mass_squared : float
            Mass squared parameter
        coupling : float
            Coupling constant
        dimension : int
            Spacetime dimension
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.lattice_shape = lattice_shape
        self.mass_squared = mass_squared
        self.coupling = coupling
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Configure JAX for GPU if requested
        if use_gpu:
            try:
                jax.config.update('jax_platform_name', 'gpu')
                print("GPU acceleration enabled")
            except Exception as e:
                warnings.warn(f"GPU acceleration failed: {e}. Falling back to CPU.")
                jax.config.update('jax_platform_name', 'cpu')
        
        # Initialize random key for JAX
        self.key = random.PRNGKey(42)
        
        # Pre-compile JIT functions
        self._compile_jit_functions()
    
    def _compile_jit_functions(self) -> None:
        """Pre-compile JIT functions for better performance."""
        print("Compiling JIT functions...")
        
        # Compile action function
        self._action_jit = jit(self._compute_action)
        
        # Compile Metropolis update
        self._metropolis_update_jit = jit(self._metropolis_update)
        
        # Compile measurement functions
        self._measure_observables_jit = jit(self._measure_observables)
        
        print("JIT compilation complete")
    
    def _compute_action(self, field: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the action for a given field configuration.
        
        Parameters:
        -----------
        field : jnp.ndarray
            Field configuration
            
        Returns:
        --------
        jnp.ndarray
            Action value
        """
        # Kinetic term (discrete Laplacian)
        kinetic_term = 0.0
        
        # Sum over all dimensions
        for axis in range(len(field.shape)):
            # Forward difference
            forward_diff = jnp.roll(field, -1, axis=axis) - field
            # Backward difference  
            backward_diff = field - jnp.roll(field, 1, axis=axis)
            # Kinetic term for this dimension
            kinetic_term += jnp.sum(forward_diff * backward_diff)
        
        # Mass term
        mass_term = self.mass_squared * jnp.sum(field**2)
        
        # Interaction term (φ⁴ theory)
        interaction_term = self.coupling * jnp.sum(field**4)
        
        # Total action
        action = 0.5 * kinetic_term + mass_term + interaction_term
        
        return action
    
    def _metropolis_update(self, field: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform one Metropolis-Hastings update step.
        
        Parameters:
        -----------
        field : jnp.ndarray
            Current field configuration
        key : jnp.ndarray
            Random key for JAX
            
        Returns:
        --------
        tuple
            (new_field, new_key)
        """
        # Generate proposal
        proposal_key, accept_key, new_key = random.split(key, 3)
        proposal = field + 0.1 * random.normal(proposal_key, field.shape)
        
        # Compute action difference
        action_old = self._action_jit(field)
        action_new = self._action_jit(proposal)
        delta_action = action_new - action_old
        
        # Metropolis acceptance probability
        acceptance_prob = jnp.minimum(1.0, jnp.exp(-delta_action))
        
        # Accept or reject
        accept = random.uniform(accept_key) < acceptance_prob
        
        # Return new field (either proposal or old)
        new_field = jnp.where(accept, proposal, field)
        
        return new_field, new_key
    
    def _measure_observables(self, field: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Measure observables for a given field configuration.
        
        Parameters:
        -----------
        field : jnp.ndarray
            Field configuration
            
        Returns:
        --------
        dict
            Dictionary of observables
        """
        # Field average
        field_avg = jnp.mean(field)
        
        # Field variance
        field_var = jnp.var(field)
        
        # Susceptibility (connected two-point function)
        susceptibility = jnp.var(field)
        
        # Binder cumulant
        field_4 = jnp.mean(field**4)
        field_2 = jnp.mean(field**2)
        binder_cumulant = 1.0 - field_4 / (3.0 * field_2**2)
        
        # Energy density
        energy_density = self._action_jit(field) / jnp.prod(jnp.array(field.shape))
        
        return {
            'field_average': field_avg,
            'field_variance': field_var,
            'susceptibility': susceptibility,
            'binder_cumulant': binder_cumulant,
            'energy_density': energy_density
        }
    
    def run_gpu_simulation(self, 
                          num_thermalization: int = 1000,
                          num_configurations: int = 1000,
                          measurement_interval: int = 10) -> Dict:
        """
        Run GPU-accelerated Monte Carlo simulation.
        
        Parameters:
        -----------
        num_thermalization : int
            Number of thermalization steps
        num_configurations : int
            Number of configurations to generate
        measurement_interval : int
            Interval between measurements
            
        Returns:
        --------
        dict
            Simulation results
        """
        print(f"Running GPU-accelerated simulation on {self.lattice_shape} lattice...")
        print(f"Thermalization steps: {num_thermalization}")
        print(f"Configuration count: {num_configurations}")
        
        # Initialize field configuration
        init_key = random.PRNGKey(42)
        field = random.normal(init_key, self.lattice_shape)
        
        # Thermalization phase
        print("Thermalizing...")
        thermalization_keys = random.split(init_key, num_thermalization)
        
        def thermalization_step(carry, key):
            field, _ = carry
            new_field, new_key = self._metropolis_update_jit(field, key)
            return (new_field, new_key), None
        
        # Run thermalization
        (field, _), _ = scan(thermalization_step, (field, init_key), thermalization_keys)
        
        # Measurement phase
        print("Taking measurements...")
        measurement_keys = random.split(random.PRNGKey(123), num_configurations)
        
        def measurement_step(carry, key):
            field, measurements = carry
            
            # Update field
            new_field, new_key = self._metropolis_update_jit(field, key)
            
            # Measure observables
            obs = self._measure_observables_jit(new_field)
            
            # Accumulate measurements
            new_measurements = {
                k: measurements[k] + [v] for k, v in obs.items()
            }
            
            return (new_field, new_measurements), None
        
        # Initialize measurements
        initial_measurements = {
            'field_average': [],
            'field_variance': [],
            'susceptibility': [],
            'binder_cumulant': [],
            'energy_density': []
        }
        
        # Run measurements
        (final_field, measurements), _ = scan(
            measurement_step, 
            (field, initial_measurements), 
            measurement_keys
        )
        
        # Convert to numpy arrays
        results = {}
        for key, values in measurements.items():
            results[key] = np.array(values)
        
        # Compute statistics
        statistics = {}
        for key, values in results.items():
            statistics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        print("Simulation complete!")
        
        return {
            'measurements': results,
            'statistics': statistics,
            'final_field': np.array(final_field),
            'parameters': {
                'lattice_shape': self.lattice_shape,
                'mass_squared': self.mass_squared,
                'coupling': self.coupling,
                'dimension': self.dimension
            }
        }
    
    def accelerated_parameter_scan(self, 
                                 mass_range: Tuple[float, float] = (0.01, 1.0),
                                 coupling_range: Tuple[float, float] = (0.01, 1.0),
                                 num_points: int = 20) -> Dict:
        """
        Perform accelerated parameter scan using GPU.
        
        Parameters:
        -----------
        mass_range : tuple
            Range of mass squared values
        coupling_range : tuple
            Range of coupling values
        num_points : int
            Number of points in each parameter direction
            
        Returns:
        --------
        dict
            Parameter scan results
        """
        print(f"Performing GPU-accelerated parameter scan...")
        print(f"Mass range: {mass_range}")
        print(f"Coupling range: {coupling_range}")
        print(f"Grid size: {num_points}x{num_points}")
        
        # Generate parameter grid
        mass_values = np.linspace(mass_range[0], mass_range[1], num_points)
        coupling_values = np.linspace(coupling_range[0], coupling_range[1], num_points)
        
        # Results storage
        scan_results = {
            'mass_values': mass_values,
            'coupling_values': coupling_values,
            'susceptibility': np.zeros((num_points, num_points)),
            'binder_cumulant': np.zeros((num_points, num_points)),
            'energy_density': np.zeros((num_points, num_points))
        }
        
        # Run simulations for each parameter combination
        for i, mass in enumerate(mass_values):
            for j, coupling in enumerate(coupling_values):
                print(f"Progress: {i*num_points + j + 1}/{num_points**2}")
                
                # Create simulation with these parameters
                sim = GPUAcceleratedLattice(
                    lattice_shape=self.lattice_shape,
                    mass_squared=mass,
                    coupling=coupling,
                    dimension=self.dimension,
                    use_gpu=self.use_gpu
                )
                
                # Run simulation
                results = sim.run_gpu_simulation(
                    num_thermalization=500,  # Shorter for parameter scan
                    num_configurations=500
                )
                
                # Store results
                scan_results['susceptibility'][i, j] = results['statistics']['susceptibility']['mean']
                scan_results['binder_cumulant'][i, j] = results['statistics']['binder_cumulant']['mean']
                scan_results['energy_density'][i, j] = results['statistics']['energy_density']['mean']
        
        return scan_results
    
    def compare_with_cpu(self, num_configurations: int = 1000) -> Dict:
        """
        Compare GPU vs CPU performance.
        
        Parameters:
        -----------
        num_configurations : int
            Number of configurations for comparison
            
        Returns:
        --------
        dict
            Performance comparison results
        """
        print("Comparing GPU vs CPU performance...")
        
        import time
        
        # GPU simulation
        print("Running GPU simulation...")
        gpu_start = time.time()
        gpu_results = self.run_gpu_simulation(num_configurations=num_configurations)
        gpu_time = time.time() - gpu_start
        
        # CPU simulation (using original framework)
        print("Running CPU simulation...")
        cpu_start = time.time()
        
        # Create CPU simulation using existing framework
        cpu_sim = LatticeScalarField(
            lattice_shape=self.lattice_shape,
            mass_squared=self.mass_squared,
            coupling=self.coupling,
            dimension=self.dimension
        )
        
        cpu_results = cpu_sim.run_simulation(
            num_thermalization=1000,
            num_configurations=num_configurations
        )
        cpu_time = time.time() - cpu_start
        
        # Performance comparison
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        return {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': speedup,
            'gpu_results': gpu_results,
            'cpu_results': cpu_results
        }


class GPUAcceleratedQGRG:
    """
    GPU-accelerated quantum gravity renormalization group.
    
    This class provides GPU acceleration for RG flow calculations
    in the quantum gravity framework.
    """
    
    def __init__(self, dim_uv: float = 2.0, dim_ir: float = 4.0, 
                 transition_scale: float = 1.0):
        """
        Initialize GPU-accelerated QG RG.
        
        Parameters:
        -----------
        dim_uv : float
            UV spectral dimension
        dim_ir : float
            IR spectral dimension
        transition_scale : float
            Transition scale in Planck units
        """
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        
        # Initialize JAX functions
        self._compile_rg_functions()
    
    def _compile_rg_functions(self) -> None:
        """Compile JAX functions for RG calculations."""
        print("Compiling RG functions...")
        
        # Compile beta function calculation
        self._beta_function_jit = jit(self._compute_beta_functions)
        
        # Compile RG flow integration
        self._rg_flow_jit = jit(self._integrate_rg_flow)
        
        print("RG compilation complete")
    
    def _compute_beta_functions(self, scale: jnp.ndarray, couplings: jnp.ndarray, 
                               dimension: jnp.ndarray) -> jnp.ndarray:
        """
        Compute beta functions with dimensional flow.
        
        Parameters:
        -----------
        scale : jnp.ndarray
            Energy scale
        couplings : jnp.ndarray
            Coupling values
        dimension : jnp.ndarray
            Spectral dimension
            
        Returns:
        --------
        jnp.ndarray
            Beta function values
        """
        # Simplified beta functions with dimensional flow
        # In practice, these would be more sophisticated
        
        # Beta function for coupling g
        beta_g = couplings[0] * (dimension - 4.0) * scale
        
        # Beta function for coupling lambda
        beta_lambda = couplings[1] * (dimension - 4.0) * scale**2
        
        return jnp.array([beta_g, beta_lambda])
    
    def _integrate_rg_flow(self, initial_couplings: jnp.ndarray, 
                           scales: jnp.ndarray) -> jnp.ndarray:
        """
        Integrate RG flow equations.
        
        Parameters:
        -----------
        initial_couplings : jnp.ndarray
            Initial coupling values
        scales : jnp.ndarray
            Energy scales for integration
            
        Returns:
        --------
        jnp.ndarray
            Coupling trajectories
        """
        def rg_step(couplings, scale):
            # Compute dimension at this scale
            dimension = self.dim_ir + (self.dim_uv - self.dim_ir) / (1.0 + (scale / self.transition_scale)**2)
            
            # Compute beta functions
            betas = self._beta_function_jit(scale, couplings, dimension)
            
            # Euler integration step
            new_couplings = couplings + betas * 0.01  # Small step size
            
            return new_couplings, new_couplings
        
        # Integrate RG flow
        final_couplings, trajectories = scan(rg_step, initial_couplings, scales)
        
        return trajectories
    
    def accelerated_rg_flow(self, scale_range: Tuple[float, float] = (1e-6, 1e3),
                           num_points: int = 1000) -> Dict:
        """
        Compute accelerated RG flow.
        
        Parameters:
        -----------
        scale_range : tuple
            Range of energy scales
        num_points : int
            Number of integration points
            
        Returns:
        --------
        dict
            RG flow results
        """
        print(f"Computing GPU-accelerated RG flow...")
        
        # Generate scale points
        scales = jnp.logspace(jnp.log10(scale_range[0]), jnp.log10(scale_range[1]), num_points)
        
        # Initial couplings
        initial_couplings = jnp.array([0.1, 0.2])  # g, lambda
        
        # Integrate RG flow
        trajectories = self._rg_flow_jit(initial_couplings, scales)
        
        # Convert to numpy for analysis
        scales_np = np.array(scales)
        trajectories_np = np.array(trajectories)
        
        return {
            'scales': scales_np,
            'couplings': trajectories_np,
            'dimensions': np.array([
                self.dim_ir + (self.dim_uv - self.dim_ir) / (1.0 + (s / self.transition_scale)**2)
                for s in scales_np
            ])
        }


def main():
    """Demonstrate GPU acceleration capabilities."""
    print("GPU Acceleration for Quantum Gravity Framework")
    print("=" * 55)
    
    # Test GPU-accelerated lattice simulation
    print("\n1. GPU-Accelerated Lattice Simulation")
    gpu_lattice = GPUAcceleratedLattice(
        lattice_shape=(32, 32),
        mass_squared=0.1,
        coupling=0.1,
        dimension=2,
        use_gpu=True
    )
    
    results = gpu_lattice.run_gpu_simulation(
        num_thermalization=1000,
        num_configurations=1000
    )
    
    print(f"   Susceptibility: {results['statistics']['susceptibility']['mean']:.6f} ± {results['statistics']['susceptibility']['std']:.6f}")
    print(f"   Binder cumulant: {results['statistics']['binder_cumulant']['mean']:.6f} ± {results['statistics']['binder_cumulant']['std']:.6f}")
    
    # Test performance comparison
    print("\n2. Performance Comparison")
    comparison = gpu_lattice.compare_with_cpu(num_configurations=500)
    print(f"   GPU time: {comparison['gpu_time']:.2f} seconds")
    print(f"   CPU time: {comparison['cpu_time']:.2f} seconds")
    print(f"   Speedup: {comparison['speedup']:.1f}x")
    
    # Test GPU-accelerated RG flow
    print("\n3. GPU-Accelerated RG Flow")
    gpu_rg = GPUAcceleratedQGRG(
        dim_uv=2.0,
        dim_ir=4.0,
        transition_scale=1.0
    )
    
    rg_results = gpu_rg.accelerated_rg_flow()
    print(f"   RG flow computed for {len(rg_results['scales'])} scale points")
    print(f"   Final couplings: {rg_results['couplings'][-1]}")


if __name__ == "__main__":
    main() 