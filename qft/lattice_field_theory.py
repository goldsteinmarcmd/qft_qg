"""Lattice Field Theory module.

This module implements lattice formulations of quantum field theories,
including scalar field theories and methods for Markov Chain Monte Carlo
simulations on the lattice.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional, Union
from dataclasses import dataclass
import random

class LatticeScalarField:
    """Lattice implementation of scalar field theory.
    
    This class provides methods for simulating scalar field theories on a
    lattice, including phi^4 theory and other theories with polynomial potentials.
    """
    
    def __init__(self, lattice_size, mass_squared=1.0, coupling=0.1, dimension=4, 
                 boundary="periodic"):
        """Initialize the lattice scalar field.
        
        Args:
            lattice_size: Tuple of lattice dimensions (L_1, L_2, ..., L_d)
            mass_squared: Mass squared parameter (can be negative for symmetry breaking)
            coupling: Self-interaction coupling constant
            dimension: Spacetime dimension
            boundary: Boundary conditions ("periodic" or "open")
        """
        self.lattice_size = lattice_size
        self.mass_squared = mass_squared
        self.coupling = coupling
        self.dimension = dimension
        self.boundary = boundary
        
        # Initialize field configuration to random values
        self.field = np.random.normal(0, 0.1, size=lattice_size)
        
        # Calculate lattice spacing (set to 1 by default)
        self.a = 1.0
        
        # Store action components for analysis
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
    
    def action(self, field=None):
        """Calculate the action for the field configuration.
        
        For phi^4 theory, the action is:
        S = Σ_x [1/2 * (∂_μφ)^2 + m^2/2 * φ^2 + λ/4! * φ^4]
        
        Args:
            field: Field configuration (uses self.field if None)
            
        Returns:
            Total action value
        """
        if field is None:
            field = self.field
        
        # Calculate kinetic term (nearest neighbor interactions)
        kinetic = 0.0
        for mu in range(self.dimension):
            # Construct shifted fields for finite differences
            shifted = np.roll(field, -1, axis=mu)
            
            # Calculate (φ(x+μ) - φ(x))^2 for each site
            # Factor of 1/a^2 for lattice spacing is absorbed into the definition of parameters
            kinetic += np.sum((shifted - field)**2)
        
        self.kinetic_energy = kinetic / 2.0
        
        # Calculate mass term and self-interaction
        mass_term = self.mass_squared * np.sum(field**2) / 2.0
        interaction = self.coupling * np.sum(field**4) / 24.0  # λ/4! * φ^4
        
        self.potential_energy = mass_term + interaction
        
        # Total action
        return self.kinetic_energy + self.potential_energy
    
    def monte_carlo_step(self, delta=1.0, beta=1.0):
        """Perform a Monte Carlo update using the Metropolis algorithm.
        
        Args:
            delta: Maximum field update amplitude
            beta: Inverse temperature (1/g^2 for coupling g)
            
        Returns:
            Boolean indicating if the update was accepted
        """
        # Select a random lattice site
        indices = tuple(random.randint(0, size-1) for size in self.lattice_size)
        
        # Store old field value and calculate old action
        old_value = self.field[indices]
        old_action = self.action()
        
        # Propose new field value
        new_value = old_value + delta * (2 * random.random() - 1)
        self.field[indices] = new_value
        
        # Calculate new action
        new_action = self.action()
        
        # Metropolis acceptance
        delta_s = new_action - old_action
        if delta_s <= 0 or random.random() < np.exp(-beta * delta_s):
            return True
        else:
            # Revert the change
            self.field[indices] = old_value
            return False
    
    def run_simulation(self, num_thermalization=1000, num_configurations=5000, 
                       measurements_interval=10, delta=1.0, beta=1.0):
        """Run a Monte Carlo simulation and collect measurements.
        
        Args:
            num_thermalization: Number of thermalization steps
            num_configurations: Number of configurations to generate
            measurements_interval: Interval between measurements
            delta: Maximum field update amplitude
            beta: Inverse temperature
            
        Returns:
            Dictionary containing measurements
        """
        # Thermalization
        print("Thermalizing...")
        acceptances = 0
        for i in range(num_thermalization):
            acceptances += int(self.monte_carlo_step(delta, beta))
            
            # Adjust delta to get ~50% acceptance
            if (i+1) % 100 == 0:
                acceptance_rate = acceptances / 100
                if acceptance_rate < 0.4:
                    delta *= 0.9
                elif acceptance_rate > 0.6:
                    delta *= 1.1
                acceptances = 0
        
        # Measurement phase
        print("Collecting measurements...")
        
        # Initialize measurement arrays
        actions = []
        field_values = []
        field_squared = []
        field_fourth = []
        susceptibility_samples = []
        
        # Run production steps
        for i in range(num_configurations * measurements_interval):
            self.monte_carlo_step(delta, beta)
            
            # Take measurements every interval
            if (i+1) % measurements_interval == 0:
                # Action
                action_value = self.action()
                actions.append(action_value)
                
                # Field expectation values
                field_mean = np.mean(self.field)
                field_values.append(field_mean)
                
                # Field squared (for susceptibility)
                field_sq = np.mean(self.field**2)
                field_squared.append(field_sq)
                
                # Field fourth power (for Binder cumulant)
                field_4 = np.mean(self.field**4)
                field_fourth.append(field_4)
                
                # Store entire field configuration periodically for susceptibility
                if (i+1) % (10 * measurements_interval) == 0:
                    susceptibility_samples.append(self.field.copy())
        
        # Calculate observables from measurements
        results = {
            "action_mean": np.mean(actions),
            "action_error": np.std(actions) / np.sqrt(len(actions)),
            "field_mean": np.mean(field_values),
            "field_abs_mean": np.mean(np.abs(field_values)),
            "field_squared_mean": np.mean(field_squared),
            "field_fourth_mean": np.mean(field_fourth),
            "action_history": actions,
            "field_history": field_values,
            "field_squared_history": field_squared,
            "susceptibility": np.var(field_values) * np.prod(self.lattice_size),
            "binder_cumulant": 1 - np.mean(field_fourth) / (3 * np.mean(field_squared)**2)
        }
        
        return results
    
    def calculate_correlation_function(self, num_samples=100, thermalization=500, beta=1.0):
        """Calculate the two-point correlation function G(r) = <φ(0)φ(r)>.
        
        Args:
            num_samples: Number of configurations to average over
            thermalization: Number of steps between samples
            beta: Inverse temperature
            
        Returns:
            Tuple of (distances, correlation values)
        """
        if self.dimension != 4:
            # For simplicity, assume 4D lattice, use 1D slice
            max_distance = min(self.lattice_size) // 2
            distances = np.arange(max_distance)
            correlations = np.zeros(max_distance)
            
            # Reference point (center of lattice)
            center = tuple(size // 2 for size in self.lattice_size)
            
            # Run simulation and measure
            for _ in range(num_samples):
                # Thermalize between measurements
                for _ in range(thermalization):
                    self.monte_carlo_step(beta=beta)
                
                # Measure along x-axis
                for r in range(max_distance):
                    point = list(center)
                    point[0] = (center[0] + r) % self.lattice_size[0]
                    correlations[r] += self.field[center] * self.field[tuple(point)]
            
            # Normalize
            correlations /= num_samples
            
            return distances, correlations
        
        # For 4D lattice, calculate correlation function along all axes
        max_distance = min(self.lattice_size) // 2
        distances = np.arange(max_distance)
        correlations = np.zeros(max_distance)
        counts = np.zeros(max_distance)
        
        # Run simulation and measure
        for sample in range(num_samples):
            # Thermalize between measurements
            for _ in range(thermalization):
                self.monte_carlo_step(beta=beta)
            
            # Use the center as the reference point
            center = tuple(size // 2 for size in self.lattice_size)
            phi_center = self.field[center]
            
            # Calculate correlations for all points
            for idx in np.ndindex(self.lattice_size):
                # Calculate Manhattan distance to center
                dist = sum((idx[i] - center[i]) % self.lattice_size[i] for i in range(self.dimension))
                
                if dist < max_distance:
                    correlations[dist] += phi_center * self.field[idx]
                    counts[dist] += 1
        
        # Normalize by counts
        correlations /= counts
        correlations /= num_samples
        
        return distances, correlations

    def calculate_mass_gap(self, correlation_function=None):
        """Calculate the mass gap from the exponential decay of the correlation function.
        
        G(r) ~ exp(-m*r) for large r, where m is the physical mass.
        
        Args:
            correlation_function: Tuple of (distances, correlations), 
                                 or None to calculate
            
        Returns:
            Extracted mass value and error estimate
        """
        if correlation_function is None:
            distances, correlations = self.calculate_correlation_function()
        else:
            distances, correlations = correlation_function
        
        # Use only the tail of the correlation function
        start_idx = len(distances) // 3
        distances = distances[start_idx:]
        correlations = correlations[start_idx:]
        
        # Ensure positive correlations for log
        if np.any(correlations <= 0):
            valid = correlations > 0
            distances = distances[valid]
            correlations = correlations[valid]
        
        if len(distances) < 3:
            return None, None
        
        # Fit log(G(r)) = -m*r + c
        log_corr = np.log(correlations)
        A = np.vstack([distances, np.ones(len(distances))]).T
        m, c = np.linalg.lstsq(A, log_corr, rcond=None)[0]
        
        # Error estimation using bootstrap
        num_bootstrap = 100
        mass_samples = []
        for _ in range(num_bootstrap):
            indices = np.random.randint(0, len(distances), size=len(distances))
            bootstrap_distances = distances[indices]
            bootstrap_correlations = correlations[indices]
            
            if np.any(bootstrap_correlations <= 0):
                continue
                
            bootstrap_log_corr = np.log(bootstrap_correlations)
            A = np.vstack([bootstrap_distances, np.ones(len(bootstrap_distances))]).T
            
            try:
                bootstrap_m, _ = np.linalg.lstsq(A, bootstrap_log_corr, rcond=None)[0]
                mass_samples.append(-bootstrap_m)
            except:
                continue
        
        if not mass_samples:
            return -m, 0
            
        mass_error = np.std(mass_samples)
        
        return -m, mass_error

class HybridMonteCarlo:
    """Hybrid Monte Carlo algorithm for lattice field theory.
    
    Implements the HMC algorithm, which uses Hamiltonian dynamics to
    generate more efficient Monte Carlo proposals.
    """
    
    def __init__(self, action_func, grad_func, field_shape, mass=1.0, step_size=0.1, n_steps=10):
        """Initialize the HMC simulator.
        
        Args:
            action_func: Function that computes the action S[phi]
            grad_func: Function that computes the gradient of the action
            field_shape: Shape of the field configuration
            mass: Mass parameter for the momenta
            step_size: Step size for leapfrog integration
            n_steps: Number of leapfrog steps per trajectory
        """
        self.action_func = action_func
        self.grad_func = grad_func
        self.field_shape = field_shape
        self.mass = mass
        self.step_size = step_size
        self.n_steps = n_steps
        
        # Initialize field configuration
        self.field = np.random.normal(0, 0.1, size=field_shape)
    
    def hamiltonian(self, field, momentum):
        """Calculate the Hamiltonian H = T + S.
        
        Args:
            field: Field configuration
            momentum: Conjugate momentum
            
        Returns:
            Value of the Hamiltonian
        """
        # Kinetic energy T = p^2 / 2m
        kinetic = np.sum(momentum**2) / (2 * self.mass)
        
        # Potential energy (action)
        potential = self.action_func(field)
        
        return kinetic + potential
    
    def leapfrog_step(self, field, momentum):
        """Perform a single leapfrog integration step.
        
        Args:
            field: Current field configuration
            momentum: Current momentum configuration
            
        Returns:
            Updated field and momentum
        """
        # Half step for momentum
        momentum -= self.step_size * self.grad_func(field) / 2
        
        # Full step for field
        field += self.step_size * momentum / self.mass
        
        # Half step for momentum
        momentum -= self.step_size * self.grad_func(field) / 2
        
        return field, momentum
    
    def hmc_step(self):
        """Perform a single HMC update.
        
        Returns:
            Boolean indicating if the update was accepted
        """
        # Generate random momentum from Gaussian distribution
        momentum = np.random.normal(0, 1, size=self.field_shape)
        
        # Store initial field and calculate initial Hamiltonian
        initial_field = self.field.copy()
        initial_h = self.hamiltonian(initial_field, momentum)
        
        # Evolve using leapfrog integration
        current_field = initial_field.copy()
        current_momentum = momentum.copy()
        
        for _ in range(self.n_steps):
            current_field, current_momentum = self.leapfrog_step(current_field, current_momentum)
        
        # Calculate final Hamiltonian
        final_h = self.hamiltonian(current_field, current_momentum)
        
        # Metropolis acceptance
        delta_h = final_h - initial_h
        if delta_h < 0 or random.random() < np.exp(-delta_h):
            self.field = current_field
            return True
        else:
            # Reject the update
            return False
    
    def run_simulation(self, n_thermalization=100, n_samples=1000, 
                      measurement_func=None, sample_interval=1):
        """Run an HMC simulation and collect measurements.
        
        Args:
            n_thermalization: Number of thermalization steps
            n_samples: Number of samples to collect
            measurement_func: Function to measure observables
            sample_interval: Steps between measurements
            
        Returns:
            List of measurements
        """
        measurements = []
        acceptances = 0
        
        # Thermalization
        print("Thermalizing...")
        for i in range(n_thermalization):
            accepted = self.hmc_step()
            acceptances += int(accepted)
            
            # Adjust step size to get ~65% acceptance
            if (i+1) % 20 == 0:
                acceptance_rate = acceptances / 20
                if acceptance_rate < 0.6:
                    self.step_size *= 0.9
                elif acceptance_rate > 0.7:
                    self.step_size *= 1.1
                acceptances = 0
        
        # Production
        print("Collecting measurements...")
        acceptances = 0
        
        for i in range(n_samples * sample_interval):
            accepted = self.hmc_step()
            acceptances += int(accepted)
            
            if (i+1) % sample_interval == 0:
                if measurement_func:
                    measurements.append(measurement_func(self.field))
                else:
                    # Default measurement is the field itself
                    measurements.append(self.field.copy())
        
        acceptance_rate = acceptances / (n_samples * sample_interval)
        print(f"Final acceptance rate: {acceptance_rate:.2f}")
        
        return measurements

# Utility functions for lattice field theory
def critical_exponents_fss(lattice_sizes, mass_squared_values, coupling=0.1, dimension=4, 
                          num_configs=1000, thermalization=500):
    """Perform finite size scaling analysis to extract critical exponents.
    
    Args:
        lattice_sizes: List of lattice sizes to simulate
        mass_squared_values: List of mass squared values around the critical point
        coupling: Coupling constant
        dimension: Spacetime dimension
        num_configs: Number of configurations per point
        thermalization: Number of thermalization steps
        
    Returns:
        Dictionary of critical exponents and critical mass
    """
    results = {}
    
    # Store susceptibility for each mass and size
    susceptibilities = {}
    binder_cumulants = {}
    
    for L in lattice_sizes:
        susceptibilities[L] = []
        binder_cumulants[L] = []
        
        lattice_shape = (L,) * dimension
        
        for m2 in mass_squared_values:
            # Create lattice simulation
            sim = LatticeScalarField(lattice_shape, mass_squared=m2, 
                                    coupling=coupling, dimension=dimension)
            
            # Run simulation
            result = sim.run_simulation(num_thermalization=thermalization, 
                                       num_configurations=num_configs)
            
            # Store results
            susceptibilities[L].append(result["susceptibility"])
            binder_cumulants[L].append(result["binder_cumulant"])
    
    # Estimate critical mass from Binder cumulant crossings
    # Critical point is where Binder cumulants for different L intersect
    critical_m2 = 0
    
    # TODO: Implement proper crossing finding algorithm
    
    # Estimate critical exponents
    # γ/ν from susceptibility scaling: χ ~ L^(γ/ν)
    gamma_nu = 0
    
    # ν from Binder cumulant derivative: dU/dm ~ L^(1/ν)
    nu = 0
    
    # β from magnetization scaling: <|m|> ~ L^(-β/ν)
    beta_nu = 0
    
    results = {
        "critical_mass_squared": critical_m2,
        "nu": nu,
        "gamma_nu": gamma_nu,
        "beta_nu": beta_nu,
        "gamma": gamma_nu * nu,
        "beta": beta_nu * nu
    }
    
    return results

def lattice_to_continuum_extrapolation(lattice_spacings, observables, observable_errors=None):
    """Extrapolate lattice results to the continuum limit.
    
    Args:
        lattice_spacings: List of lattice spacing values
        observables: List of measured observables at each spacing
        observable_errors: List of errors on the observables
        
    Returns:
        Tuple of (continuum value, error, fit parameters)
    """
    import scipy.optimize as opt
    
    # For a typical observable, expect O(a) = O(0) + c1*a + c2*a^2 + ...
    def linear_fit(a, o0, c1):
        return o0 + c1 * a
    
    def quadratic_fit(a, o0, c1, c2):
        return o0 + c1 * a + c2 * a**2
    
    # Convert to arrays
    a_values = np.array(lattice_spacings)
    o_values = np.array(observables)
    
    if observable_errors is not None:
        o_errors = np.array(observable_errors)
    else:
        o_errors = np.ones_like(o_values)
    
    # Try linear and quadratic fits
    try:
        linear_params, linear_cov = opt.curve_fit(
            linear_fit, a_values, o_values, sigma=o_errors, absolute_sigma=True
        )
        linear_continuum = linear_params[0]
        linear_error = np.sqrt(linear_cov[0, 0])
        
        quadratic_params, quadratic_cov = opt.curve_fit(
            quadratic_fit, a_values, o_values, sigma=o_errors, absolute_sigma=True
        )
        quadratic_continuum = quadratic_params[0]
        quadratic_error = np.sqrt(quadratic_cov[0, 0])
        
        # Calculate chi-squared per degree of freedom
        linear_chi2 = np.sum(((o_values - linear_fit(a_values, *linear_params)) / o_errors)**2)
        linear_dof = len(a_values) - 2
        linear_chi2_dof = linear_chi2 / linear_dof if linear_dof > 0 else float('inf')
        
        quadratic_chi2 = np.sum(((o_values - quadratic_fit(a_values, *quadratic_params)) / o_errors)**2)
        quadratic_dof = len(a_values) - 3
        quadratic_chi2_dof = quadratic_chi2 / quadratic_dof if quadratic_dof > 0 else float('inf')
        
        # Choose the better fit
        if quadratic_dof > 0 and (quadratic_chi2_dof < linear_chi2_dof or linear_dof <= 0):
            return quadratic_continuum, quadratic_error, {
                "type": "quadratic",
                "params": quadratic_params,
                "covariance": quadratic_cov,
                "chi2_dof": quadratic_chi2_dof
            }
        else:
            return linear_continuum, linear_error, {
                "type": "linear",
                "params": linear_params,
                "covariance": linear_cov,
                "chi2_dof": linear_chi2_dof
            }
    except:
        # If fitting fails, return the smallest a value as an estimate
        idx = np.argmin(a_values)
        return o_values[idx], o_errors[idx], {
            "type": "failed",
            "message": "Fitting failed, returning smallest lattice spacing value"
        }

# Example usage
if __name__ == "__main__":
    # Example 1: Simple phi^4 theory simulation
    print("Example 1: Phi^4 Theory Simulation")
    
    # Create a 2D lattice for visualization
    lattice_size = (20, 20)
    mass_squared = -0.5  # Symmetry breaking
    coupling = 0.5
    
    sim = LatticeScalarField(
        lattice_size, 
        mass_squared=mass_squared, 
        coupling=coupling, 
        dimension=2
    )
    
    # Run simulation and collect measurements
    print("Running Monte Carlo simulation...")
    results = sim.run_simulation(
        num_thermalization=1000,
        num_configurations=2000,
        measurements_interval=5,
        beta=1.0
    )
    
    # Print results
    print(f"Mean field: {results['field_mean']:.6f}")
    print(f"Mean |field|: {results['field_abs_mean']:.6f}")
    print(f"Mean field²: {results['field_squared_mean']:.6f}")
    print(f"Susceptibility: {results['susceptibility']:.6f}")
    print(f"Binder cumulant: {results['binder_cumulant']:.6f}")
    
    # Calculate correlation function and extract mass
    print("\nCalculating correlation function...")
    distances, correlations = sim.calculate_correlation_function(
        num_samples=100,
        thermalization=100
    )
    
    mass, mass_error = sim.calculate_mass_gap((distances, correlations))
    print(f"Extracted mass: {mass:.6f} ± {mass_error:.6f}")
    
    # Example 2: Hybrid Monte Carlo
    print("\nExample 2: Hybrid Monte Carlo")
    
    # Define phi^4 action and gradient for HMC
    def phi4_action(field):
        # Kinetic term
        kinetic = 0
        for i in range(len(field.shape)):
            shifted = np.roll(field, -1, axis=i)
            kinetic += np.sum((shifted - field)**2)
        
        kinetic *= 0.5
        
        # Mass and interaction terms
        mass_term = mass_squared * np.sum(field**2) * 0.5
        interaction = coupling * np.sum(field**4) / 24.0
        
        return kinetic + mass_term + interaction
    
    def phi4_gradient(field):
        # Gradient of the action with respect to the field
        grad = np.zeros_like(field)
        
        # Gradient of kinetic term
        for i in range(len(field.shape)):
            # ∂S/∂φ(x) = -∑_μ [φ(x+μ) + φ(x-μ) - 2φ(x)]
            forward = np.roll(field, -1, axis=i)
            backward = np.roll(field, 1, axis=i)
            grad += -1 * (forward + backward - 2 * field)
        
        # Gradient of mass term: m²φ
        grad += mass_squared * field
        
        # Gradient of interaction term: λφ³/6
        grad += coupling * field**3 / 6.0
        
        return grad
    
    # Create and run HMC simulation
    hmc = HybridMonteCarlo(
        action_func=phi4_action,
        grad_func=phi4_gradient,
        field_shape=(10, 10),
        mass=1.0,
        step_size=0.1,
        n_steps=10
    )
    
    # Measure average field
    def measure_field_avg(field):
        return np.mean(field)
    
    print("Running HMC simulation...")
    measurements = hmc.run_simulation(
        n_thermalization=100,
        n_samples=200,
        measurement_func=measure_field_avg,
        sample_interval=2
    )
    
    # Print results
    mean_field = np.mean(measurements)
    std_field = np.std(measurements)
    print(f"HMC mean field: {mean_field:.6f} ± {std_field/np.sqrt(len(measurements)):.6f}") 