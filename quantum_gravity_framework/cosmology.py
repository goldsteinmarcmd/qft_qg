"""
Cosmological Applications

This module implements cosmological applications of quantum gravity, 
focusing on inflation, dark energy, and early universe phenomena.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class QuantumCosmology:
    """
    Implements cosmological models with quantum gravity corrections.
    
    This class provides methods to simulate modified cosmological evolution
    with dimensional flow effects, running couplings, and quantum corrections.
    """
    
    def __init__(self, dim_profile, expansion_rate=70.0):
        """
        Initialize quantum cosmology model.
        
        Parameters:
        -----------
        dim_profile : callable
            Function that returns dimension as a function of energy scale
        expansion_rate : float
            Current Hubble parameter in km/s/Mpc
        """
        self.dim_profile = dim_profile
        self.H0 = expansion_rate  # km/s/Mpc
        
        # Convert H0 to Planck units
        self.H0_planck = self.H0 * 1.0e3 / (299792458.0 * 1.22e19)
        
        # Cosmological parameters (current values)
        self.omega_m = 0.3  # Matter density parameter
        self.omega_r = 9.24e-5  # Radiation density parameter
        self.omega_lambda = 0.7  # Dark energy density parameter
        
        # Inflation parameters
        self.inflation_scale = 1e16  # GeV
        self.inflation_scale_planck = self.inflation_scale / 1.22e19
        self.e_foldings = 60  # Number of e-foldings
    
    def running_newton_constant(self, energy_scale):
        """
        Calculate running Newton's constant at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Value of Newton's constant at this scale
        """
        # Simple model for running G with asymptotic safety behavior
        dim = self.dim_profile(energy_scale)
        
        # Fixed point behavior at high energies
        if energy_scale > 0.1:
            g_fixed = 0.4 * (4.0 / dim)
            return g_fixed
        else:
            # Standard value at low energies with smooth transition
            return 1.0 / (1.0 + np.log(1.0 + energy_scale))
    
    def running_cosmological_constant(self, energy_scale):
        """
        Calculate running cosmological constant at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Value of cosmological constant at this scale
        """
        # Simple model for running Lambda
        dim = self.dim_profile(energy_scale)
        
        # Value decreases with energy scale
        lambda_value = 0.1 / (1.0 + energy_scale**2)
        
        # Apply dimension correction
        lambda_value *= (4.0 / dim)**2
        
        return lambda_value
    
    def friedmann_equation_qg(self, scale_factor, energy_scale):
        """
        Compute Friedmann equation with quantum gravity corrections.
        
        Parameters:
        -----------
        scale_factor : float
            Scale factor a(t)
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Modified Hubble parameter squared (H²)
        """
        # Get effective dimension at this energy scale
        dim = self.dim_profile(energy_scale)
        
        # Get running Newton's constant
        G_running = self.running_newton_constant(energy_scale)
        
        # Get running cosmological constant
        Lambda_running = self.running_cosmological_constant(energy_scale)
        
        # Convert omega parameters to energy densities
        # Standard Friedmann equation: H² = (8πG/3) ρ
        # With quantum corrections and running couplings
        
        # Matter contribution: ρ_m ~ a^(-3)
        rho_m = self.omega_m * self.H0_planck**2 * scale_factor**(-3)
        
        # Radiation contribution: ρ_r ~ a^(-4)
        rho_r = self.omega_r * self.H0_planck**2 * scale_factor**(-4)
        
        # Dark energy contribution with running Λ
        rho_lambda = Lambda_running / (8 * np.pi * G_running)
        
        # Additional quantum gravity correction from dimensional flow
        # Simple multiplicative correction 
        qg_correction = (4.0 / dim)**(2 - dim/2)
        
        # Compute modified Hubble parameter
        H_squared = (8 * np.pi * G_running / 3) * (rho_m + rho_r + rho_lambda) * qg_correction
        
        return H_squared
    
    def evolve_universe(self, t_range, a_initial=1e-30):
        """
        Evolve universe from early times to present.
        
        Parameters:
        -----------
        t_range : tuple
            (t_start, t_end) in Planck times
        a_initial : float
            Initial scale factor
            
        Returns:
        --------
        dict
            Evolution results
        """
        print(f"Evolving universe from a={a_initial} to present...")
        
        # Function for numerical integration
        def universe_evolution(t, y):
            # y[0] = a (scale factor)
            # y[1] = da/dt
            
            # Current scale factor
            a = max(y[0], 1e-30)  # Prevent division by zero
            
            # Estimate energy scale from scale factor (E ~ 1/a)
            energy_scale = min(1.0 / a, 1e6)  # Cap at 10^6 Planck energy
            
            # Compute Hubble parameter from modified Friedmann equation
            H_squared = self.friedmann_equation_qg(a, energy_scale)
            H = np.sqrt(max(H_squared, 0))  # Ensure non-negative
            
            # Scale factor evolution: da/dt = a * H
            da_dt = a * H
            
            # Second derivative from acceleration equation (not used but included for completeness)
            d2a_dt2 = a * H**2 * (1 - 3 * self.omega_m / (2 * a**3 * H**2))
            
            return [da_dt, d2a_dt2]
        
        # Solve ODE system
        t_span = t_range
        y0 = [a_initial, a_initial * np.sqrt(self.friedmann_equation_qg(a_initial, 1.0/a_initial))]
        
        solution = solve_ivp(
            universe_evolution,
            t_span,
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract results
        times = solution.t
        scale_factors = solution.y[0]
        
        # Compute energy scales
        energy_scales = 1.0 / scale_factors
        energy_scales = np.minimum(energy_scales, 1e6)  # Cap at high energy
        
        # Compute dimensions at each time
        dimensions = np.array([self.dim_profile(e) for e in energy_scales])
        
        # Compute Hubble parameter at each time
        hubble_params = np.zeros_like(times)
        for i, (a, e) in enumerate(zip(scale_factors, energy_scales)):
            hubble_params[i] = np.sqrt(self.friedmann_equation_qg(a, e))
        
        print(f"Evolution complete. Current Hubble: {hubble_params[-1] * 299792.458 * 1.22e19 / 1.0e3:.4f} km/s/Mpc")
        
        return {
            'times': times,
            'scale_factors': scale_factors,
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'hubble_parameters': hubble_params
        }
    
    def inflation_quantum_effects(self, num_efolds=60):
        """
        Simulate inflation with quantum gravity effects.
        
        Parameters:
        -----------
        num_efolds : int
            Number of e-foldings to simulate
            
        Returns:
        --------
        dict
            Inflation results
        """
        print(f"Simulating inflation with {num_efolds} e-foldings...")
        
        # Initialize arrays for results
        n_steps = 1000
        efolds = np.linspace(0, num_efolds, n_steps)
        scale_factors = np.exp(efolds)
        
        # Energy scale during inflation (approximately constant)
        energy_scales = self.inflation_scale_planck * np.exp(-0.1 * efolds / num_efolds)
        
        # Get dimensions during inflation
        dimensions = np.array([self.dim_profile(e) for e in energy_scales])
        
        # Compute slow-roll parameters with quantum corrections
        epsilon = np.zeros_like(efolds)
        eta = np.zeros_like(efolds)
        
        # Quantum gravity corrections to slow-roll parameters
        # Simplified model
        for i, (e, d) in enumerate(zip(energy_scales, dimensions)):
            # Standard slow-roll parameter
            eps_std = 0.01 * (1 - efolds[i] / num_efolds)
            eta_std = 0.01
            
            # Quantum gravity correction factor
            qg_factor = (4.0 / d) * (1 + e**2)
            
            # Apply corrections
            epsilon[i] = eps_std * qg_factor
            eta[i] = eta_std * qg_factor**0.5
        
        # Compute scalar spectral index and tensor-to-scalar ratio
        ns = 1 - 2 * epsilon - eta
        r = 16 * epsilon
        
        # Log results
        print(f"Inflation results:")
        print(f"  Initial dimension: {dimensions[0]:.4f}")
        print(f"  Final dimension: {dimensions[-1]:.4f}")
        print(f"  Scalar spectral index (ns): {ns[-1]:.6f}")
        print(f"  Tensor-to-scalar ratio (r): {r[-1]:.6e}")
        
        return {
            'e_foldings': efolds,
            'scale_factors': scale_factors,
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'epsilon': epsilon,
            'eta': eta,
            'ns': ns,
            'r': r
        }
    
    def dark_energy_equation_of_state(self, redshift_range=(0, 2), num_points=100):
        """
        Compute dark energy equation of state with quantum gravity effects.
        
        Parameters:
        -----------
        redshift_range : tuple
            (z_min, z_max) redshift range
        num_points : int
            Number of points to compute
            
        Returns:
        --------
        dict
            Dark energy equation of state results
        """
        print(f"Computing dark energy EoS from z={redshift_range[0]} to z={redshift_range[1]}...")
        
        # Generate redshift points
        redshifts = np.linspace(redshift_range[0], redshift_range[1], num_points)
        
        # Compute scale factors: a = 1/(1+z)
        scale_factors = 1.0 / (1.0 + redshifts)
        
        # Corresponding energy scales (rough approximation)
        energy_scales = np.minimum(1e-6 / scale_factors, 1.0)  # Keep below Planck scale
        
        # Get effective dimensions
        dimensions = np.array([self.dim_profile(e) for e in energy_scales])
        
        # Get running cosmological constant
        lambda_values = np.array([
            self.running_cosmological_constant(e) 
            for e in energy_scales
        ])
        
        # Get running Newton's constant
        G_values = np.array([
            self.running_newton_constant(e)
            for e in energy_scales
        ])
        
        # Compute dark energy equation of state: w = p/ρ
        # In general relativity with cosmological constant: w = -1
        # With quantum gravity corrections, it can vary
        w_de = -1.0 + (dimensions - 4.0) / 30.0
        
        # Log results
        print(f"Dark energy results:")
        print(f"  Current dark energy equation of state: {w_de[0]:.6f}")
        
        return {
            'redshifts': redshifts,
            'scale_factors': scale_factors,
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'lambda_values': lambda_values,
            'G_values': G_values,
            'w_de': w_de
        }
    
    def visualize_evolution(self, results, plot_type='scale_factor'):
        """
        Visualize cosmological evolution.
        
        Parameters:
        -----------
        results : dict
            Results from evolution simulation
        plot_type : str
            Type of plot: 'scale_factor', 'hubble', 'dimension', etc.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'scale_factor':
            # Plot scale factor evolution
            ax.semilogx(results['times'], results['scale_factors'], 'b-', linewidth=2)
            ax.set_xlabel('Time (Planck time)')
            ax.set_ylabel('Scale Factor a(t)')
            ax.set_title('Universe Scale Factor Evolution')
            ax.grid(True)
            
        elif plot_type == 'hubble':
            # Plot Hubble parameter evolution
            ax.loglog(results['scale_factors'], results['hubble_parameters'], 'r-', linewidth=2)
            ax.set_xlabel('Scale Factor a')
            ax.set_ylabel('Hubble Parameter H (Planck units)')
            ax.set_title('Hubble Parameter Evolution')
            ax.grid(True)
            
        elif plot_type == 'dimension':
            # Plot dimension evolution
            ax.semilogx(results['scale_factors'], results['dimensions'], 'g-', linewidth=2)
            ax.set_xlabel('Scale Factor a')
            ax.set_ylabel('Effective Dimension')
            ax.set_title('Dimensional Flow Through Cosmic History')
            ax.grid(True)
            
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Create dimension profile for testing
    # Asymptotic UV dimension: 2
    # Asymptotic IR dimension: 4
    dim_profile = lambda E: 2.0 + 2.0 / (1.0 + (100.0 * E)**2)
    
    # Create quantum cosmology model
    qc = QuantumCosmology(dim_profile)
    
    # Test inflation simulation
    inflation_results = qc.inflation_quantum_effects(num_efolds=60)
    
    # Plot epsilon evolution during inflation
    plt.figure(figsize=(10, 6))
    plt.semilogy(inflation_results['e_foldings'], inflation_results['epsilon'], 'b-', linewidth=2)
    plt.xlabel('e-foldings')
    plt.ylabel('Slow-roll parameter $\\epsilon$')
    plt.title('Evolution of Slow-Roll Parameter During Inflation')
    plt.grid(True)
    plt.savefig("inflation_epsilon_evolution.png")
    print("\nSaved inflation plot to inflation_epsilon_evolution.png")
    
    # Test dark energy evolution
    de_results = qc.dark_energy_equation_of_state(redshift_range=(0, 2))
    
    # Plot equation of state parameter
    plt.figure(figsize=(10, 6))
    plt.plot(de_results['redshifts'], de_results['w_de'], 'r-', linewidth=2)
    plt.xlabel('Redshift z')
    plt.ylabel('Dark Energy Equation of State w')
    plt.title('Dark Energy Equation of State Evolution')
    plt.grid(True)
    plt.savefig("dark_energy_eos_evolution.png")
    print("Saved dark energy plot to dark_energy_eos_evolution.png") 