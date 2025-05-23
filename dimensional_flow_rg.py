"""
Dimensional Flow Renormalization Group for Quantum Gravity

This module implements advanced renormalization group techniques that explicitly 
incorporate spacetime dimensional flow from our quantum gravity framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings

# Fix imports for local testing
try:
    from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
    from quantum_gravity_framework.qft_integration import QFTIntegration
except ImportError:
    from quantum_spacetime import QuantumSpacetimeAxioms
    from qft_integration import QFTIntegration


class DimensionalFlowRG:
    """
    Implements renormalization group methods incorporating spacetime dimensional flow.
    """
    
    def __init__(self, dim_uv=2.0, dim_ir=4.0, transition_scale=1.0, couplings=None):
        """
        Initialize the dimensional flow RG.
        
        Parameters:
        -----------
        dim_uv : float
            UV (high energy) spectral dimension
        dim_ir : float
            IR (low energy) spectral dimension
        transition_scale : float
            Scale of dimension transition (in Planck units)
        couplings : dict, optional
            Initial couplings (e.g., {'g': 0.1, 'lambda': 0.5})
        """
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        
        # Initialize couplings with default values if not provided
        if couplings is None:
            self.couplings = {
                'g': 0.1,       # Gauge coupling (like in QCD)
                'lambda': 0.2,  # Self-interaction coupling (like in φ⁴ theory)
                'y': 0.5,       # Yukawa coupling (fermion-scalar interaction)
                'G': 1.0        # Gravitational coupling (Newton's constant)
            }
        else:
            self.couplings = couplings
            
        # Initialize QG components
        self.qst = QuantumSpacetimeAxioms(dim=self.dim_ir)
        self.qft = QFTIntegration(dim=self.dim_ir)
        
        # Store RG flow results
        self.flow_results = {}
    
    def compute_spectral_dimension(self, energy_scale):
        """
        Compute the spectral dimension at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Spectral dimension
        """
        # Convert energy scale to diffusion time
        # In a diffusion process, time is inversely related to energy squared
        diffusion_time = 1.0 / (energy_scale * energy_scale)
        
        # Use our quantum spacetime model to compute spectral dimension
        try:
            return self.qst.compute_spectral_dimension(diffusion_time)
        except:
            # Fallback to smooth interpolation if direct computation fails
            x = np.log10(energy_scale)
            return self.dim_ir + (self.dim_uv - self.dim_ir) / (1 + np.exp(-2 * (x - np.log10(self.transition_scale))))
    
    def beta_functions(self, scale, couplings, dimension=None):
        """
        Compute beta functions for all couplings at a given scale.
        
        Parameters:
        -----------
        scale : float
            Energy scale in Planck units
        couplings : dict
            Current coupling values
        dimension : float, optional
            Spectral dimension (computed from scale if not provided)
            
        Returns:
        --------
        dict
            Beta functions for all couplings
        """
        # Get dimension at this scale if not provided
        if dimension is None:
            dimension = self.compute_spectral_dimension(scale)
            
        # Extract couplings
        g = couplings.get('g', 0.0)
        lambda_val = couplings.get('lambda', 0.0)
        y = couplings.get('y', 0.0)
        G = couplings.get('G', 0.0)
        
        # Compute dimension-dependent beta functions
        
        # For gauge coupling g:
        # In 4D: β(g) = -b0*g^3 (QCD-like)
        # Near 2D: β(g) = (d-2)/2 * g + O(g^3)
        b0 = 11.0 / (3.0 * (4*np.pi)**2)  # QCD-like coefficient
        beta_g = ((dimension - 4.0) / (dimension - 2.0)) * (-b0 * g**3) + (dimension - 2.0) * g / 2.0
        
        # For scalar self-coupling lambda:
        # In 4D: β(λ) = (3λ^2 + y^2λ - y^4)/16π^2 (Standard Model Higgs-like)
        # In 2D: λ becomes more relevant
        beta_lambda = ((dimension - 4.0) / (dimension - 2.0)) * (3*lambda_val**2 + y**2*lambda_val - y**4) / (16*np.pi**2)
        beta_lambda += (dimension - 4.0) * lambda_val / 2.0  # Canonical scaling
        
        # For Yukawa coupling y:
        # In 4D: β(y) = y*(y^2 - g^2)/16π^2
        # Near 2D: Different scaling
        beta_y = ((dimension - 4.0) / (dimension - 2.0)) * y * (y**2 - g**2) / (16*np.pi**2)
        beta_y += (dimension - 4.0) * y / 2.0  # Canonical scaling
        
        # For gravitational coupling G:
        # In 4D: β(G) = 2G - ω*G^2 with ω=167/30π
        # Near 2D: β(G) = (d-2)*G
        omega = 167.0 / (30.0 * np.pi)
        beta_G = (dimension - 2.0) * G - ((dimension - 2.0) / 2.0) * omega * G**2
        
        return {
            'g': beta_g,
            'lambda': beta_lambda,
            'y': beta_y,
            'G': beta_G
        }
    
    def _rg_flow_equations(self, t, y):
        """
        System of RG flow equations for solve_ivp.
        
        Parameters:
        -----------
        t : float
            Log scale (t = ln(μ/μ0))
        y : array
            Coupling values [g, lambda, y, G]
            
        Returns:
        --------
        array
            Beta functions [dg/dt, dlambda/dt, dy/dt, dG/dt]
        """
        # Extract scale and couplings
        scale = np.exp(t)  # Actual energy scale
        
        # Convert y array to coupling dict
        coupling_keys = list(self.couplings.keys())
        coupling_dict = {key: y[i] for i, key in enumerate(coupling_keys)}
        
        # Compute spectral dimension at this scale
        dimension = self.compute_spectral_dimension(scale)
        
        # Get beta functions
        betas = self.beta_functions(scale, coupling_dict, dimension)
        
        # Convert beta dict to array in the same order as y
        beta_array = np.array([betas[key] for key in coupling_keys])
        
        return beta_array
    
    def compute_rg_flow(self, scale_range=(1e-6, 1e3), num_points=100):
        """
        Compute the renormalization group flow over a range of energy scales.
        
        Parameters:
        -----------
        scale_range : tuple
            Min and max energy scales (in Planck units)
        num_points : int
            Number of points to compute
            
        Returns:
        --------
        dict
            RG flow results
        """
        print("Computing RG flow with dimensional flow effects...")
        
        # Convert scale range to log scale for integration
        t_min = np.log(scale_range[0])
        t_max = np.log(scale_range[1])
        
        # Extract initial coupling values as array
        coupling_keys = list(self.couplings.keys())
        y0 = np.array([self.couplings[key] for key in coupling_keys])
        
        # Solve the RG flow equations
        sol = solve_ivp(
            self._rg_flow_equations,
            [t_min, t_max],
            y0,
            method='Radau',
            t_eval=np.linspace(t_min, t_max, num_points)
        )
        
        # Extract results
        scales = np.exp(sol.t)
        dimensions = np.array([self.compute_spectral_dimension(s) for s in scales])
        coupling_trajectories = {
            key: sol.y[i] for i, key in enumerate(coupling_keys)
        }
        
        # Compute anomalous dimensions at each scale
        anomalous_dims = []
        for i, scale in enumerate(scales):
            couplings = {key: coupling_trajectories[key][i] for key in coupling_keys}
            # A simple model for anomalous dimensions
            g, lambda_val = couplings.get('g', 0.0), couplings.get('lambda', 0.0)
            anomalous_dim = 0.1 * g**2 + 0.05 * lambda_val
            anomalous_dims.append(anomalous_dim)
        
        # Store and return results
        results = {
            'scales': scales,
            'dimensions': dimensions,
            'coupling_trajectories': coupling_trajectories,
            'anomalous_dimensions': np.array(anomalous_dims),
            'coupling_keys': coupling_keys
        }
        
        self.flow_results = results
        return results
    
    def find_fixed_points(self, dimension=None):
        """
        Find fixed points of the RG flow at a specific dimension.
        
        Parameters:
        -----------
        dimension : float, optional
            Spacetime dimension (uses UV dimension if not specified)
            
        Returns:
        --------
        list
            Fixed points (list of coupling dictionaries)
        """
        if dimension is None:
            dimension = self.dim_uv
            
        print(f"Finding fixed points at dimension d = {dimension}...")
        
        # We'll use a simple grid search approach
        # For a more sophisticated approach, we would use a root-finding algorithm
        
        # Define search ranges for couplings
        search_ranges = {
            'g': np.linspace(0.0, 2.0, 10),
            'lambda': np.linspace(-1.0, 1.0, 10),
            'y': np.linspace(0.0, 2.0, 10),
            'G': np.linspace(0.0, 2.0, 10)
        }
        
        # List to store fixed points
        fixed_points = []
        
        # For a full grid search we would need nested loops
        # For simplicity, we'll just search along 1D lines
        for key in self.couplings.keys():
            for value in search_ranges[key]:
                # Create a coupling dict with this value
                test_couplings = {k: 0.01 for k in self.couplings.keys()}
                test_couplings[key] = value
                
                # Compute beta functions
                betas = self.beta_functions(1.0, test_couplings, dimension)
                
                # Check if this is approximately a fixed point
                if max(abs(beta) for beta in betas.values()) < 0.01:
                    fixed_points.append(test_couplings.copy())
        
        # Also check Gaussian fixed point explicitly
        gaussian_point = {k: 0.0 for k in self.couplings.keys()}
        betas = self.beta_functions(1.0, gaussian_point, dimension)
        if max(abs(beta) for beta in betas.values()) < 0.01:
            fixed_points.append(gaussian_point)
        
        # Filter out duplicates
        unique_points = []
        for point in fixed_points:
            if not any(all(abs(point[k] - p[k]) < 0.02 for k in point.keys()) for p in unique_points):
                unique_points.append(point)
        
        print(f"Found {len(unique_points)} fixed points.")
        for i, point in enumerate(unique_points):
            print(f"Fixed point {i+1}:")
            for k, v in point.items():
                print(f"  {k} = {v:.4f}")
        
        return unique_points
    
    def plot_rg_flow(self, fixed_points=None, save_path=None):
        """
        Plot the RG flow results.
        
        Parameters:
        -----------
        fixed_points : list, optional
            List of fixed points to mark on the plot
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if not self.flow_results:
            raise ValueError("No RG flow results available. Run compute_rg_flow first.")
            
        # Extract data
        scales = self.flow_results['scales']
        dimensions = self.flow_results['dimensions']
        trajectories = self.flow_results['coupling_trajectories']
        anomalous_dims = self.flow_results['anomalous_dimensions']
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 10))
        
        # Plot 1: Spectral dimension vs scale
        ax1 = fig.add_subplot(221)
        ax1.semilogx(scales, dimensions, 'b-', linewidth=2)
        ax1.set_xlabel('Energy Scale (Planck units)')
        ax1.set_ylabel('Spectral Dimension')
        ax1.set_title('Dimensional Flow')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference lines at UV and IR dimensions
        ax1.axhline(y=self.dim_uv, color='r', linestyle='--', alpha=0.7, 
                   label=f'UV: d = {self.dim_uv}')
        ax1.axhline(y=self.dim_ir, color='g', linestyle='--', alpha=0.7,
                   label=f'IR: d = {self.dim_ir}')
        ax1.legend()
        
        # Plot 2: Coupling trajectories
        ax2 = fig.add_subplot(222)
        for key, trajectory in trajectories.items():
            ax2.loglog(scales, np.abs(trajectory), '-', linewidth=2, label=key)
        ax2.set_xlabel('Energy Scale (Planck units)')
        ax2.set_ylabel('Coupling Value (abs)')
        ax2.set_title('RG Flow of Couplings')
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot 3: Anomalous dimensions
        ax3 = fig.add_subplot(223)
        ax3.semilogx(scales, anomalous_dims, 'g-', linewidth=2)
        ax3.set_xlabel('Energy Scale (Planck units)')
        ax3.set_ylabel('Anomalous Dimension')
        ax3.set_title('Anomalous Dimension')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Selected coupling phase space
        if 'g' in trajectories and 'G' in trajectories:
            ax4 = fig.add_subplot(224)
            ax4.plot(trajectories['g'], trajectories['G'], 'b-', linewidth=2, alpha=0.7)
            ax4.plot(trajectories['g'][0], trajectories['G'][0], 'go', label='IR')
            ax4.plot(trajectories['g'][-1], trajectories['G'][-1], 'ro', label='UV')
            
            # Add fixed points if provided
            if fixed_points:
                for i, point in enumerate(fixed_points):
                    if 'g' in point and 'G' in point:
                        ax4.plot(point['g'], point['G'], 'ko', markersize=8, 
                               label=f'FP {i+1}')
            
            ax4.set_xlabel('g (Gauge Coupling)')
            ax4.set_ylabel('G (Gravitational Coupling)')
            ax4.set_title('Phase Space Projection')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def compute_critical_exponents(self, fixed_point, dimension=None):
        """
        Compute critical exponents around a fixed point.
        
        Parameters:
        -----------
        fixed_point : dict
            Fixed point couplings
        dimension : float, optional
            Spacetime dimension
            
        Returns:
        --------
        dict
            Critical exponents
        """
        if dimension is None:
            dimension = self.dim_uv
            
        # Small perturbation for numerical derivatives
        epsilon = 1e-6
        
        # Get coupling keys
        keys = list(fixed_point.keys())
        
        # Compute Jacobian matrix of beta functions
        jacobian = np.zeros((len(keys), len(keys)))
        
        for i, key_i in enumerate(keys):
            # Compute beta functions at the fixed point
            betas_at_fp = self.beta_functions(1.0, fixed_point, dimension)
            
            for j, key_j in enumerate(keys):
                # Perturb the j-th coupling
                perturbed_couplings = fixed_point.copy()
                perturbed_couplings[key_j] += epsilon
                
                # Compute beta functions with perturbed coupling
                perturbed_betas = self.beta_functions(1.0, perturbed_couplings, dimension)
                
                # Compute derivative: dβ_i/dg_j
                derivative = (perturbed_betas[key_i] - betas_at_fp[key_i]) / epsilon
                jacobian[i, j] = derivative
        
        # Compute eigenvalues of the stability matrix
        try:
            eigenvalues = np.linalg.eigvals(jacobian)
        except:
            warnings.warn("Failed to compute eigenvalues. Using approximate method.")
            # Approximate eigenvalues for diagonal Jacobian
            eigenvalues = np.diag(jacobian)
        
        # Critical exponents are related to eigenvalues
        # Relevant directions have negative eigenvalues
        relevant = [ev for ev in eigenvalues if ev.real < 0]
        irrelevant = [ev for ev in eigenvalues if ev.real > 0]
        marginal = [ev for ev in eigenvalues if abs(ev.real) < 1e-10]
        
        return {
            'eigenvalues': eigenvalues,
            'relevant': relevant,
            'irrelevant': irrelevant,
            'marginal': marginal,
            'stability_matrix': jacobian
        }
    
    def physical_predictions_from_flow(self):
        """
        Extract physical predictions from the RG flow.
        
        Returns:
        --------
        dict
            Physical predictions
        """
        if not self.flow_results:
            raise ValueError("No RG flow results available. Run compute_rg_flow first.")
            
        # Extract data
        scales = self.flow_results['scales']
        dimensions = self.flow_results['dimensions']
        trajectories = self.flow_results['coupling_trajectories']
        
        # Find physical scale (e.g., where dimension is 3.9, close to IR but with QG effects)
        phys_idx = np.abs(dimensions - 3.9).argmin()
        phys_scale = scales[phys_idx]
        
        # Extract couplings at this scale
        phys_couplings = {k: trajectories[k][phys_idx] for k in trajectories}
        
        # Make physical predictions based on these couplings
        # These are simplified examples
        
        # Effective Newton's constant
        G_eff = phys_couplings.get('G', 1.0)
        # Newton's constant in ~10^19 GeV, convert to more standard units
        G_newton = G_eff * 6.7e-39  # GeV^-2
        
        # Running gauge coupling affecting LHC physics
        g_eff = phys_couplings.get('g', 0.1)
        # Cross section modification from QG effects
        xsec_modification = 1.0 + 0.1 * (phys_scale / self.transition_scale)**2
        
        # Prediction for potential dimensionful observables
        # e.g., correlation length near a phase transition
        xi = (1.0 / phys_scale) * (1.0 / max(0.1, phys_couplings.get('lambda', 0.1)))
        
        # Store predictions
        predictions = {
            'effective_physical_scale': phys_scale,
            'effective_dimension': dimensions[phys_idx],
            'effective_couplings': phys_couplings,
            'effective_newton_constant': G_newton,
            'gauge_coupling_value': g_eff,
            'cross_section_modification': xsec_modification,
            'correlation_length': xi
        }
        
        # Print summary
        print("\nPhysical predictions from RG flow:")
        print(f"  Effective physical scale: {phys_scale:.2e} Planck units")
        print(f"  Effective dimension: {dimensions[phys_idx]:.4f}")
        print(f"  Effective Newton's constant: {G_newton:.2e} GeV^-2")
        print(f"  LHC cross section modification: {xsec_modification:.2%}")
        print(f"  Correlation length: {xi:.2e} Planck units")
        
        return predictions


if __name__ == "__main__":
    # Test the dimensional flow RG implementation
    
    # Create a dimensional flow RG instance
    dim_flow_rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
    
    # Initial couplings
    initial_couplings = {
        'g': 0.5,       # Gauge coupling
        'lambda': 0.1,  # Self-interaction coupling
        'y': 0.2,       # Yukawa coupling
        'G': 1.0        # Gravitational coupling
    }
    dim_flow_rg.couplings = initial_couplings
    
    # Compute RG flow
    flow_results = dim_flow_rg.compute_rg_flow(scale_range=(1e-4, 1e3), num_points=200)
    
    # Find fixed points in the UV
    fixed_points = dim_flow_rg.find_fixed_points(dimension=2.0)
    
    # Compute critical exponents for the first fixed point
    if fixed_points:
        exponents = dim_flow_rg.compute_critical_exponents(fixed_points[0], dimension=2.0)
        print("Critical Exponents:")
        print(f"  Eigenvalues: {exponents['eigenvalues']}")
        print(f"  Relevant directions: {len(exponents['relevant'])}")
        print(f"  Irrelevant directions: {len(exponents['irrelevant'])}")
    
    # Plot the RG flow
    dim_flow_rg.plot_rg_flow(fixed_points=fixed_points, save_path="dimensional_flow_rg.png")
    
    # Make physical predictions
    predictions = dim_flow_rg.physical_predictions_from_flow() 