"""
Unified Approach: Combining Dimensional Flow RG with Non-Perturbative Path Integrals

This module implements a unified approach that combines dimensional flow renormalization
group techniques with non-perturbative path integral methods to provide a more 
comprehensive framework for quantum gravity calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG
from quantum_gravity_framework.non_perturbative_path_integral import NonPerturbativePathIntegral


class UnifiedQGApproach:
    """
    Implements a unified approach to quantum gravity combining dimensional 
    flow RG with non-perturbative path integral techniques.
    """
    
    def __init__(self, dim_uv=2.0, dim_ir=4.0, transition_scale=1.0, 
                 lattice_size=10, beta=1.0, coupling=0.1):
        """
        Initialize the unified QG approach.
        
        Parameters:
        -----------
        dim_uv : float
            UV (high energy) spectral dimension
        dim_ir : float
            IR (low energy) spectral dimension
        transition_scale : float
            Scale of dimension transition (in Planck units)
        lattice_size : int
            Size of the lattice for path integral discretization
        beta : float
            Inverse temperature parameter
        coupling : float
            Coupling strength parameter
        """
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        self.lattice_size = lattice_size
        self.beta = beta
        self.coupling = coupling
        
        # Initialize dimensional flow RG
        self.rg = DimensionalFlowRG(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize path integral at IR dimension initially
        self.path_integral = NonPerturbativePathIntegral(
            dim=int(self.dim_ir),  # Discretized spacetime needs integer dimension
            lattice_size=lattice_size,
            beta=beta,
            coupling=coupling
        )
        
        # Store results
        self.unified_results = {}
    
    def compute_dimension_dependent_action(self, energy_scale):
        """
        Compute the effective action with dimension-dependent couplings at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Effective action parameters
        """
        print(f"Computing dimension-dependent action at energy scale {energy_scale:.2e}...")
        
        # Get spectral dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Compute RG flow if not already computed
        if not self.rg.flow_results:
            self.rg.compute_rg_flow(scale_range=(energy_scale*0.1, energy_scale*10))
        
        # Get running couplings at this scale
        # Extract nearest scale from RG results
        scales = self.rg.flow_results['scales']
        idx = np.abs(scales - energy_scale).argmin()
        scale = scales[idx]
        
        # Extract couplings
        coupling_dict = {}
        for key in self.rg.flow_results['coupling_trajectories']:
            coupling_dict[key] = self.rg.flow_results['coupling_trajectories'][key][idx]
        
        # Map RG couplings to path integral parameters
        # In a φ⁴ theory, we need to map λ → coupling and m² → mass
        action_params = {
            'coupling': coupling_dict.get('lambda', self.coupling),
            'mass_squared': 1.0,  # Default value
            'dimension': dimension,
            'energy_scale': energy_scale
        }
        
        # Dimension-dependent modifications to the path integral measure
        # In d dimensions, the measure is ∏ᵢ dφᵢ μ(φᵢ)
        # where μ is a dimension-dependent function
        
        # For non-integer dimensions, we need to modify the measure
        measure_factor = np.power(energy_scale / self.transition_scale, dimension - self.dim_ir)
        action_params['measure_factor'] = measure_factor
        
        return action_params
    
    def compute_effective_action(self, field_amplitude=1.0, num_scales=5):
        """
        Compute the effective action across multiple energy scales.
        
        Parameters:
        -----------
        field_amplitude : float
            Amplitude of the background field
        num_scales : int
            Number of energy scales to compute
            
        Returns:
        --------
        dict
            Effective action results at different scales
        """
        print(f"Computing effective action across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-3, 3, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'effective_actions': [],
            'field_amplitude': field_amplitude,
            'measure_factors': []
        }
        
        # Compute effective action at each scale
        for scale in energy_scales:
            # Get dimension-dependent action parameters
            action_params = self.compute_dimension_dependent_action(scale)
            
            # Store dimension
            results['dimensions'].append(action_params['dimension'])
            results['measure_factors'].append(action_params['measure_factor'])
            
            # Set up path integral with appropriate dimension and couplings
            self.path_integral.coupling = action_params['coupling']
            
            # For non-integer dimensions, we use the nearest integer for discretization
            # but modify the measure and couplings to approximate the fractional dimension
            dim_int = int(np.round(action_params['dimension']))
            if dim_int != self.path_integral.dim:
                # Recreate path integral with new dimension
                self.path_integral = NonPerturbativePathIntegral(
                    dim=dim_int,
                    lattice_size=self.lattice_size,
                    beta=self.beta * action_params['measure_factor'],  # Scale beta by measure factor
                    coupling=action_params['coupling']
                )
            
            # Compute effective action for a background field
            # S_eff[φ] = -log(∫Dφ' exp(-S[φ+φ']))
            # Here we approximate with a quadratic expansion around φ
            
            # For simplicity, we'll compute the effective potential at fixed field value
            eff_action_result = self._compute_effective_potential_at_field(field_amplitude)
            
            # Add results
            results['effective_actions'].append(eff_action_result)
            
            print(f"  Scale: {scale:.2e}, Dimension: {action_params['dimension']:.3f}, "
                  f"Effective action: {eff_action_result['value']:.6f}")
        
        # Store and return results
        self.unified_results['effective_action'] = results
        return results
    
    def _compute_effective_potential_at_field(self, field_amplitude):
        """
        Compute the effective potential at a fixed background field value.
        
        Parameters:
        -----------
        field_amplitude : float
            Amplitude of the background field
            
        Returns:
        --------
        dict
            Effective potential result
        """
        # In a more complete implementation, we would:
        # 1. Set up a background field configuration
        # 2. Perform Monte Carlo integration around this background
        # 3. Compute the effective action
        
        # For now, we'll use a simplified approach where we compute the effective
        # potential using the path integral's existing methods
        
        # Generate a constant field configuration with the given amplitude
        field_config = np.ones(self.path_integral.total_sites) * field_amplitude
        
        # Compute the action for this configuration
        classical_action = self.path_integral._compute_action(field_config)
        
        # Compute fluctuation contribution (approximate)
        # Use path integral to compute Z = ∫Dφ exp(-S[φ])
        z_result = self.path_integral.compute_path_integral(
            observable="partition_function",
            num_samples=100  # Small number for demonstration
        )
        
        # Effective action is: S_eff = S_classical - log(Z)
        eff_action = classical_action - np.log(z_result['value'])
        
        return {
            'value': eff_action,
            'field_amplitude': field_amplitude,
            'classical_action': classical_action,
            'quantum_correction': -np.log(z_result['value']),
            'uncertainty': z_result['error'] / z_result['value']  # Propagate error
        }
    
    def compute_multiscale_correlators(self, num_scales=3, num_samples=500):
        """
        Compute correlation functions across multiple energy scales.
        
        Parameters:
        -----------
        num_scales : int
            Number of energy scales to compute
        num_samples : int
            Number of Monte Carlo samples
            
        Returns:
        --------
        dict
            Correlation functions at different scales
        """
        print(f"Computing correlation functions across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-2, 2, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'correlation_functions': [],
            'correlation_lengths': []
        }
        
        # Compute correlators at each scale
        for scale in energy_scales:
            # Get dimension-dependent action parameters
            action_params = self.compute_dimension_dependent_action(scale)
            
            # Store dimension
            results['dimensions'].append(action_params['dimension'])
            
            # Set up path integral with appropriate dimension and couplings
            self.path_integral.coupling = action_params['coupling']
            
            # For non-integer dimensions, use nearest integer for discretization
            dim_int = int(np.round(action_params['dimension']))
            if dim_int != self.path_integral.dim:
                # Recreate path integral with new dimension
                self.path_integral = NonPerturbativePathIntegral(
                    dim=dim_int,
                    lattice_size=self.lattice_size,
                    beta=self.beta * action_params['measure_factor'],
                    coupling=action_params['coupling']
                )
            
            # Compute correlation function
            corr_result = self.path_integral.compute_path_integral(
                observable="correlation",
                num_samples=num_samples
            )
            
            # Fit to extract correlation length
            # G(r) ~ exp(-r/ξ)
            distances = corr_result['distances']
            corr_values = corr_result['mean']
            
            # Simple exponential fit for distances > 1
            mask = distances > 1
            if np.sum(mask) >= 3:  # Need at least 3 points for reliable fit
                log_corr = np.log(np.maximum(corr_values[mask], 1e-10))  # Avoid log(0)
                fit = np.polyfit(distances[mask], log_corr, 1)
                corr_length = -1.0 / fit[0] if fit[0] < 0 else float('inf')
            else:
                corr_length = np.nan
            
            # Store results
            results['correlation_functions'].append(corr_result)
            results['correlation_lengths'].append(corr_length)
            
            print(f"  Scale: {scale:.2e}, Dimension: {action_params['dimension']:.3f}, "
                  f"Correlation length: {corr_length:.3f}")
        
        # Store and return results
        self.unified_results['correlators'] = results
        return results
    
    def plot_unified_results(self, save_path=None):
        """
        Plot the unified QG results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Check if we have results to plot
        if not self.unified_results:
            raise ValueError("No unified results available. Run computations first.")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 10))
        
        # Plot 1: Spectral dimension vs energy scale
        ax1 = fig.add_subplot(221)
        
        if 'effective_action' in self.unified_results:
            data = self.unified_results['effective_action']
            ax1.semilogx(data['energy_scales'], data['dimensions'], 'bo-', linewidth=2)
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
        
        # Plot 2: Effective action vs spectral dimension
        ax2 = fig.add_subplot(222)
        
        if 'effective_action' in self.unified_results:
            data = self.unified_results['effective_action']
            effective_actions = [result['value'] for result in data['effective_actions']]
            
            ax2.plot(data['dimensions'], effective_actions, 'ro-', linewidth=2)
            ax2.set_xlabel('Spectral Dimension')
            ax2.set_ylabel('Effective Action')
            ax2.set_title(f'Effective Action at Field Amplitude = {data["field_amplitude"]}')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Correlation length vs spectral dimension
        ax3 = fig.add_subplot(223)
        
        if 'correlators' in self.unified_results:
            data = self.unified_results['correlators']
            
            ax3.plot(data['dimensions'], data['correlation_lengths'], 'go-', linewidth=2)
            ax3.set_xlabel('Spectral Dimension')
            ax3.set_ylabel('Correlation Length')
            ax3.set_title('Correlation Length vs Dimension')
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Measure factor vs spectral dimension
        ax4 = fig.add_subplot(224)
        
        if 'effective_action' in self.unified_results:
            data = self.unified_results['effective_action']
            
            ax4.semilogy(data['dimensions'], data['measure_factors'], 'mo-', linewidth=2)
            ax4.set_xlabel('Spectral Dimension')
            ax4.set_ylabel('Measure Factor')
            ax4.set_title('Path Integral Measure Factor')
            ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def derive_unified_scaling_law(self):
        """
        Derive a unified scaling law by combining RG flow with path integral results.
        
        Returns:
        --------
        dict
            Unified scaling law parameters
        """
        print("Deriving unified scaling law...")
        
        # Need both types of results
        if 'effective_action' not in self.unified_results or 'correlators' not in self.unified_results:
            raise ValueError("Need both effective action and correlator results. Run computations first.")
        
        ea_data = self.unified_results['effective_action']
        corr_data = self.unified_results['correlators']
        
        # Extract data
        dimensions = ea_data['dimensions']
        energies = ea_data['energy_scales']
        eff_actions = [result['value'] for result in ea_data['effective_actions']]
        corr_lengths = corr_data['correlation_lengths']
        
        # Look for scaling relation of the form:
        # S_eff ~ E^α(d) and ξ ~ E^β(d)
        # where α(d) and β(d) are dimension-dependent exponents
        
        # Compute numerical derivatives (finite differences)
        alpha_values = []
        beta_values = []
        
        for i in range(1, len(energies)):
            # log(S_eff) / log(E) gives scaling exponent
            log_ratio_E = np.log(energies[i] / energies[i-1])
            
            if eff_actions[i] > 0 and eff_actions[i-1] > 0:
                log_ratio_S = np.log(eff_actions[i] / eff_actions[i-1])
                alpha = log_ratio_S / log_ratio_E
                alpha_values.append(alpha)
            else:
                alpha_values.append(np.nan)
            
            if corr_lengths[i] > 0 and corr_lengths[i-1] > 0:
                log_ratio_xi = np.log(corr_lengths[i] / corr_lengths[i-1])
                beta = log_ratio_xi / log_ratio_E
                beta_values.append(beta)
            else:
                beta_values.append(np.nan)
        
        # Associate with dimensions (use midpoints)
        dim_midpoints = [(dimensions[i] + dimensions[i-1])/2 for i in range(1, len(dimensions))]
        
        # Fit alpha(d) and beta(d) to analytical forms
        # α(d) = c₁(d-2) + c₂(d-2)²
        # β(d) = c₃(d-4) + c₄(d-4)²
        
        # Filter out nans
        valid_alpha = ~np.isnan(alpha_values)
        valid_beta = ~np.isnan(beta_values)
        
        alpha_fit = {}
        beta_fit = {}
        
        # Only fit if we have enough valid data points
        if np.sum(valid_alpha) >= 3:
            x_alpha = np.array([d - 2 for i, d in enumerate(dim_midpoints) if valid_alpha[i]])
            y_alpha = np.array([alpha for i, alpha in enumerate(alpha_values) if valid_alpha[i]])
            alpha_coeffs = np.polyfit(x_alpha, y_alpha, 2)
            alpha_fit = {
                'coeffs': alpha_coeffs,
                'form': 'α(d) = {:.4f}(d-2) + {:.4f}(d-2)² + {:.4f}'.format(
                    alpha_coeffs[1], alpha_coeffs[0], alpha_coeffs[2])
            }
        
        if np.sum(valid_beta) >= 3:
            x_beta = np.array([d - 4 for i, d in enumerate(dim_midpoints) if valid_beta[i]])
            y_beta = np.array([beta for i, beta in enumerate(beta_values) if valid_beta[i]])
            beta_coeffs = np.polyfit(x_beta, y_beta, 2)
            beta_fit = {
                'coeffs': beta_coeffs,
                'form': 'β(d) = {:.4f}(d-4) + {:.4f}(d-4)² + {:.4f}'.format(
                    beta_coeffs[1], beta_coeffs[0], beta_coeffs[2])
            }
        
        # Unified scaling law
        # This connects the macroscopic physics (correlation length)
        # with the microscopic theory (effective action)
        unified_relation = "S_eff ~ ξ^γ where γ(d) depends on dimension"
        
        # If we have both fits, compute γ(d) = α(d)/β(d)
        gamma_function = None
        if alpha_fit and beta_fit:
            unified_relation = "S_eff ~ ξ^(-α(d)/β(d))"
            gamma_function = "γ(d) = -α(d)/β(d)"
        
        # Store and return results
        scaling_law = {
            'dimensions': dim_midpoints,
            'alpha_values': alpha_values,
            'beta_values': beta_values,
            'alpha_fit': alpha_fit,
            'beta_fit': beta_fit,
            'unified_relation': unified_relation,
            'gamma_function': gamma_function
        }
        
        print(f"Derived unified scaling law: {unified_relation}")
        if alpha_fit:
            print(f"  α(d) = {alpha_fit['form']}")
        if beta_fit:
            print(f"  β(d) = {beta_fit['form']}")
        
        self.unified_results['scaling_law'] = scaling_law
        return scaling_law


if __name__ == "__main__":
    # Test the unified QG approach
    
    # Create unified approach instance
    unified_qg = UnifiedQGApproach(
        dim_uv=2.0,
        dim_ir=4.0,
        transition_scale=1.0,
        lattice_size=8,  # Small for testing
        beta=1.0,
        coupling=0.1
    )
    
    # Compute effective action across energy scales
    ea_results = unified_qg.compute_effective_action(field_amplitude=0.5, num_scales=4)
    
    # Compute correlation functions across energy scales
    corr_results = unified_qg.compute_multiscale_correlators(num_scales=4, num_samples=200)
    
    # Derive unified scaling law
    scaling_law = unified_qg.derive_unified_scaling_law()
    
    # Plot results
    unified_qg.plot_unified_results(save_path="unified_qg_results.png")
    
    print("\nUnified QG approach test complete.") 