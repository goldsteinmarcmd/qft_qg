"""
Unified Framework for QFT and Quantum Gravity

This module provides a formal bridge between quantum field theory and quantum gravity,
implementing a common mathematical structure with consistent APIs across both frameworks.
It demonstrates how QFT emerges as an approximation from the more fundamental quantum
gravity theory in the appropriate energy limit.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .quantum_spacetime_foundations import SpectralGeometry, DiscreteToContinum
from .path_integral import PathIntegral
from .non_perturbative import AsymptoticallyFreeMethods
from .dimensional_flow_rg import DimensionalFlowRG
from .black_hole_microstates import BlackHoleMicrostates
from .category_qft_connection import CategoryQFTBridge


class UnifiedFramework:
    """
    A unified framework integrating quantum field theory and quantum gravity.
    
    This class provides a consistent mathematical structure for calculations
    that span both QFT and quantum gravity regimes, with appropriate transition
    between them based on energy scale.
    """
    
    def __init__(self, dim_profile=None, coupling_scheme='asymptotic_safety'):
        """
        Initialize the unified framework.
        
        Parameters:
        -----------
        dim_profile : callable, optional
            Function that returns the effective dimension at given energy scale.
            If None, a default profile going from dim=2 (UV) to dim=4 (IR) is used.
        coupling_scheme : str
            Scheme for running couplings: 'asymptotic_safety', 'holographic', etc.
        """
        # Use default dimension profile if none provided
        if dim_profile is None:
            self.dim_profile = lambda E: 4.0 - 2.0 / (1.0 + (E * 0.1)**(-2))
        else:
            self.dim_profile = dim_profile
        
        self.coupling_scheme = coupling_scheme
        
        # Initialize components from both frameworks
        self._initialize_components()
        
        # Store calculation results
        self.results_cache = {}
    
    def _initialize_components(self):
        """Initialize components from both frameworks."""
        # QG components
        self.spectral_geometry = SpectralGeometry(dim=4, size=50)
        self.path_integral = PathIntegral(
            self.dim_profile, 
            lambda E: 1.0 / (1.0 + (E * 0.1)**(-2))
        )
        self.rg_flow = DimensionalFlowRG(
            dim_uv=2.0, 
            dim_ir=4.0, 
            transition_scale=1.0
        )
        
        # Common mathematical bridges
        self.discrete_continuum = DiscreteToContinum(
            dim_uv=2.0,
            dim_ir=4.0,
            transition_scale=1.0
        )
        
        self.category_bridge = CategoryQFTBridge()
        
        # Track last used energy scale
        self.current_energy_scale = 1.0  # Default is Planck scale
    
    def set_energy_scale(self, energy_scale):
        """
        Set the energy scale for subsequent calculations.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Information about the physics at this scale
        """
        self.current_energy_scale = energy_scale
        
        # Get dimension at this scale
        dimension = self.dim_profile(energy_scale)
        
        # Determine regime (QFT or QG dominated)
        if energy_scale < 0.01:  # Below 1% of Planck energy
            regime = "QFT_dominated"
        elif energy_scale > 0.1:  # Above 10% of Planck energy
            regime = "QG_dominated"
        else:
            regime = "transition_zone"
        
        # Cache this information
        scale_info = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'regime': regime
        }
        
        self.results_cache['current_scale'] = scale_info
        return scale_info
    
    def compute_propagator(self, particle_type, momentum, include_qg_corrections=True):
        """
        Compute particle propagator with appropriate QG corrections if needed.
        
        Parameters:
        -----------
        particle_type : str
            Type of particle: 'scalar', 'fermion', 'vector', 'graviton'
        momentum : float or array_like
            Momentum in Planck units
        include_qg_corrections : bool
            Whether to include quantum gravity corrections
            
        Returns:
        --------
        float or array_like
            Propagator value(s)
        """
        # Convert input to numpy array for vectorized calculation
        mom_array = np.atleast_1d(momentum)
        
        # Compute standard QFT propagator
        if particle_type == 'scalar':
            # Standard scalar propagator: 1/(p² + m²)
            mass = 0.0  # Simplified - could be a parameter
            qft_propagator = 1.0 / (mom_array**2 + mass**2)
        elif particle_type == 'fermion':
            # Simplified fermion propagator
            mass = 0.0
            qft_propagator = 1.0 / (mom_array + mass)
        elif particle_type == 'vector':
            # Simplified vector propagator
            qft_propagator = 1.0 / mom_array**2
        elif particle_type == 'graviton':
            # Simplified graviton propagator
            qft_propagator = 1.0 / mom_array**2
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")
        
        if not include_qg_corrections or np.all(mom_array < 0.001):
            # Return standard QFT result for very low energies or if corrections disabled
            result = qft_propagator
        else:
            # Apply QG corrections from path integral module
            corrections = np.zeros_like(mom_array, dtype=float)
            
            for i, p in enumerate(mom_array):
                # Determine correction factor from QG
                if particle_type == 'graviton':
                    correction = self.path_integral._graviton_propagator_correction(
                        p, self.current_energy_scale, self.dim_profile(p)
                    )
                else:
                    correction = self.path_integral._scalar_propagator_correction(
                        p, self.current_energy_scale, self.dim_profile(p)
                    )
                corrections[i] = correction
            
            # Apply corrections to standard QFT result
            result = qft_propagator * corrections
        
        # Return scalar if input was scalar
        if np.isscalar(momentum):
            return result[0]
        else:
            return result
    
    def compute_running_coupling(self, coupling_name, energy_scale=None):
        """
        Compute running coupling constant with QG effects.
        
        Parameters:
        -----------
        coupling_name : str
            Name of coupling: 'alpha_em', 'alpha_s', 'G_newton', etc.
        energy_scale : float, optional
            Energy scale. If None, uses the current scale.
            
        Returns:
        --------
        float
            Value of the coupling constant
        """
        if energy_scale is None:
            energy_scale = self.current_energy_scale
        
        # Get dimension at this scale
        dimension = self.dim_profile(energy_scale)
        
        # Use dimensional flow RG to compute running coupling
        return self.rg_flow.compute_running_coupling(
            coupling_name, energy_scale, dimension
        )
    
    def compute_effective_action(self, energy_scale=None, field_config=None):
        """
        Compute effective action including quantum gravity effects.
        
        Parameters:
        -----------
        energy_scale : float, optional
            Energy scale. If None, uses the current scale.
        field_config : ndarray, optional
            Field configuration to evaluate the action on
            
        Returns:
        --------
        dict
            Effective action results
        """
        if energy_scale is None:
            energy_scale = self.current_energy_scale
        
        # Get effective dimension at this scale
        dimension = self.dim_profile(energy_scale)
        
        # Base field configuration if none provided
        if field_config is None:
            size = 10
            field_config = np.random.randn(size, size) * 0.1
        
        # Determine if we need pure QFT, QG, or mixed treatment
        if energy_scale < 0.001:  # Well below Planck scale - QFT dominates
            # Simple scalar field action
            action_value = np.sum(0.5 * ((np.roll(field_config, 1, axis=0) - field_config)**2 +
                                        (np.roll(field_config, 1, axis=1) - field_config)**2 +
                                        field_config**2))
            
            result = {
                'action_value': action_value,
                'dimension': dimension,
                'energy_scale': energy_scale,
                'regime': 'QFT_dominated',
                'corrections': 0.0
            }
            
        elif energy_scale > 0.1:  # Near/above Planck scale - QG dominates
            # Use path integral module for QG action
            # This is a simplification - we'd need more sophisticated coupling
            result = self.path_integral._compute_effective_action(
                np.mean(field_config**2), 
                np.ones_like(field_config), 
                dimension
            )
            
            result.update({
                'energy_scale': energy_scale,
                'regime': 'QG_dominated'
            })
            
        else:  # Transition zone - mixed QFT and QG
            # QFT contribution
            qft_action = np.sum(0.5 * ((np.roll(field_config, 1, axis=0) - field_config)**2 +
                                       (np.roll(field_config, 1, axis=1) - field_config)**2 +
                                       field_config**2))
            
            # QG correction
            qg_factor = energy_scale**2 * (4.0 / dimension)**(dimension / 2)
            qg_correction = qg_factor * np.sum(field_config**4)  # Simplified QG correction
            
            # Total action
            total_action = qft_action + qg_correction
            
            result = {
                'action_value': total_action,
                'qft_contribution': qft_action,
                'qg_correction': qg_correction,
                'dimension': dimension,
                'energy_scale': energy_scale,
                'regime': 'transition_zone'
            }
        
        return result
    
    def qft_to_qg_mapping(self, qft_object, target_energy_scale=None):
        """
        Map a QFT object to its QG representation.
        
        Parameters:
        -----------
        qft_object : dict
            QFT object representation
        target_energy_scale : float, optional
            Target energy scale for mapping
            
        Returns:
        --------
        dict
            QG representation of the object
        """
        if target_energy_scale is None:
            target_energy_scale = self.current_energy_scale
        
        # Use category theory bridge to map between frameworks
        return self.category_bridge.map_qft_to_qg(
            qft_object, 
            target_energy_scale,
            self.dim_profile(target_energy_scale)
        )
    
    def qg_to_qft_mapping(self, qg_object, target_energy_scale=0.001):
        """
        Map a QG object to its QFT representation.
        
        Parameters:
        -----------
        qg_object : dict
            QG object representation
        target_energy_scale : float, optional
            Target energy scale for mapping
            
        Returns:
        --------
        dict
            QFT representation of the object
        """
        # Use category theory bridge to map between frameworks
        return self.category_bridge.map_qg_to_qft(
            qg_object, 
            target_energy_scale,
            self.dim_profile(target_energy_scale)
        )
    
    def compute_transition_amplitudes(self, initial_state, final_state, energy_scale=None):
        """
        Compute transition amplitudes using appropriate framework.
        
        Parameters:
        -----------
        initial_state : dict
            Initial state specification
        final_state : dict
            Final state specification
        energy_scale : float, optional
            Energy scale. If None, uses the current scale.
            
        Returns:
        --------
        dict
            Transition amplitude results
        """
        if energy_scale is None:
            energy_scale = self.current_energy_scale
        
        # Get effective dimension
        dimension = self.dim_profile(energy_scale)
        
        # Determine appropriate framework based on energy scale
        if energy_scale < 0.001:  # QFT dominated
            # Use QFT methods (simplified)
            amplitude = np.exp(-abs(initial_state.get('energy', 1.0) - 
                               final_state.get('energy', 0.0))**2)
            
            result = {
                'amplitude': amplitude,
                'framework': 'QFT',
                'dimension': dimension,
                'energy_scale': energy_scale
            }
            
        elif energy_scale > 0.1:  # QG dominated
            # Use path integral with QG effects
            amplitude = self.path_integral.compute_scattering_amplitude(
                '2to2', energy_scale, 
                [initial_state.get('particle_type', 'scalar'), 
                 final_state.get('particle_type', 'scalar')]
            )
            
            # Extract amplitude value
            amplitude_value = amplitude['total_amplitude']
            
            result = {
                'amplitude': amplitude_value,
                'cross_section': amplitude.get('cross_section', 0.0),
                'framework': 'QG',
                'dimension': dimension,
                'energy_scale': energy_scale
            }
            
        else:  # Transition zone
            # Compute both and combine
            qft_amp = np.exp(-abs(initial_state.get('energy', 1.0) - 
                             final_state.get('energy', 0.0))**2)
            
            qg_amp_result = self.path_integral.compute_scattering_amplitude(
                '2to2', energy_scale, 
                [initial_state.get('particle_type', 'scalar'), 
                 final_state.get('particle_type', 'scalar')]
            )
            qg_amp = qg_amp_result['total_amplitude']
            
            # Interpolate based on energy scale
            interp_factor = (energy_scale - 0.001) / (0.1 - 0.001)
            combined_amp = qft_amp * (1 - interp_factor) + qg_amp * interp_factor
            
            result = {
                'amplitude': combined_amp,
                'qft_contribution': qft_amp,
                'qg_contribution': qg_amp,
                'framework': 'hybrid',
                'dimension': dimension,
                'energy_scale': energy_scale
            }
        
        return result
    
    def demonstrate_qft_emergence(self, energy_range=None, num_points=20):
        """
        Demonstrate how QFT emerges from QG at low energies.
        
        Parameters:
        -----------
        energy_range : tuple, optional
            (min_scale, max_scale) in Planck units
        num_points : int
            Number of energy points to compute
            
        Returns:
        --------
        dict
            Results demonstrating emergence
        """
        if energy_range is None:
            energy_range = (1e-6, 1.0)  # From well below Planck scale to Planck scale
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(
            np.log10(energy_range[0]),
            np.log10(energy_range[1]),
            num_points
        )
        
        # Compute dimensions at each scale
        dimensions = np.array([self.dim_profile(e) for e in energy_scales])
        
        # Compute propagator for scalar particle
        propagators_qft = np.zeros(num_points)
        propagators_qg = np.zeros(num_points)
        propagators_unified = np.zeros(num_points)
        
        # Test momentum
        test_momentum = 0.01  # Well below Planck scale
        
        for i, scale in enumerate(energy_scales):
            self.set_energy_scale(scale)
            
            # Standard QFT propagator (without QG corrections)
            propagators_qft[i] = 1.0 / (test_momentum**2)
            
            # Pure QG calculation from path integral
            qg_correction = self.path_integral._scalar_propagator_correction(
                test_momentum, scale, dimensions[i]
            )
            propagators_qg[i] = qg_correction / (test_momentum**2)
            
            # Unified framework calculation
            propagators_unified[i] = self.compute_propagator(
                'scalar', test_momentum, include_qg_corrections=True
            )
        
        # Calculate relative deviation
        deviation = np.abs(propagators_unified - propagators_qft) / propagators_qft
        
        # Compare effective actions
        action_qft = np.zeros(num_points)
        action_qg = np.zeros(num_points)
        action_unified = np.zeros(num_points)
        
        # Simple field configuration for testing
        test_field = np.ones((4, 4)) * 0.1
        
        for i, scale in enumerate(energy_scales):
            self.set_energy_scale(scale)
            
            # Get actions
            effective_action = self.compute_effective_action(scale, test_field)
            action_unified[i] = effective_action['action_value']
            
            if 'qft_contribution' in effective_action:
                action_qft[i] = effective_action['qft_contribution']
                action_qg[i] = effective_action['qg_correction']
            elif effective_action['regime'] == 'QFT_dominated':
                action_qft[i] = effective_action['action_value']
                action_qg[i] = 0.0
            else:
                action_qft[i] = 0.0
                action_qg[i] = effective_action['action_value']
        
        return {
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'propagators': {
                'qft': propagators_qft,
                'qg': propagators_qg,
                'unified': propagators_unified,
                'deviation': deviation
            },
            'actions': {
                'qft': action_qft,
                'qg': action_qg,
                'unified': action_unified
            }
        }
    
    def visualize_qft_emergence(self, emergence_results=None):
        """
        Visualize how QFT emerges from QG at low energies.
        
        Parameters:
        -----------
        emergence_results : dict, optional
            Results from demonstrate_qft_emergence. If None, will compute.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with visualizations
        """
        if emergence_results is None:
            emergence_results = self.demonstrate_qft_emergence()
        
        # Extract data
        energy_scales = emergence_results['energy_scales']
        dimensions = emergence_results['dimensions']
        
        propagators_qft = emergence_results['propagators']['qft']
        propagators_qg = emergence_results['propagators']['qg']
        propagators_unified = emergence_results['propagators']['unified']
        deviation = emergence_results['propagators']['deviation']
        
        action_qft = emergence_results['actions']['qft']
        action_qg = emergence_results['actions']['qg']
        action_unified = emergence_results['actions']['unified']
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot dimension vs energy scale
        axes[0, 0].semilogx(energy_scales, dimensions, 'k-', linewidth=2)
        axes[0, 0].set_xlabel('Energy Scale (Planck Units)')
        axes[0, 0].set_ylabel('Effective Dimension')
        axes[0, 0].set_title('Dimensional Flow')
        axes[0, 0].grid(True)
        
        # Plot propagators
        axes[0, 1].loglog(energy_scales, propagators_qft, 'b--', label='QFT')
        axes[0, 1].loglog(energy_scales, propagators_qg, 'r--', label='QG')
        axes[0, 1].loglog(energy_scales, propagators_unified, 'g-', label='Unified')
        axes[0, 1].set_xlabel('Energy Scale (Planck Units)')
        axes[0, 1].set_ylabel('Propagator Value')
        axes[0, 1].set_title('Scalar Propagator')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Plot deviation from QFT
        axes[1, 0].loglog(energy_scales, deviation, 'k-', linewidth=2)
        axes[1, 0].set_xlabel('Energy Scale (Planck Units)')
        axes[1, 0].set_ylabel('Relative Deviation')
        axes[1, 0].set_title('Deviation from Standard QFT')
        axes[1, 0].grid(True)
        
        # Mark transition zones
        axes[1, 0].axvspan(0.001, 0.1, alpha=0.2, color='gray')
        axes[1, 0].text(0.01, np.min(deviation)*2, 'Transition Zone', ha='center')
        
        # Plot effective actions
        axes[1, 1].loglog(energy_scales, action_qft, 'b--', label='QFT Contribution')
        axes[1, 1].loglog(energy_scales, action_qg, 'r--', label='QG Contribution')
        axes[1, 1].loglog(energy_scales, action_unified, 'g-', label='Unified Action')
        axes[1, 1].set_xlabel('Energy Scale (Planck Units)')
        axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].set_title('Effective Action Components')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig


# Example usage if this module is run directly
if __name__ == "__main__":
    # Create unified framework
    unified = UnifiedFramework()
    
    # Demonstrate QFT emergence
    emergence_results = unified.demonstrate_qft_emergence(
        energy_range=(1e-6, 1.0), num_points=30
    )
    
    # Visualize results
    fig = unified.visualize_qft_emergence(emergence_results)
    plt.savefig("qft_emergence_from_qg.png")
    print("Visualization saved to qft_emergence_from_qg.png") 