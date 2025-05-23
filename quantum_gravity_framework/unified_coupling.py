"""
Unified Coupling Framework

This module implements a comprehensive framework connecting all fundamental forces
with mechanisms for coupling unification at high energies, incorporating dimensional
flow effects.
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

class UnifiedCouplingFramework:
    """
    Framework for unifying fundamental interactions across energy scales.
    
    This class implements running coupling constants with quantum gravity effects
    and dimensional flow, facilitating the investigation of force unification.
    """
    
    def __init__(self, dim_profile):
        """
        Initialize unified coupling framework.
        
        Parameters:
        -----------
        dim_profile : callable
            Function that returns dimension as a function of energy scale
        """
        self.dim_profile = dim_profile
        
        # Standard model couplings at MZ (91.1876 GeV)
        # alpha_i = g_i^2/(4*pi)
        self.alpha_em_mz = 1/127.916  # Electromagnetic
        self.alpha_s_mz = 0.1179      # Strong
        self.alpha_w_mz = 0.03493     # Weak
        
        # Gravitational coupling (G_N * MZ^2) - extremely small at MZ
        self.alpha_g_mz = 6.707e-33
        
        # Energy scales in Planck units
        self.mz_scale = 91.1876 / (1.22e19)  # MZ/M_Planck
        self.gut_scale_estimate = 1e16 / (1.22e19)  # GUT scale / M_Planck
        
        # Beta function parameters
        self._initialize_beta_functions()
        
        # Caching for efficiency
        self.coupling_cache = {}
    
    def _initialize_beta_functions(self):
        """Initialize parameters for beta functions of coupling constants."""
        # SM beta function coefficients (one-loop approximation)
        # For standard RG flow in 4D: dg/dt = -b0/(16π²)g³
        
        # For alpha_1 = (5/3)α_Y (U(1))
        self.b0_1 = -41/10
        
        # For alpha_2 (SU(2))
        self.b0_2 = 19/6
        
        # For alpha_3 (SU(3))
        self.b0_3 = 7
        
        # Gravitational beta function (simplified)
        self.b0_g = 2
        
        # Dimension-dependent corrections
        self.dim_correction = lambda d: (4/d)**1.5
        
        # Unification threshold effects
        self.thresholds = {
            # SUSY scale: particles entering the theory
            'susy': {
                'scale': 1e3 / (1.22e19),
                'delta_b1': -1/2,
                'delta_b2': -1/6,
                'delta_b3': -3,
                'enabled': True
            },
            # Intermediate symmetry breaking scale
            'intermediate': {
                'scale': 1e14 / (1.22e19),
                'delta_b1': -10/3,
                'delta_b2': -2/3,
                'delta_b3': 0,
                'enabled': False
            }
        }
    
    def run_sm_couplings(self, energy_scale):
        """
        Calculate SM couplings at given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Dictionary of coupling values
        """
        # Check if previously calculated
        cache_key = f"sm_{energy_scale:.8e}"
        if cache_key in self.coupling_cache:
            return self.coupling_cache[cache_key]
        
        # Get dimension at this scale
        dim = self.dim_profile(energy_scale)
        
        # Energy ratio for standard log running
        t = np.log(energy_scale / self.mz_scale)
        
        # Dimension-dependent correction factor
        dim_factor = self.dim_correction(dim)
        
        # Apply threshold corrections to beta functions
        b1 = self.b0_1
        b2 = self.b0_2
        b3 = self.b0_3
        
        for threshold_name, threshold in self.thresholds.items():
            if threshold['enabled'] and energy_scale > threshold['scale']:
                b1 += threshold['delta_b1']
                b2 += threshold['delta_b2']
                b3 += threshold['delta_b3']
        
        # One-loop running equations with dimension-dependent scaling
        alpha_1 = 1 / ((1 / self.alpha_em_mz) * (5/3) - (b1 * dim_factor * t) / (2*np.pi))
        alpha_2 = 1 / ((1 / self.alpha_w_mz) - (b2 * dim_factor * t) / (2*np.pi))
        alpha_3 = 1 / ((1 / self.alpha_s_mz) - (b3 * dim_factor * t) / (2*np.pi))
        
        # Normalize alpha_1 back to alpha_em (fine structure constant)
        alpha_em = alpha_1 * (3/5)
        
        # Gravitational coupling with modified running due to dimensional flow
        if energy_scale > 0.01:  # Near Planck scale, non-perturbative effects dominate
            # Asymptotic safety inspired behavior
            alpha_g = self.asymptotic_safety_g(energy_scale, dim)
        else:
            # Standard log running at lower energies
            alpha_g = self.alpha_g_mz + (self.b0_g * (dim - 4) * t) / (2*np.pi)
        
        # Ensure physically meaningful values (positive and not too large)
        results = {
            'alpha_1': max(0, min(alpha_1, 10)),
            'alpha_2': max(0, min(alpha_2, 10)),
            'alpha_3': max(0, min(alpha_3, 10)),
            'alpha_em': max(0, min(alpha_em, 10)),
            'alpha_g': max(0, min(alpha_g, 10)),
            'dimension': dim
        }
        
        # Cache results
        self.coupling_cache[cache_key] = results
        
        return results
    
    def asymptotic_safety_g(self, energy_scale, dim):
        """
        Calculate gravitational coupling using asymptotic safety principles.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        dim : float
            Effective dimension at this scale
            
        Returns:
        --------
        float
            Gravitational coupling
        """
        # Fixed point value (approximate)
        g_fixed = 0.4 * (4 / dim)
        
        # Define transition function
        trans_scale = 0.1  # Transition around 0.1 M_Planck
        transition = 1 / (1 + (trans_scale / energy_scale)**2)
        
        # Combine fixed point with low-energy behavior
        alpha_g = self.alpha_g_mz * (1 - transition) + g_fixed * transition
        
        return alpha_g
    
    def compute_unification_scale(self, coupling_pairs=None, precision=100):
        """
        Find energy scale where specified couplings unify.
        
        Parameters:
        -----------
        coupling_pairs : list, optional
            List of coupling pairs to check, e.g. [('alpha_1', 'alpha_2')]
        precision : int
            Number of energy points to check
            
        Returns:
        --------
        dict
            Unification results
        """
        if coupling_pairs is None:
            # Default: check standard GUT unification
            coupling_pairs = [('alpha_1', 'alpha_2'), ('alpha_2', 'alpha_3')]
        
        # Sample energy scales logarithmically from MZ to above Planck scale
        energy_scales = np.logspace(
            np.log10(self.mz_scale),
            np.log10(10.0),  # Up to 10 M_Planck
            precision
        )
        
        # Compute couplings at each scale
        coupling_values = {}
        dimensions = []
        
        print("Computing coupling constants across energy scales...")
        for scale in energy_scales:
            couplings = self.run_sm_couplings(scale)
            dimensions.append(couplings['dimension'])
            
            for coupling, value in couplings.items():
                if coupling not in coupling_values:
                    coupling_values[coupling] = []
                coupling_values[coupling].append(value)
        
        # Find unification points
        unification_points = []
        
        for couple in coupling_pairs:
            if len(couple) != 2:
                continue
                
            c1, c2 = couple
            if c1 not in coupling_values or c2 not in coupling_values:
                continue
                
            # Calculate differences between couplings
            diffs = np.abs(np.array(coupling_values[c1]) - np.array(coupling_values[c2]))
            
            # Find local minima in differences
            minima_indices = []
            for i in range(1, len(diffs)-1):
                if diffs[i] < diffs[i-1] and diffs[i] < diffs[i+1]:
                    minima_indices.append(i)
            
            # Add endpoints if they're minima
            if diffs[0] < diffs[1]:
                minima_indices.insert(0, 0)
            if diffs[-1] < diffs[-2]:
                minima_indices.append(len(diffs)-1)
            
            # Get scales and differences at minima
            for idx in minima_indices:
                if diffs[idx] < 0.05:  # Threshold for considering "unified"
                    point = {
                        'couplings': couple,
                        'scale': energy_scales[idx],
                        'dimension': dimensions[idx],
                        'value': (coupling_values[c1][idx] + coupling_values[c2][idx]) / 2,
                        'difference': diffs[idx]
                    }
                    unification_points.append(point)
        
        # Find common unification scale if multiple pairs unify
        common_unification = None
        if len(unification_points) > 1 and len(coupling_pairs) > 1:
            # Check if all pairs unify at similar scales
            scales = [p['scale'] for p in unification_points]
            min_scale, max_scale = min(scales), max(scales)
            
            if max_scale / min_scale < 10:
                # Close enough to consider common unification
                avg_scale = np.exp(np.mean(np.log(scales)))
                avg_dim = np.mean([p['dimension'] for p in unification_points])
                avg_diff = np.mean([p['difference'] for p in unification_points])
                
                common_unification = {
                    'scale': avg_scale,
                    'dimension': avg_dim,
                    'difference': avg_diff,
                    'couplings': [p['couplings'] for p in unification_points]
                }
        
        # Compile results
        results = {
            'energy_scales': energy_scales,
            'coupling_values': coupling_values,
            'dimensions': dimensions,
            'unification_points': unification_points,
            'common_unification': common_unification
        }
        
        return results
    
    def compute_dimensional_flow_effects(self, energy_range=None, num_points=50):
        """
        Analyze how dimensional flow affects coupling unification.
        
        Parameters:
        -----------
        energy_range : tuple, optional
            (min_scale, max_scale) in Planck units
        num_points : int
            Number of energy points to check
            
        Returns:
        --------
        dict
            Analysis results
        """
        if energy_range is None:
            # Default: from well below to above Planck scale
            energy_range = (self.mz_scale, 10.0)
        
        # Sample energy scales logarithmically
        energy_scales = np.logspace(
            np.log10(energy_range[0]),
            np.log10(energy_range[1]),
            num_points
        )
        
        # Get dimension profile
        dimensions = np.array([self.dim_profile(e) for e in energy_scales])
        
        # Compute coupling parameters with and without dimensional flow
        couplings_with_dimflow = []
        couplings_without_dimflow = []
        
        # Store original dimension profile
        original_dim_profile = self.dim_profile
        
        print("Computing couplings with dimensional flow...")
        for scale in energy_scales:
            couplings_with_dimflow.append(self.run_sm_couplings(scale))
        
        # Replace with constant 4D profile
        print("Computing couplings without dimensional flow...")
        self.dim_profile = lambda e: 4.0
        
        # Clear cache
        self.coupling_cache = {}
        
        for scale in energy_scales:
            couplings_without_dimflow.append(self.run_sm_couplings(scale))
        
        # Restore original profile
        self.dim_profile = original_dim_profile
        
        # Clear cache again
        self.coupling_cache = {}
        
        # Extract coupling values
        alpha_1_with = [c['alpha_1'] for c in couplings_with_dimflow]
        alpha_2_with = [c['alpha_2'] for c in couplings_with_dimflow]
        alpha_3_with = [c['alpha_3'] for c in couplings_with_dimflow]
        alpha_g_with = [c['alpha_g'] for c in couplings_with_dimflow]
        
        alpha_1_without = [c['alpha_1'] for c in couplings_without_dimflow]
        alpha_2_without = [c['alpha_2'] for c in couplings_without_dimflow]
        alpha_3_without = [c['alpha_3'] for c in couplings_without_dimflow]
        alpha_g_without = [c['alpha_g'] for c in couplings_without_dimflow]
        
        # Calculate intersection points
        unification_with = self._find_intersection_points(
            energy_scales, alpha_1_with, alpha_2_with, alpha_3_with
        )
        
        unification_without = self._find_intersection_points(
            energy_scales, alpha_1_without, alpha_2_without, alpha_3_without
        )
        
        # Return results
        return {
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'with_dimflow': {
                'alpha_1': alpha_1_with,
                'alpha_2': alpha_2_with,
                'alpha_3': alpha_3_with,
                'alpha_g': alpha_g_with,
                'unification': unification_with
            },
            'without_dimflow': {
                'alpha_1': alpha_1_without,
                'alpha_2': alpha_2_without,
                'alpha_3': alpha_3_without,
                'alpha_g': alpha_g_without,
                'unification': unification_without
            }
        }
    
    def _find_intersection_points(self, scales, alpha_1, alpha_2, alpha_3):
        """
        Find intersection points between coupling curves.
        
        Parameters:
        -----------
        scales : array
            Energy scales
        alpha_1, alpha_2, alpha_3 : array
            Coupling values at each scale
            
        Returns:
        --------
        dict
            Intersection points
        """
        # Find where alpha_1 ≈ alpha_2
        intersect_12 = []
        for i in range(1, len(scales)):
            diff_prev = alpha_1[i-1] - alpha_2[i-1]
            diff_curr = alpha_1[i] - alpha_2[i]
            
            if diff_prev * diff_curr <= 0:  # Sign change means intersection
                # Interpolate to find intersection point
                t = -diff_prev / (diff_curr - diff_prev)
                scale = scales[i-1] * (scales[i]/scales[i-1])**t
                value = alpha_1[i-1] + t * (alpha_1[i] - alpha_1[i-1])
                
                intersect_12.append({
                    'scale': scale,
                    'value': value
                })
        
        # Similarly for alpha_2 ≈ alpha_3
        intersect_23 = []
        for i in range(1, len(scales)):
            diff_prev = alpha_2[i-1] - alpha_3[i-1]
            diff_curr = alpha_2[i] - alpha_3[i]
            
            if diff_prev * diff_curr <= 0:
                t = -diff_prev / (diff_curr - diff_prev)
                scale = scales[i-1] * (scales[i]/scales[i-1])**t
                value = alpha_2[i-1] + t * (alpha_2[i] - alpha_2[i-1])
                
                intersect_23.append({
                    'scale': scale,
                    'value': value
                })
        
        # And for alpha_1 ≈ alpha_3
        intersect_13 = []
        for i in range(1, len(scales)):
            diff_prev = alpha_1[i-1] - alpha_3[i-1]
            diff_curr = alpha_1[i] - alpha_3[i]
            
            if diff_prev * diff_curr <= 0:
                t = -diff_prev / (diff_curr - diff_prev)
                scale = scales[i-1] * (scales[i]/scales[i-1])**t
                value = alpha_1[i-1] + t * (alpha_1[i] - alpha_1[i-1])
                
                intersect_13.append({
                    'scale': scale,
                    'value': value
                })
        
        # Check if we have triple unification
        triple_unification = []
        for i12 in intersect_12:
            for i23 in intersect_23:
                # Check if the scales are close (within 20%)
                if 0.8 <= i12['scale']/i23['scale'] <= 1.25:
                    # Check if the values are close (within 10%)
                    if 0.9 <= i12['value']/i23['value'] <= 1.1:
                        triple_unification.append({
                            'scale': (i12['scale'] + i23['scale'])/2,
                            'value': (i12['value'] + i23['value'])/2
                        })
        
        return {
            'alpha_1_alpha_2': intersect_12,
            'alpha_2_alpha_3': intersect_23,
            'alpha_1_alpha_3': intersect_13,
            'triple_unification': triple_unification
        }
    
    def run_unified_model(self, model_name='minimal_gut', energy_scale=1.0):
        """
        Compute couplings in a specific unified model at given energy.
        
        Parameters:
        -----------
        model_name : str
            Name of the unified model
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Unified model results
        """
        # Get dimension at this scale
        dim = self.dim_profile(energy_scale)
        
        # Basic couplings
        sm_couplings = self.run_sm_couplings(energy_scale)
        
        if model_name == 'minimal_gut':
            # SU(5) minimal GUT
            if energy_scale >= 0.001:  # Above GUT scale
                # Unified coupling
                alpha_gut = (sm_couplings['alpha_1'] + sm_couplings['alpha_2'] + sm_couplings['alpha_3']) / 3
                
                # Apply dimension-dependent corrections
                dim_factor = (4/dim)**2
                alpha_gut *= dim_factor
                
                # Estimated mass scales
                x_boson_mass = 1e-3 * (energy_scale)**0.25
                
                return {
                    'alpha_gut': alpha_gut,
                    'dimension': dim,
                    'energy': energy_scale,
                    'x_boson_mass': x_boson_mass,
                    'model': 'SU(5) minimal GUT'
                }
            else:
                # Below GUT scale, return SM couplings
                return {
                    **sm_couplings,
                    'model': 'SM (below GUT scale)'
                }
                
        elif model_name == 'so10':
            # SO(10) model
            if energy_scale >= 0.005:  # Above SO(10) scale
                # Different unified coupling
                alpha_gut = (sm_couplings['alpha_1'] + sm_couplings['alpha_2'] + sm_couplings['alpha_3']) / 3
                
                # Apply dimension-dependent corrections
                dim_factor = (4/dim)**1.8
                alpha_gut *= dim_factor
                
                # Estimated mass scales
                x_boson_mass = 5e-3 * (energy_scale)**0.2
                
                return {
                    'alpha_gut': alpha_gut,
                    'dimension': dim,
                    'energy': energy_scale,
                    'x_boson_mass': x_boson_mass,
                    'model': 'SO(10) GUT'
                }
            else:
                # Below GUT scale, return SM couplings
                return {
                    **sm_couplings,
                    'model': 'SM (below SO(10) scale)'
                }
        
        elif model_name == 'e6':
            # E6 model
            if energy_scale >= 0.01:  # Above E6 scale
                # Different unified coupling
                alpha_gut = (sm_couplings['alpha_1'] + sm_couplings['alpha_2'] + sm_couplings['alpha_3']) / 3
                
                # Apply dimension-dependent corrections
                dim_factor = (4/dim)**1.5
                alpha_gut *= dim_factor
                
                # Gravity coupling near unification
                alpha_g = sm_couplings['alpha_g']
                
                # Check if close to gravitational unification
                gravity_unified = (abs(alpha_gut - alpha_g) / alpha_gut) < 0.2
                
                return {
                    'alpha_gut': alpha_gut,
                    'alpha_g': alpha_g,
                    'dimension': dim,
                    'energy': energy_scale,
                    'gravity_unified': gravity_unified,
                    'model': 'E6 GUT'
                }
            else:
                # Below GUT scale, return SM couplings
                return {
                    **sm_couplings,
                    'model': 'SM (below E6 scale)'
                }
        
        else:
            # Unknown model, return SM couplings
            return sm_couplings
    
    def run_theory_of_everything(self, energy_scale):
        """
        Compute parameters in a theory of everything scenario at given energy.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        dict
            Theory of everything predictions
        """
        # In a TOE, all forces unify, including gravity
        dim = self.dim_profile(energy_scale)
        
        # Get standard model couplings
        sm_couplings = self.run_sm_couplings(energy_scale)
        
        # Proximity to Planck scale
        planck_proximity = energy_scale
        
        # In true TOE regime, dimension approaches the UV fixed point
        uv_dim = 2.0  # Typical UV dimension in many QG approaches
        
        # Measure of dimensional flow effect
        dim_flow_factor = abs(dim - 4) / abs(uv_dim - 4)
        
        # For TOE, we expect all couplings to approach a common value
        if energy_scale > 0.5:  # Near Planck scale
            # Estimate unified coupling value with dimensional flow effects
            unified_coupling = 0.025 * (dim/4)**(-2)
            coupling_spread = 0.001 * energy_scale
            
            # Generate unified couplings with small fluctuations
            toe_couplings = {
                'alpha_1': unified_coupling * (1 + np.random.normal(0, coupling_spread)),
                'alpha_2': unified_coupling * (1 + np.random.normal(0, coupling_spread)),
                'alpha_3': unified_coupling * (1 + np.random.normal(0, coupling_spread)),
                'alpha_g': unified_coupling * (1 + np.random.normal(0, coupling_spread)),
                'alpha_unified': unified_coupling,
                'dimension': dim,
                'energy': energy_scale,
                'dim_flow_factor': dim_flow_factor,
                'planck_proximity': planck_proximity,
                'model': 'Theory of Everything'
            }
            
            return toe_couplings
        else:
            # Below Planck scale, use running couplings
            # but show progress toward unification
            unification_progress = energy_scale**2
            
            return {
                **sm_couplings,
                'dim_flow_factor': dim_flow_factor,
                'planck_proximity': planck_proximity,
                'unification_progress': unification_progress,
                'model': 'Approaching TOE'
            }
    
    def visualize_coupling_flow(self, results=None):
        """
        Visualize running couplings and dimensional flow.
        
        Parameters:
        -----------
        results : dict, optional
            Results from compute_dimensional_flow_effects
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if results is None:
            # Compute if not provided
            results = self.compute_dimensional_flow_effects()
        
        # Extract data
        energy_scales = results['energy_scales']
        dimensions = results['dimensions']
        
        # With dimensional flow
        alpha_1_with = results['with_dimflow']['alpha_1']
        alpha_2_with = results['with_dimflow']['alpha_2']
        alpha_3_with = results['with_dimflow']['alpha_3']
        alpha_g_with = results['with_dimflow']['alpha_g']
        
        # Without dimensional flow
        alpha_1_without = results['without_dimflow']['alpha_1']
        alpha_2_without = results['without_dimflow']['alpha_2']
        alpha_3_without = results['without_dimflow']['alpha_3']
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot couplings with dimensional flow
        ax1 = fig.add_subplot(221)
        ax1.loglog(energy_scales, alpha_1_with, 'b-', label='α₁')
        ax1.loglog(energy_scales, alpha_2_with, 'g-', label='α₂')
        ax1.loglog(energy_scales, alpha_3_with, 'r-', label='α₃')
        ax1.loglog(energy_scales, alpha_g_with, 'k-', label='αᵍ')
        ax1.set_xlabel('Energy (M_Planck)')
        ax1.set_ylabel('Coupling α')
        ax1.set_title('Running Couplings with Dimensional Flow')
        ax1.grid(True)
        ax1.legend()
        
        # Plot couplings without dimensional flow
        ax2 = fig.add_subplot(222)
        ax2.loglog(energy_scales, alpha_1_without, 'b--', label='α₁')
        ax2.loglog(energy_scales, alpha_2_without, 'g--', label='α₂')
        ax2.loglog(energy_scales, alpha_3_without, 'r--', label='α₃')
        ax2.set_xlabel('Energy (M_Planck)')
        ax2.set_ylabel('Coupling α')
        ax2.set_title('Standard Running Couplings (4D only)')
        ax2.grid(True)
        ax2.legend()
        
        # Plot dimension vs energy
        ax3 = fig.add_subplot(223)
        ax3.semilogx(energy_scales, dimensions, 'k-', linewidth=2)
        ax3.set_xlabel('Energy (M_Planck)')
        ax3.set_ylabel('Effective Dimension')
        ax3.set_title('Dimensional Flow')
        ax3.grid(True)
        
        # Plot comparing unification with/without dimensional flow
        ax4 = fig.add_subplot(224)
        
        # With dimensional flow
        ax4.loglog(energy_scales, alpha_1_with, 'b-', label='α₁ (dim flow)')
        ax4.loglog(energy_scales, alpha_2_with, 'g-', label='α₂ (dim flow)')
        ax4.loglog(energy_scales, alpha_3_with, 'r-', label='α₃ (dim flow)')
        
        # Without dimensional flow
        ax4.loglog(energy_scales, alpha_1_without, 'b--', label='α₁ (standard)')
        ax4.loglog(energy_scales, alpha_2_without, 'g--', label='α₂ (standard)')
        ax4.loglog(energy_scales, alpha_3_without, 'r--', label='α₃ (standard)')
        
        # Add unification points if found
        with_unif = results['with_dimflow']['unification']
        without_unif = results['without_dimflow']['unification']
        
        # Plot triple unification points if found
        if with_unif['triple_unification']:
            for point in with_unif['triple_unification']:
                ax4.plot(point['scale'], point['value'], 'ko', markersize=10, 
                       label='Unification (dim flow)')
                
        if without_unif['triple_unification']:
            for point in without_unif['triple_unification']:
                ax4.plot(point['scale'], point['value'], 'k^', markersize=10, 
                       label='Unification (standard)')
        
        ax4.set_xlabel('Energy (M_Planck)')
        ax4.set_ylabel('Coupling α')
        ax4.set_title('Comparison of Unification Scenarios')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Define a dimension profile for testing
    dim_profile = lambda E: 4.0 - 2.0 / (1 + (E * 0.1)**(-2))
    
    # Create unified coupling framework
    ucf = UnifiedCouplingFramework(dim_profile)
    
    # Test running couplings at different scales
    print("\nRunning coupling constants at different energy scales:")
    test_scales = [ucf.mz_scale, 1e-10, 1e-6, 1e-3, 0.1, 1.0]
    
    for scale in test_scales:
        couplings = ucf.run_sm_couplings(scale)
        print(f"E = {scale:.3e} M_Planck, dimension = {couplings['dimension']:.3f}")
        print(f"  α₁ = {couplings['alpha_1']:.6f}, α₂ = {couplings['alpha_2']:.6f}, " 
             f"α₃ = {couplings['alpha_3']:.6f}, αᵍ = {couplings['alpha_g']:.6e}")
    
    # Test unification scale computation
    unif_results = ucf.compute_unification_scale()
    
    print("\nUnification analysis:")
    if unif_results['common_unification']:
        unif = unif_results['common_unification']
        print(f"Common unification found at E ≈ {unif['scale']:.3e} M_Planck")
        print(f"  Dimension at unification: {unif['dimension']:.3f}")
        print(f"  Average difference: {unif['difference']:.6f}")
    else:
        print("No common unification found")
        
        if unif_results['unification_points']:
            print("Individual unification points:")
            for point in unif_results['unification_points']:
                print(f"  {point['couplings'][0]}-{point['couplings'][1]} at " 
                     f"E = {point['scale']:.3e}, value = {point['value']:.6f}")
    
    # Compute dimensional flow effects
    flow_results = ucf.compute_dimensional_flow_effects()
    
    print("\nDimensional flow effects on unification:")
    
    # With dimensional flow
    with_unif = flow_results['with_dimflow']['unification']
    if with_unif['triple_unification']:
        print("Triple unification WITH dimensional flow:")
        for point in with_unif['triple_unification']:
            print(f"  Scale: {point['scale']:.3e} M_Planck, Value: {point['value']:.6f}")
    
    # Without dimensional flow
    without_unif = flow_results['without_dimflow']['unification']
    if without_unif['triple_unification']:
        print("Triple unification WITHOUT dimensional flow:")
        for point in without_unif['triple_unification']:
            print(f"  Scale: {point['scale']:.3e} M_Planck, Value: {point['value']:.6f}")
    
    # Visualize results
    fig = ucf.visualize_coupling_flow(flow_results)
    plt.savefig("unified_coupling_visualization.png")
    print("\nVisualization saved to unified_coupling_visualization.png") 