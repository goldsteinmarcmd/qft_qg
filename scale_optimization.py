#!/usr/bin/env python
"""
Scale Parameter Optimization for Quantum Gravity Framework

This module optimizes QG scale parameters to produce detectable experimental signatures,
building on the existing experimental predictions framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import warnings

# Import existing framework components
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class ScaleParameterOptimizer:
    """
    Optimizes QG scale parameters to produce detectable experimental signatures.
    
    This class builds on the existing experimental predictions framework to find
    optimal scale parameters that maximize experimental detectability while maintaining
    theoretical consistency.
    """
    
    def __init__(self, target_significance: float = 5.0, 
                 experimental_limits: Optional[Dict] = None):
        """
        Initialize the scale parameter optimizer.
        
        Parameters:
        -----------
        target_significance : float
            Target significance level for detection (default: 5.0 sigma)
        experimental_limits : dict, optional
            Dictionary of experimental facility limits and sensitivities
        """
        self.target_significance = target_significance
        
        # Default experimental limits if not provided
        if experimental_limits is None:
            self.experimental_limits = {
                'LHC_Run3': {
                    'energy': 13.6e3,  # GeV
                    'luminosity': 300.0,  # fb^-1
                    'sensitivity': 0.05,  # 5% relative uncertainty
                    'systematic': 0.03,  # 3% systematic uncertainty
                    'statistical': 0.04   # 4% statistical uncertainty
                },
                'HL_LHC': {
                    'energy': 14.0e3,  # GeV
                    'luminosity': 3000.0,  # fb^-1
                    'sensitivity': 0.02,  # 2% relative uncertainty
                    'systematic': 0.015,  # 1.5% systematic uncertainty
                    'statistical': 0.012  # 1.2% statistical uncertainty
                },
                'FCC': {
                    'energy': 100e3,  # GeV
                    'luminosity': 30000.0,  # fb^-1
                    'sensitivity': 0.01,  # 1% relative uncertainty
                    'systematic': 0.008,  # 0.8% systematic uncertainty
                    'statistical': 0.006  # 0.6% statistical uncertainty
                }
            }
        else:
            self.experimental_limits = experimental_limits
        
        # Initialize with default parameters
        self.current_scale = 1.22e19  # Planck scale in GeV
        self.current_dim_uv = 2.0
        self.current_dim_ir = 4.0
        self.current_transition_scale = 1.0  # In Planck units
        
        # Store optimization history
        self.optimization_history = []
    
    def find_detectable_scale(self, max_iterations: int = 50, 
                             tolerance: float = 1e-6) -> Dict:
        """
        Find QG scale parameters that produce detectable experimental signatures.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of optimization iterations
        tolerance : float
            Convergence tolerance for scale parameter changes
            
        Returns:
        --------
        dict
            Optimized scale parameters and detection metrics
        """
        print("Starting scale parameter optimization...")
        print(f"Target significance: {self.target_significance}σ")
        
        # Initial assessment
        initial_assessment = self._assess_current_detectability()
        print(f"Initial detectability: {initial_assessment['overall_detectable']}")
        
        if initial_assessment['overall_detectable']:
            print("Current parameters already produce detectable effects!")
            return self._format_optimization_result(initial_assessment)
        
        # Gradient descent optimization
        optimized_params = self._gradient_descent_optimization(
            max_iterations=max_iterations, 
            tolerance=tolerance
        )
        
        # Final assessment
        final_assessment = self._assess_current_detectability()
        
        print(f"Optimization complete!")
        print(f"Final detectability: {final_assessment['overall_detectable']}")
        print(f"Best significance: {final_assessment['max_significance']:.2f}σ")
        
        return self._format_optimization_result(final_assessment)
    
    def _assess_current_detectability(self) -> Dict:
        """
        Assess detectability of current scale parameters across all experiments.
        
        Returns:
        --------
        dict
            Detection assessment for all experimental facilities
        """
        assessment = {
            'facilities': {},
            'overall_detectable': False,
            'max_significance': 0.0,
            'best_facility': None
        }
        
        # Test each experimental facility
        for facility_name, limits in self.experimental_limits.items():
            facility_result = self._test_facility_detectability(facility_name, limits)
            assessment['facilities'][facility_name] = facility_result
            
            # Track best result
            if facility_result['significance'] > assessment['max_significance']:
                assessment['max_significance'] = facility_result['significance']
                assessment['best_facility'] = facility_name
        
        # Overall detectability (any facility above target)
        assessment['overall_detectable'] = (
            assessment['max_significance'] >= self.target_significance
        )
        
        return assessment
    
    def _test_facility_detectability(self, facility_name: str, limits: Dict) -> Dict:
        """
        Test detectability for a specific experimental facility.
        
        Parameters:
        -----------
        facility_name : str
            Name of the experimental facility
        limits : dict
            Experimental limits and sensitivities
            
        Returns:
        --------
        dict
            Detection test results for this facility
        """
        # Create experimental predictions with current parameters
        exp_pred = ExperimentalPredictions(
            dim_uv=self.current_dim_uv,
            dim_ir=self.current_dim_ir,
            transition_scale=self.current_transition_scale
        )
        
        # Get predictions for this facility's energy
        energy_tev = limits['energy'] / 1000.0  # Convert to TeV
        
        try:
            # Use existing high-energy prediction method
            predictions = exp_pred.predict_high_energy_deviations(
                collider_energy_tev=energy_tev
            )
            
            # Extract cross-section deviation
            xsec_deviation = predictions.get('cross_section_deviation', 0.0)
            
            # Calculate significance
            total_uncertainty = np.sqrt(
                limits['statistical']**2 + limits['systematic']**2
            )
            significance = xsec_deviation / total_uncertainty
            
            # Check if detectable
            is_detectable = significance >= self.target_significance
            
            return {
                'energy_tev': energy_tev,
                'xsec_deviation': xsec_deviation,
                'significance': significance,
                'is_detectable': is_detectable,
                'uncertainty': total_uncertainty,
                'statistical_uncertainty': limits['statistical'],
                'systematic_uncertainty': limits['systematic']
            }
            
        except Exception as e:
            warnings.warn(f"Error testing {facility_name}: {str(e)}")
            return {
                'energy_tev': energy_tev,
                'xsec_deviation': 0.0,
                'significance': 0.0,
                'is_detectable': False,
                'uncertainty': limits['sensitivity'],
                'error': str(e)
            }
    
    def _gradient_descent_optimization(self, max_iterations: int, 
                                     tolerance: float) -> Dict:
        """
        Perform gradient descent optimization of scale parameters.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        dict
            Optimization results
        """
        print("Starting gradient descent optimization...")
        
        # Initial parameters
        current_params = {
            'transition_scale': self.current_transition_scale,
            'dim_uv': self.current_dim_uv,
            'dim_ir': self.current_dim_ir
        }
        
        # Learning rates for each parameter
        learning_rates = {
            'transition_scale': 0.1,
            'dim_uv': 0.01,
            'dim_ir': 0.01
        }
        
        best_significance = 0.0
        best_params = current_params.copy()
        
        for iteration in range(max_iterations):
            # Calculate current objective (maximize significance)
            current_assessment = self._assess_current_detectability()
            current_significance = current_assessment['max_significance']
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': iteration,
                'params': current_params.copy(),
                'significance': current_significance,
                'detectable': current_assessment['overall_detectable']
            })
            
            # Check if we found a better solution
            if current_significance > best_significance:
                best_significance = current_significance
                best_params = current_params.copy()
            
            # Check convergence
            if iteration > 0:
                prev_significance = self.optimization_history[-2]['significance']
                if abs(current_significance - prev_significance) < tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
            
            # Calculate gradients for each parameter
            gradients = self._calculate_gradients(current_params, learning_rates)
            
            # Update parameters
            for param_name, gradient in gradients.items():
                current_params[param_name] -= learning_rates[param_name] * gradient
                
                # Apply bounds
                if param_name == 'transition_scale':
                    current_params[param_name] = max(0.01, min(10.0, current_params[param_name]))
                elif param_name in ['dim_uv', 'dim_ir']:
                    current_params[param_name] = max(1.0, min(6.0, current_params[param_name]))
            
            # Update current parameters
            self.current_transition_scale = current_params['transition_scale']
            self.current_dim_uv = current_params['dim_uv']
            self.current_dim_ir = current_params['dim_ir']
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: significance = {current_significance:.3f}σ")
        
        # Restore best parameters
        self.current_transition_scale = best_params['transition_scale']
        self.current_dim_uv = best_params['dim_uv']
        self.current_dim_ir = best_params['dim_ir']
        
        return {
            'best_params': best_params,
            'best_significance': best_significance,
            'iterations': len(self.optimization_history)
        }
    
    def _calculate_gradients(self, params: Dict, learning_rates: Dict) -> Dict:
        """
        Calculate gradients for each parameter using finite differences.
        
        Parameters:
        -----------
        params : dict
            Current parameter values
        learning_rates : dict
            Learning rates for each parameter
            
        Returns:
        --------
        dict
            Gradients for each parameter
        """
        gradients = {}
        
        for param_name in params.keys():
            # Store original parameter
            original_value = params[param_name]
            
            # Calculate finite difference with safety bounds
            delta = max(original_value * 0.01, 1e-6)  # At least 1e-6 perturbation
            
            try:
                # Forward difference
                params[param_name] = original_value + delta
                self._update_current_params(params)
                forward_assessment = self._assess_current_detectability()
                forward_significance = forward_assessment['max_significance']
                
                # Backward difference
                params[param_name] = original_value - delta
                self._update_current_params(params)
                backward_assessment = self._assess_current_detectability()
                backward_significance = backward_assessment['max_significance']
                
                # Safety check for NaN or infinite values
                if (np.isnan(forward_significance) or np.isnan(backward_significance) or
                    np.isinf(forward_significance) or np.isinf(backward_significance)):
                    print(f"Warning: NaN/Inf detected in gradient calculation for {param_name}")
                    gradients[param_name] = 0.0
                else:
                    # Calculate gradient
                    gradient = (forward_significance - backward_significance) / (2 * delta)
                    # Clip gradient to prevent explosion
                    gradient = np.clip(gradient, -10.0, 10.0)
                    gradients[param_name] = gradient
                    
            except Exception as e:
                print(f"Warning: Error in gradient calculation for {param_name}: {e}")
                gradients[param_name] = 0.0
            
            # Restore original value
            params[param_name] = original_value
        
        return gradients
    
    def _update_current_params(self, params: Dict) -> None:
        """Update current parameters from optimization."""
        self.current_transition_scale = params['transition_scale']
        self.current_dim_uv = params['dim_uv']
        self.current_dim_ir = params['dim_ir']
    
    def _format_optimization_result(self, assessment: Dict) -> Dict:
        """Format the final optimization result."""
        return {
            'optimized_parameters': {
                'transition_scale': self.current_transition_scale,
                'dim_uv': self.current_dim_uv,
                'dim_ir': self.current_dim_ir
            },
            'detection_assessment': assessment,
            'optimization_history': self.optimization_history,
            'target_significance': self.target_significance
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.optimization_history:
            print("No optimization history available.")
            return
        
        iterations = [h['iteration'] for h in self.optimization_history]
        significances = [h['significance'] for h in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, significances, 'b-', linewidth=2)
        plt.axhline(y=self.target_significance, color='r', linestyle='--', 
                   label=f'Target ({self.target_significance}σ)')
        plt.xlabel('Optimization Iteration')
        plt.ylabel('Maximum Significance (σ)')
        plt.title('Scale Parameter Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Demonstrate the scale parameter optimizer."""
    print("Quantum Gravity Scale Parameter Optimization")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ScaleParameterOptimizer(target_significance=3.0)  # 3σ detection
    
    # Find optimal parameters
    result = optimizer.find_detectable_scale()
    
    # Display results
    print("\nOptimization Results:")
    print(f"Best significance: {result['detection_assessment']['max_significance']:.2f}σ")
    print(f"Overall detectable: {result['detection_assessment']['overall_detectable']}")
    print(f"Best facility: {result['detection_assessment']['best_facility']}")
    
    print("\nOptimized Parameters:")
    for param, value in result['optimized_parameters'].items():
        print(f"  {param}: {value:.6f}")
    
    # Plot optimization history
    optimizer.plot_optimization_history()


if __name__ == "__main__":
    main() 