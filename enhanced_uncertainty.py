#!/usr/bin/env python
"""
Enhanced Uncertainty Quantification for Quantum Gravity Framework

This module provides comprehensive uncertainty quantification for QG predictions,
building on the existing experimental predictions framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional, Tuple, Callable
import warnings

# Import existing framework components
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class EnhancedUncertaintyQuantifier:
    """
    Enhanced uncertainty quantification for QG predictions.
    
    This class builds on the existing experimental predictions framework to provide
    comprehensive statistical analysis including Monte Carlo uncertainty propagation,
    confidence intervals, and systematic error analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 n_monte_carlo_samples: int = 10000):
        """
        Initialize the enhanced uncertainty quantifier.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for uncertainty intervals (default: 0.95)
        n_monte_carlo_samples : int
            Number of Monte Carlo samples for uncertainty propagation
        """
        self.confidence_level = confidence_level
        self.n_monte_carlo_samples = n_monte_carlo_samples
        
        # Z-score for confidence level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Default parameter ranges for Monte Carlo sampling
        self.default_parameter_ranges = {
            'dim_uv': (1.5, 2.5),      # UV spectral dimension
            'dim_ir': (3.8, 4.2),      # IR spectral dimension  
            'transition_scale': (0.1, 10.0),  # Transition scale in Planck units
            'beta1': (0.05, 0.15),     # Higher-derivative coefficient
            'beta2': (0.02, 0.08),     # Mixed kinetic-mass coefficient
            'beta3': (0.005, 0.015)    # Modified mass coefficient
        }
    
    def monte_carlo_uncertainty_propagation(self, 
                                           prediction_function: Callable,
                                           parameter_ranges: Optional[Dict] = None,
                                           n_samples: Optional[int] = None) -> Dict:
        """
        Perform Monte Carlo uncertainty propagation for QG predictions.
        
        Parameters:
        -----------
        prediction_function : callable
            Function that computes predictions given parameters
        parameter_ranges : dict, optional
            Parameter ranges for sampling (uses defaults if None)
        n_samples : int, optional
            Number of Monte Carlo samples (uses default if None)
            
        Returns:
        --------
        dict
            Uncertainty analysis results with confidence intervals
        """
        if parameter_ranges is None:
            parameter_ranges = self.default_parameter_ranges
        
        if n_samples is None:
            n_samples = self.n_monte_carlo_samples
        
        print(f"Running Monte Carlo uncertainty propagation with {n_samples} samples...")
        
        # Generate parameter samples
        parameter_samples = self._generate_parameter_samples(parameter_ranges, n_samples)
        
        # Compute predictions for each sample
        predictions = []
        valid_predictions = 0
        
        for i, params in enumerate(parameter_samples):
            try:
                prediction = prediction_function(**params)
                predictions.append(prediction)
                valid_predictions += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{n_samples} samples...")
                    
            except Exception as e:
                warnings.warn(f"Failed to compute prediction for sample {i}: {str(e)}")
                continue
        
        if valid_predictions == 0:
            raise ValueError("No valid predictions computed!")
        
        print(f"Successfully computed {valid_predictions}/{n_samples} predictions")
        
        # Convert to numpy array for analysis
        predictions_array = np.array(predictions)
        
        # Compute statistics
        mean_prediction = np.mean(predictions_array)
        std_prediction = np.std(predictions_array)
        
        # Compute confidence interval
        confidence_interval = (
            mean_prediction - self.z_score * std_prediction,
            mean_prediction + self.z_score * std_prediction
        )
        
        # Compute percentiles
        percentiles = {
            '5th': np.percentile(predictions_array, 5),
            '25th': np.percentile(predictions_array, 25),
            '50th': np.percentile(predictions_array, 50),
            '75th': np.percentile(predictions_array, 75),
            '95th': np.percentile(predictions_array, 95)
        }
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'confidence_interval': confidence_interval,
            'confidence_level': self.confidence_level,
            'percentiles': percentiles,
            'samples': predictions_array,
            'valid_samples': valid_predictions,
            'total_samples': n_samples,
            'parameter_samples': parameter_samples[:valid_predictions]
        }
    
    def _generate_parameter_samples(self, parameter_ranges: Dict, n_samples: int) -> List[Dict]:
        """
        Generate parameter samples for Monte Carlo analysis.
        
        Parameters:
        -----------
        parameter_ranges : dict
            Parameter ranges for sampling
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                # Use uniform sampling within ranges
                sample[param_name] = np.random.uniform(min_val, max_val)
            samples.append(sample)
        
        return samples
    
    def lhc_prediction_uncertainty(self, 
                                  energy_tev: float = 13.6,
                                  include_systematics: bool = True) -> Dict:
        """
        Compute uncertainty for LHC predictions using Monte Carlo analysis.
        
        Parameters:
        -----------
        energy_tev : float
            Collider energy in TeV
        include_systematics : bool
            Whether to include systematic uncertainties
            
        Returns:
        --------
        dict
            LHC prediction uncertainty analysis
        """
        print(f"Computing LHC prediction uncertainty for {energy_tev} TeV...")
        
        def lhc_prediction_function(**params):
            """LHC prediction function for Monte Carlo analysis."""
            # Create experimental predictions with sampled parameters
            exp_pred = ExperimentalPredictions(
                dim_uv=params.get('dim_uv', 2.0),
                dim_ir=params.get('dim_ir', 4.0),
                transition_scale=params.get('transition_scale', 1.0)
            )
            
            # Get LHC predictions
            lhc_results = exp_pred.predict_lhc_deviations()
            
            # Extract cross-section deviation
            xsec_deviation = lhc_results.get('cross_section_deviation', 0.0)
            
            # Add systematic uncertainty if requested
            if include_systematics:
                systematic_uncertainty = 0.03  # 3% systematic
                xsec_deviation += np.random.normal(0, systematic_uncertainty)
            
            return xsec_deviation
        
        # Run Monte Carlo analysis
        mc_results = self.monte_carlo_uncertainty_propagation(lhc_prediction_function)
        
        # Add LHC-specific analysis
        mc_results['facility'] = 'LHC'
        mc_results['energy_tev'] = energy_tev
        mc_results['include_systematics'] = include_systematics
        
        return mc_results
    
    def higgs_pt_uncertainty(self, 
                            pt_range: Tuple[float, float] = (0, 1500),
                            collision_energy: float = 13.6e3) -> Dict:
        """
        Compute uncertainty for Higgs pT spectrum predictions.
        
        Parameters:
        -----------
        pt_range : tuple
            Range of pT values in GeV
        collision_energy : float
            Collision energy in GeV
            
        Returns:
        --------
        dict
            Higgs pT uncertainty analysis
        """
        print(f"Computing Higgs pT uncertainty for {collision_energy/1000:.1f} TeV...")
        
        def higgs_pt_prediction_function(**params):
            """Higgs pT prediction function for Monte Carlo analysis."""
            # Create experimental predictions with sampled parameters
            exp_pred = ExperimentalPredictions(
                dim_uv=params.get('dim_uv', 2.0),
                dim_ir=params.get('dim_ir', 4.0),
                transition_scale=params.get('transition_scale', 1.0)
            )
            
            # Get high-energy predictions
            predictions = exp_pred.predict_high_energy_deviations(
                collider_energy_tev=collision_energy/1000.0
            )
            
            # Extract Higgs production modification
            higgs_modification = predictions.get('cross_sections', {}).get('2to2_scalar', {})
            if isinstance(higgs_modification, dict):
                return higgs_modification.get('mean', 1.0)
            else:
                return higgs_modification
        
        # Run Monte Carlo analysis
        mc_results = self.monte_carlo_uncertainty_propagation(higgs_pt_prediction_function)
        
        # Add Higgs-specific analysis
        mc_results['process'] = 'Higgs_pT_spectrum'
        mc_results['pt_range'] = pt_range
        mc_results['collision_energy'] = collision_energy
        
        return mc_results
    
    def gravitational_wave_uncertainty(self, 
                                     frequency_range: Tuple[float, float] = (10, 2000)) -> Dict:
        """
        Compute uncertainty for gravitational wave predictions.
        
        Parameters:
        -----------
        frequency_range : tuple
            Range of frequencies in Hz
            
        Returns:
        --------
        dict
            Gravitational wave uncertainty analysis
        """
        print("Computing gravitational wave prediction uncertainty...")
        
        def gw_prediction_function(**params):
            """Gravitational wave prediction function for Monte Carlo analysis."""
            # Create experimental predictions with sampled parameters
            exp_pred = ExperimentalPredictions(
                dim_uv=params.get('dim_uv', 2.0),
                dim_ir=params.get('dim_ir', 4.0),
                transition_scale=params.get('transition_scale', 1.0)
            )
            
            # Get GW predictions
            gw_results = exp_pred.predict_gravitational_wave_modifications()
            
            # Extract phase speed modification
            v_modifications = gw_results.get('v_modifications', [1.0])
            return np.mean(v_modifications)  # Average over frequency range
        
        # Run Monte Carlo analysis
        mc_results = self.monte_carlo_uncertainty_propagation(gw_prediction_function)
        
        # Add GW-specific analysis
        mc_results['process'] = 'gravitational_wave_propagation'
        mc_results['frequency_range'] = frequency_range
        
        return mc_results
    
    def parameter_sensitivity_analysis(self, 
                                    base_parameters: Optional[Dict] = None) -> Dict:
        """
        Perform parameter sensitivity analysis.
        
        Parameters:
        -----------
        base_parameters : dict, optional
            Base parameter values for sensitivity analysis
            
        Returns:
        --------
        dict
            Parameter sensitivity analysis results
        """
        if base_parameters is None:
            base_parameters = {
                'dim_uv': 2.0,
                'dim_ir': 4.0,
                'transition_scale': 1.0,
                'beta1': 0.1,
                'beta2': 0.05,
                'beta3': 0.01
            }
        
        print("Performing parameter sensitivity analysis...")
        
        sensitivity_results = {}
        
        for param_name, base_value in base_parameters.items():
            # Test parameter variations
            variations = np.linspace(0.5 * base_value, 1.5 * base_value, 11)
            predictions = []
            
            for variation in variations:
                # Create test parameters
                test_params = base_parameters.copy()
                test_params[param_name] = variation
                
                # Compute prediction
                try:
                    exp_pred = ExperimentalPredictions(
                        dim_uv=test_params['dim_uv'],
                        dim_ir=test_params['dim_ir'],
                        transition_scale=test_params['transition_scale']
                    )
                    
                    lhc_results = exp_pred.predict_lhc_deviations()
                    prediction = lhc_results.get('cross_section_deviation', 0.0)
                    predictions.append(prediction)
                    
                except Exception as e:
                    warnings.warn(f"Failed to compute prediction for {param_name}={variation}: {str(e)}")
                    predictions.append(0.0)
            
            # Compute sensitivity metrics
            predictions = np.array(predictions)
            sensitivity = np.std(predictions) / np.mean(predictions) if np.mean(predictions) != 0 else 0
            
            sensitivity_results[param_name] = {
                'variations': variations,
                'predictions': predictions,
                'sensitivity': sensitivity,
                'base_value': base_value
            }
        
        return sensitivity_results
    
    def plot_uncertainty_analysis(self, 
                                mc_results: Dict, 
                                save_path: Optional[str] = None) -> None:
        """
        Plot uncertainty analysis results.
        
        Parameters:
        -----------
        mc_results : dict
            Monte Carlo uncertainty analysis results
        save_path : str, optional
            Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram of predictions
        ax1.hist(mc_results['samples'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(mc_results['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(mc_results['confidence_interval'][0], color='orange', linestyle=':', linewidth=2, label=f'{self.confidence_level*100:.0f}% CI')
        ax1.axvline(mc_results['confidence_interval'][1], color='orange', linestyle=':', linewidth=2)
        ax1.set_xlabel('Prediction Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Monte Carlo Prediction Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        sorted_samples = np.sort(mc_results['samples'])
        cumulative = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax2.plot(sorted_samples, cumulative, 'b-', linewidth=2)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Median')
        ax2.axhline(0.025, color='orange', linestyle=':', alpha=0.7, label=f'{self.confidence_level*100:.0f}% CI')
        ax2.axhline(0.975, color='orange', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Prediction Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter correlation (if available)
        if 'parameter_samples' in mc_results and len(mc_results['parameter_samples']) > 0:
            param_names = list(mc_results['parameter_samples'][0].keys())
            if len(param_names) >= 2:
                # Plot correlation between first two parameters
                param1_values = [s[param_names[0]] for s in mc_results['parameter_samples']]
                param2_values = [s[param_names[1]] for s in mc_results['parameter_samples']]
                
                ax3.scatter(param1_values, param2_values, alpha=0.6, s=20)
                ax3.set_xlabel(param_names[0])
                ax3.set_ylabel(param_names[1])
                ax3.set_title('Parameter Correlation')
                ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        Monte Carlo Uncertainty Analysis
        
        Sample Statistics:
        - Mean: {mc_results['mean']:.6f}
        - Std Dev: {mc_results['std']:.6f}
        - Median: {mc_results['percentiles']['50th']:.6f}
        
        Confidence Interval ({self.confidence_level*100:.0f}%):
        - Lower: {mc_results['confidence_interval'][0]:.6f}
        - Upper: {mc_results['confidence_interval'][1]:.6f}
        
        Sample Information:
        - Valid samples: {mc_results['valid_samples']}
        - Total samples: {mc_results['total_samples']}
        - Success rate: {mc_results['valid_samples']/mc_results['total_samples']*100:.1f}%
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Demonstrate the enhanced uncertainty quantifier."""
    print("Enhanced Uncertainty Quantification for Quantum Gravity")
    print("=" * 60)
    
    # Initialize uncertainty quantifier
    quantifier = EnhancedUncertaintyQuantifier(confidence_level=0.95)
    
    # Test LHC prediction uncertainty
    print("\n1. LHC Prediction Uncertainty Analysis")
    lhc_results = quantifier.lhc_prediction_uncertainty(energy_tev=13.6)
    
    print(f"   Mean prediction: {lhc_results['mean']:.6f}")
    print(f"   Standard deviation: {lhc_results['std']:.6f}")
    print(f"   {quantifier.confidence_level*100:.0f}% CI: [{lhc_results['confidence_interval'][0]:.6f}, {lhc_results['confidence_interval'][1]:.6f}]")
    
    # Test Higgs pT uncertainty
    print("\n2. Higgs pT Uncertainty Analysis")
    higgs_results = quantifier.higgs_pt_uncertainty()
    
    print(f"   Mean modification: {higgs_results['mean']:.6f}")
    print(f"   Standard deviation: {higgs_results['std']:.6f}")
    
    # Test parameter sensitivity
    print("\n3. Parameter Sensitivity Analysis")
    sensitivity_results = quantifier.parameter_sensitivity_analysis()
    
    print("   Parameter sensitivities:")
    for param, result in sensitivity_results.items():
        print(f"     {param}: {result['sensitivity']:.4f}")
    
    # Plot results
    print("\n4. Generating uncertainty analysis plots...")
    quantifier.plot_uncertainty_analysis(lhc_results)


if __name__ == "__main__":
    main() 