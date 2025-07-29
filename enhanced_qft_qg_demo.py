#!/usr/bin/env python
"""
Enhanced QFT-QG Integration Demo

This script demonstrates the three key enhancements to the quantum gravity framework:
1. Scale Parameter Optimization - Find detectable QG signatures
2. Enhanced Uncertainty Quantification - Proper statistical analysis
3. GPU Acceleration - Faster parameter exploration

The script shows how these enhancements work together to provide
experimentally testable predictions with proper error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Optional
import warnings

# Import the enhanced modules
from scale_optimization import ScaleParameterOptimizer
from enhanced_uncertainty import EnhancedUncertaintyQuantifier
from gpu_acceleration import GPUAcceleratedLattice, GPUAcceleratedQGRG

# Import existing framework components
from quantum_gravity_framework.experimental_predictions import ExperimentalPredictions
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class EnhancedQFTQGDemo:
    """
    Comprehensive demonstration of enhanced QFT-QG integration.
    
    This class showcases how the three enhancements work together
    to provide experimentally testable quantum gravity predictions.
    """
    
    def __init__(self, target_significance: float = 3.0):
        """
        Initialize the enhanced QFT-QG demo.
        
        Parameters:
        -----------
        target_significance : float
            Target significance level for detection
        """
        self.target_significance = target_significance
        
        # Initialize enhancement modules
        self.scale_optimizer = ScaleParameterOptimizer(target_significance=target_significance)
        self.uncertainty_quantifier = EnhancedUncertaintyQuantifier()
        
        # Store results
        self.results = {}
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis using all three enhancements.
        
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        print("Enhanced QFT-QG Integration Analysis")
        print("=" * 50)
        
        # Step 1: Scale Parameter Optimization
        print("\n1. Scale Parameter Optimization")
        print("-" * 30)
        optimization_results = self._optimize_scale_parameters()
        
        # Step 2: Uncertainty Quantification
        print("\n2. Uncertainty Quantification")
        print("-" * 30)
        uncertainty_results = self._quantify_uncertainties()
        
        # Step 3: GPU-Accelerated Parameter Exploration
        print("\n3. GPU-Accelerated Parameter Exploration")
        print("-" * 30)
        gpu_results = self._gpu_parameter_exploration()
        
        # Step 4: Combined Analysis
        print("\n4. Combined Analysis")
        print("-" * 30)
        combined_results = self._combine_analysis_results(
            optimization_results, uncertainty_results, gpu_results
        )
        
        # Store all results
        self.results = {
            'optimization': optimization_results,
            'uncertainty': uncertainty_results,
            'gpu_acceleration': gpu_results,
            'combined': combined_results
        }
        
        return self.results
    
    def _optimize_scale_parameters(self) -> Dict:
        """Optimize scale parameters for detectable signatures."""
        print("Finding optimal QG scale parameters...")
        
        # Run scale optimization
        optimization_result = self.scale_optimizer.find_detectable_scale(
            max_iterations=30,  # Reduced for demo
            tolerance=1e-4
        )
        
        # Extract optimized parameters
        optimized_params = optimization_result['optimized_parameters']
        detection_assessment = optimization_result['detection_assessment']
        
        print(f"Optimization complete!")
        print(f"Best significance: {detection_assessment['max_significance']:.2f}σ")
        print(f"Overall detectable: {detection_assessment['overall_detectable']}")
        print(f"Best facility: {detection_assessment['best_facility']}")
        
        print("\nOptimized Parameters:")
        for param, value in optimized_params.items():
            print(f"  {param}: {value:.6f}")
        
        return optimization_result
    
    def _quantify_uncertainties(self) -> Dict:
        """Quantify uncertainties for QG predictions."""
        print("Performing comprehensive uncertainty analysis...")
        
        # Test different prediction types
        uncertainty_results = {}
        
        # 1. LHC prediction uncertainty
        print("  Analyzing LHC prediction uncertainties...")
        lhc_uncertainty = self.uncertainty_quantifier.lhc_prediction_uncertainty(
            energy_tev=13.6,
            include_systematics=True
        )
        uncertainty_results['lhc'] = lhc_uncertainty
        
        # 2. Higgs pT uncertainty
        print("  Analyzing Higgs pT uncertainties...")
        higgs_uncertainty = self.uncertainty_quantifier.higgs_pt_uncertainty()
        uncertainty_results['higgs_pt'] = higgs_uncertainty
        
        # 3. Gravitational wave uncertainty
        print("  Analyzing gravitational wave uncertainties...")
        gw_uncertainty = self.uncertainty_quantifier.gravitational_wave_uncertainty()
        uncertainty_results['gravitational_wave'] = gw_uncertainty
        
        # 4. Parameter sensitivity analysis
        print("  Performing parameter sensitivity analysis...")
        sensitivity = self.uncertainty_quantifier.parameter_sensitivity_analysis()
        uncertainty_results['sensitivity'] = sensitivity
        
        print("Uncertainty analysis complete!")
        
        return uncertainty_results
    
    def _gpu_parameter_exploration(self) -> Dict:
        """Perform GPU-accelerated parameter exploration."""
        print("Running GPU-accelerated parameter exploration...")
        
        gpu_results = {}
        
        # 1. GPU-accelerated lattice simulation
        print("  Running GPU lattice simulation...")
        gpu_lattice = GPUAcceleratedLattice(
            lattice_shape=(32, 32),
            mass_squared=0.1,
            coupling=0.1,
            dimension=2,
            use_gpu=True
        )
        
        lattice_results = gpu_lattice.run_gpu_simulation(
            num_thermalization=500,
            num_configurations=500
        )
        gpu_results['lattice_simulation'] = lattice_results
        
        # 2. Performance comparison
        print("  Comparing GPU vs CPU performance...")
        performance_comparison = gpu_lattice.compare_with_cpu(num_configurations=200)
        gpu_results['performance_comparison'] = performance_comparison
        
        # 3. GPU-accelerated RG flow
        print("  Computing GPU-accelerated RG flow...")
        gpu_rg = GPUAcceleratedQGRG(
            dim_uv=2.0,
            dim_ir=4.0,
            transition_scale=1.0
        )
        
        rg_results = gpu_rg.accelerated_rg_flow()
        gpu_results['rg_flow'] = rg_results
        
        print("GPU acceleration analysis complete!")
        
        return gpu_results
    
    def _combine_analysis_results(self, optimization_results: Dict, 
                                uncertainty_results: Dict, 
                                gpu_results: Dict) -> Dict:
        """Combine all analysis results into comprehensive summary."""
        print("Combining analysis results...")
        
        # Extract key metrics
        best_significance = optimization_results['detection_assessment']['max_significance']
        best_facility = optimization_results['detection_assessment']['best_facility']
        
        # Average uncertainty across predictions
        lhc_uncertainty = uncertainty_results['lhc']['std']
        higgs_uncertainty = uncertainty_results['higgs_pt']['std']
        gw_uncertainty = uncertainty_results['gravitational_wave']['std']
        
        avg_uncertainty = np.mean([lhc_uncertainty, higgs_uncertainty, gw_uncertainty])
        
        # GPU performance metrics
        gpu_speedup = gpu_results['performance_comparison']['speedup']
        
        # Overall assessment
        is_detectable = best_significance >= self.target_significance
        confidence_level = min(0.99, best_significance / 5.0)  # Scale to confidence
        
        combined_results = {
            'detectability': {
                'is_detectable': is_detectable,
                'best_significance': best_significance,
                'best_facility': best_facility,
                'target_significance': self.target_significance
            },
            'uncertainty': {
                'average_uncertainty': avg_uncertainty,
                'lhc_uncertainty': lhc_uncertainty,
                'higgs_uncertainty': higgs_uncertainty,
                'gw_uncertainty': gw_uncertainty
            },
            'performance': {
                'gpu_speedup': gpu_speedup,
                'computation_efficient': gpu_speedup > 2.0
            },
            'confidence': {
                'overall_confidence': confidence_level,
                'high_confidence': confidence_level > 0.8
            }
        }
        
        print("Combined analysis complete!")
        print(f"Detection capability: {'✅' if is_detectable else '❌'}")
        print(f"Best significance: {best_significance:.2f}σ")
        print(f"GPU speedup: {gpu_speedup:.1f}x")
        print(f"Average uncertainty: {avg_uncertainty:.6f}")
        
        return combined_results
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> None:
        """
        Generate comprehensive analysis report.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report
        """
        if not self.results:
            print("No results available. Run comprehensive_analysis first.")
            return
        
        print("\n" + "="*60)
        print("ENHANCED QFT-QG INTEGRATION REPORT")
        print("="*60)
        
        # Executive Summary
        combined = self.results['combined']
        print(f"\nEXECUTIVE SUMMARY")
        print(f"Detection Capability: {'✅ DETECTABLE' if combined['detectability']['is_detectable'] else '❌ NOT DETECTABLE'}")
        print(f"Best Significance: {combined['detectability']['best_significance']:.2f}σ")
        print(f"Best Facility: {combined['detectability']['best_facility']}")
        print(f"GPU Speedup: {combined['performance']['gpu_speedup']:.1f}x")
        print(f"Overall Confidence: {combined['confidence']['overall_confidence']:.1%}")
        
        # Detailed Results
        print(f"\nDETAILED RESULTS")
        
        # Scale Optimization
        opt = self.results['optimization']
        print(f"\n1. Scale Parameter Optimization:")
        print(f"   Iterations: {len(opt['optimization_history'])}")
        print(f"   Final Parameters:")
        for param, value in opt['optimized_parameters'].items():
            print(f"     {param}: {value:.6f}")
        
        # Uncertainty Analysis
        unc = self.results['uncertainty']
        print(f"\n2. Uncertainty Quantification:")
        print(f"   LHC Uncertainty: {unc['lhc']['std']:.6f}")
        print(f"   Higgs pT Uncertainty: {unc['higgs_pt']['std']:.6f}")
        print(f"   GW Uncertainty: {unc['gravitational_wave']['std']:.6f}")
        
        # GPU Performance
        gpu = self.results['gpu_acceleration']
        perf = gpu['performance_comparison']
        print(f"\n3. GPU Acceleration:")
        print(f"   GPU Time: {perf['gpu_time']:.2f}s")
        print(f"   CPU Time: {perf['cpu_time']:.2f}s")
        print(f"   Speedup: {perf['speedup']:.1f}x")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS")
        if combined['detectability']['is_detectable']:
            print(f"✅ Proceed with experimental validation at {combined['detectability']['best_facility']}")
        else:
            print(f"⚠️  Further optimization needed for detection")
        
        if combined['performance']['computation_efficient']:
            print(f"✅ GPU acceleration provides significant performance improvement")
        else:
            print(f"⚠️  Consider additional GPU optimization")
        
        if combined['confidence']['high_confidence']:
            print(f"✅ High confidence in predictions")
        else:
            print(f"⚠️  Additional uncertainty analysis recommended")
        
        print(f"\n" + "="*60)
    
    def plot_comprehensive_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive analysis results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plots
        """
        if not self.results:
            print("No results available. Run comprehensive_analysis first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Optimization history
        opt_history = self.results['optimization']['optimization_history']
        iterations = [h['iteration'] for h in opt_history]
        significances = [h['significance'] for h in opt_history]
        
        ax1.plot(iterations, significances, 'b-', linewidth=2)
        ax1.axhline(y=self.target_significance, color='r', linestyle='--', 
                   label=f'Target ({self.target_significance}σ)')
        ax1.set_xlabel('Optimization Iteration')
        ax1.set_ylabel('Maximum Significance (σ)')
        ax1.set_title('Scale Parameter Optimization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Uncertainty distribution
        lhc_samples = self.results['uncertainty']['lhc']['samples']
        ax2.hist(lhc_samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.results['uncertainty']['lhc']['mean'], color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('LHC Prediction Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('LHC Prediction Uncertainty')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. GPU performance comparison
        perf = self.results['gpu_acceleration']['performance_comparison']
        facilities = ['GPU', 'CPU']
        times = [perf['gpu_time'], perf['cpu_time']]
        
        bars = ax3.bar(facilities, times, color=['green', 'blue'], alpha=0.7)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('GPU vs CPU Performance')
        ax3.grid(True, alpha=0.3)
        
        # Add speedup annotation
        speedup = perf['speedup']
        ax3.text(0.5, max(times) * 0.8, f'Speedup: {speedup:.1f}x', 
                ha='center', fontsize=12, fontweight='bold')
        
        # 4. Combined metrics
        combined = self.results['combined']
        metrics = ['Detection\nCapability', 'GPU\nSpeedup', 'Uncertainty\nControl', 'Overall\nConfidence']
        values = [
            1.0 if combined['detectability']['is_detectable'] else 0.0,
            min(combined['performance']['gpu_speedup'] / 10.0, 1.0),  # Normalize to 0-1
            1.0 - min(combined['uncertainty']['average_uncertainty'] * 100, 1.0),  # Invert uncertainty
            combined['confidence']['overall_confidence']
        ]
        colors = ['green' if v > 0.5 else 'red' for v in values]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Score (0-1)')
        ax4.set_title('Overall Framework Assessment')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run comprehensive enhanced QFT-QG integration demo."""
    print("Enhanced QFT-QG Integration Framework Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = EnhancedQFTQGDemo(target_significance=3.0)
    
    # Run comprehensive analysis
    results = demo.run_comprehensive_analysis()
    
    # Generate report
    demo.generate_comprehensive_report()
    
    # Plot results
    demo.plot_comprehensive_results()
    
    print("\nDemo complete! The enhanced framework provides:")
    print("✅ Optimized scale parameters for detection")
    print("✅ Comprehensive uncertainty quantification")
    print("✅ GPU acceleration for faster computation")
    print("✅ Experimentally testable predictions")


if __name__ == "__main__":
    main() 