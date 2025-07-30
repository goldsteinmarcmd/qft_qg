#!/usr/bin/env python3
"""
Cosmic Ray Detection Methods for Quantum Gravity

This module implements theoretical frameworks for detecting quantum gravity effects
in ultra-high energy cosmic rays, providing the highest energy access to QG physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

# Import core components
from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG


class CosmicRayDetection:
    """
    Cosmic ray detection methods for quantum gravity effects.
    
    This class implements theoretical frameworks for detecting QG effects in
    ultra-high energy cosmic rays, which provide access to energies far beyond
    what's achievable in terrestrial accelerators.
    """
    
    def __init__(self, planck_energy: float = 1.22e19):
        """
        Initialize cosmic ray detection framework.
        
        Parameters:
        -----------
        planck_energy : float
            Planck energy in eV (default: 1.22e19)
        """
        self.planck_energy = planck_energy
        self.qst = QuantumSpacetimeAxioms(dim=4, planck_length=1.0, spectral_cutoff=10)
        self.rg = DimensionalFlowRG(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
        
        # Cosmic ray observatories and their characteristics
        self.observatories = {
            'Pierre Auger': {
                'energy_range': (1e17, 1e21),  # eV
                'exposure': 1e4,  # km²·sr·yr
                'resolution': 0.1,  # energy resolution
                'location': 'Argentina'
            },
            'Telescope Array': {
                'energy_range': (1e17, 1e21),  # eV
                'exposure': 3e3,  # km²·sr·yr
                'resolution': 0.15,  # energy resolution
                'location': 'Utah, USA'
            },
            'IceCube': {
                'energy_range': (1e14, 1e18),  # eV
                'exposure': 1e6,  # km³·sr·yr
                'resolution': 0.2,  # energy resolution
                'location': 'Antarctica'
            },
            'Future CTA': {
                'energy_range': (1e11, 1e15),  # eV
                'exposure': 1e2,  # km²·sr·yr
                'resolution': 0.05,  # energy resolution
                'location': 'Multiple sites'
            }
        }
        
        # QG effect scaling parameters
        self.qg_scaling = {
            'lorentz_violation': 1e-20,  # Dimensionless parameter
            'dispersion_relation': 1e-18,  # Energy scaling
            'threshold_effects': 1e-16,  # Threshold energy effects
            'composition_effects': 1e-19  # Particle composition effects
        }
    
    def analyze_lorentz_violation(self, energy: float, particle_type: str = 'proton') -> Dict:
        """
        Analyze Lorentz violation effects in cosmic rays.
        
        Parameters:
        -----------
        energy : float
            Cosmic ray energy in eV
        particle_type : str
            Type of cosmic ray particle ('proton', 'nucleus', 'photon')
            
        Returns:
        --------
        Dict
            Lorentz violation analysis results
        """
        print(f"Analyzing Lorentz violation at {energy:.2e} eV...")
        
        # Energy ratio to Planck scale
        energy_ratio = energy / self.planck_energy
        
        # Lorentz violation parameter (simplified model)
        # In many QG models, Lorentz violation scales as (E/M_Planck)^n
        n_lorentz = 1.0  # Power law index
        lorentz_violation = self.qg_scaling['lorentz_violation'] * (energy_ratio ** n_lorentz)
        
        # Time delay effect (simplified)
        # Higher energy particles arrive later due to Lorentz violation
        distance = 1e26  # cm (typical cosmic ray source distance)
        c = 3e10  # cm/s
        time_delay = lorentz_violation * distance / c
        
        # Energy-dependent effects
        if particle_type == 'proton':
            mass_factor = 1.0
        elif particle_type == 'nucleus':
            mass_factor = 2.0  # Heavier nuclei show stronger effects
        else:  # photon
            mass_factor = 0.5  # Photons show different effects
        
        # Detectability assessment
        current_precision = 1e-12  # Current experimental precision
        detectable = lorentz_violation > current_precision
        improvement_needed = current_precision / lorentz_violation if lorentz_violation > 0 else float('inf')
        
        print(f"  Lorentz violation parameter: {lorentz_violation:.2e}")
        print(f"  Time delay: {time_delay:.2e} seconds")
        print(f"  Detectable: {'✅ YES' if detectable else '❌ NO'}")
        print(f"  Improvement needed: {improvement_needed:.2e}x")
        
        return {
            'energy': energy,
            'particle_type': particle_type,
            'lorentz_violation': lorentz_violation,
            'time_delay': time_delay,
            'energy_ratio': energy_ratio,
            'detectable': detectable,
            'improvement_needed': improvement_needed,
            'current_precision': current_precision
        }
    
    def analyze_dispersion_relation(self, energy: float, redshift: float = 0.0) -> Dict:
        """
        Analyze modified dispersion relations in cosmic rays.
        
        Parameters:
        -----------
        energy : float
            Cosmic ray energy in eV
        redshift : float
            Redshift of the source (for cosmological effects)
            
        Returns:
        --------
        Dict
            Dispersion relation analysis results
        """
        print(f"Analyzing dispersion relation at {energy:.2e} eV...")
        
        # Energy ratio to Planck scale
        energy_ratio = energy / self.planck_energy
        
        # Modified dispersion relation (simplified)
        # E² = p²c² + m²c⁴ + QG correction
        qg_correction = self.qg_scaling['dispersion_relation'] * (energy_ratio ** 2)
        
        # Cosmological effects (redshift-dependent)
        cosmological_factor = (1 + redshift) ** 3  # Simplified redshift scaling
        qg_correction *= cosmological_factor
        
        # Phase velocity modification
        c = 3e10  # cm/s
        phase_velocity_modification = qg_correction / (energy * 1.6e-12)  # Convert eV to erg
        
        # Group velocity modification
        group_velocity_modification = phase_velocity_modification * 2  # Simplified relation
        
        # Detectability assessment
        current_precision = 1e-15  # Current experimental precision
        detectable = abs(phase_velocity_modification) > current_precision
        improvement_needed = current_precision / abs(phase_velocity_modification) if abs(phase_velocity_modification) > 0 else float('inf')
        
        print(f"  QG correction: {qg_correction:.2e}")
        print(f"  Phase velocity modification: {phase_velocity_modification:.2e}")
        print(f"  Group velocity modification: {group_velocity_modification:.2e}")
        print(f"  Detectable: {'✅ YES' if detectable else '❌ NO'}")
        print(f"  Improvement needed: {improvement_needed:.2e}x")
        
        return {
            'energy': energy,
            'redshift': redshift,
            'qg_correction': qg_correction,
            'phase_velocity_modification': phase_velocity_modification,
            'group_velocity_modification': group_velocity_modification,
            'cosmological_factor': cosmological_factor,
            'detectable': detectable,
            'improvement_needed': improvement_needed,
            'current_precision': current_precision
        }
    
    def analyze_threshold_effects(self, energy: float, particle_type: str = 'proton') -> Dict:
        """
        Analyze threshold effects in cosmic ray interactions.
        
        Parameters:
        -----------
        energy : float
            Cosmic ray energy in eV
        particle_type : str
            Type of cosmic ray particle
            
        Returns:
        --------
        Dict
            Threshold effects analysis results
        """
        print(f"Analyzing threshold effects at {energy:.2e} eV...")
        
        # Energy ratio to Planck scale
        energy_ratio = energy / self.planck_energy
        
        # Threshold energy modification
        # QG effects can modify interaction thresholds
        threshold_modification = self.qg_scaling['threshold_effects'] * (energy_ratio ** 1.5)
        
        # Cross-section modification
        # QG effects can modify interaction cross-sections
        cross_section_modification = threshold_modification * 0.1  # Simplified relation
        
        # GZK cutoff modification
        # GZK cutoff occurs when cosmic ray protons interact with CMB photons
        gzk_energy = 5e19  # eV (standard GZK cutoff)
        gzk_modification = threshold_modification * (energy / gzk_energy) ** 2
        
        # Detectability assessment
        current_precision = 1e-10  # Current experimental precision
        detectable = abs(threshold_modification) > current_precision
        improvement_needed = current_precision / abs(threshold_modification) if abs(threshold_modification) > 0 else float('inf')
        
        print(f"  Threshold modification: {threshold_modification:.2e}")
        print(f"  Cross-section modification: {cross_section_modification:.2e}")
        print(f"  GZK modification: {gzk_modification:.2e}")
        print(f"  Detectable: {'✅ YES' if detectable else '❌ NO'}")
        print(f"  Improvement needed: {improvement_needed:.2e}x")
        
        return {
            'energy': energy,
            'particle_type': particle_type,
            'threshold_modification': threshold_modification,
            'cross_section_modification': cross_section_modification,
            'gzk_modification': gzk_modification,
            'detectable': detectable,
            'improvement_needed': improvement_needed,
            'current_precision': current_precision
        }
    
    def analyze_composition_effects(self, energy: float, composition: str = 'proton') -> Dict:
        """
        Analyze particle composition effects in cosmic rays.
        
        Parameters:
        -----------
        energy : float
            Cosmic ray energy in eV
        composition : str
            Particle composition ('proton', 'iron', 'mixed')
            
        Returns:
        --------
        Dict
            Composition effects analysis results
        """
        print(f"Analyzing composition effects at {energy:.2e} eV...")
        
        # Energy ratio to Planck scale
        energy_ratio = energy / self.planck_energy
        
        # Composition-dependent QG effects
        if composition == 'proton':
            mass_factor = 1.0
            charge_factor = 1.0
        elif composition == 'iron':
            mass_factor = 56.0  # Iron nucleus
            charge_factor = 26.0  # Iron charge
        else:  # mixed
            mass_factor = 10.0  # Average composition
            charge_factor = 5.0
        
        # QG effects scale with mass and charge
        composition_effect = self.qg_scaling['composition_effects'] * (energy_ratio ** 2) * mass_factor * charge_factor
        
        # Fragmentation modification
        # QG effects can modify nuclear fragmentation
        fragmentation_modification = composition_effect * 0.5
        
        # Energy loss modification
        # QG effects can modify energy loss processes
        energy_loss_modification = composition_effect * 0.3
        
        # Detectability assessment
        current_precision = 1e-8  # Current experimental precision
        detectable = abs(composition_effect) > current_precision
        improvement_needed = current_precision / abs(composition_effect) if abs(composition_effect) > 0 else float('inf')
        
        print(f"  Composition effect: {composition_effect:.2e}")
        print(f"  Fragmentation modification: {fragmentation_modification:.2e}")
        print(f"  Energy loss modification: {energy_loss_modification:.2e}")
        print(f"  Detectable: {'✅ YES' if detectable else '❌ NO'}")
        print(f"  Improvement needed: {improvement_needed:.2e}x")
        
        return {
            'energy': energy,
            'composition': composition,
            'composition_effect': composition_effect,
            'fragmentation_modification': fragmentation_modification,
            'energy_loss_modification': energy_loss_modification,
            'mass_factor': mass_factor,
            'charge_factor': charge_factor,
            'detectable': detectable,
            'improvement_needed': improvement_needed,
            'current_precision': current_precision
        }
    
    def evaluate_observatory_sensitivity(self, observatory_name: str, energy_range: Tuple[float, float]) -> Dict:
        """
        Evaluate the sensitivity of a cosmic ray observatory to QG effects.
        
        Parameters:
        -----------
        observatory_name : str
            Name of the observatory
        energy_range : Tuple[float, float]
            Energy range to evaluate (min, max) in eV
            
        Returns:
        --------
        Dict
            Observatory sensitivity analysis
        """
        print(f"Evaluating {observatory_name} sensitivity...")
        
        if observatory_name not in self.observatories:
            raise ValueError(f"Unknown observatory: {observatory_name}")
        
        observatory = self.observatories[observatory_name]
        
        # Generate energy points
        energies = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), 50)
        
        # Analyze all effects
        lorentz_results = []
        dispersion_results = []
        threshold_results = []
        composition_results = []
        
        for energy in energies:
            # Lorentz violation
            lorentz_result = self.analyze_lorentz_violation(energy)
            lorentz_results.append(lorentz_result)
            
            # Dispersion relation
            dispersion_result = self.analyze_dispersion_relation(energy)
            dispersion_results.append(dispersion_result)
            
            # Threshold effects
            threshold_result = self.analyze_threshold_effects(energy)
            threshold_results.append(threshold_result)
            
            # Composition effects
            composition_result = self.analyze_composition_effects(energy)
            composition_results.append(composition_result)
        
        # Calculate sensitivity metrics
        detectable_lorentz = sum(1 for r in lorentz_results if r['detectable'])
        detectable_dispersion = sum(1 for r in dispersion_results if r['detectable'])
        detectable_threshold = sum(1 for r in threshold_results if r['detectable'])
        detectable_composition = sum(1 for r in composition_results if r['detectable'])
        
        total_measurements = len(energies)
        
        # Overall sensitivity
        overall_sensitivity = (detectable_lorentz + detectable_dispersion + 
                             detectable_threshold + detectable_composition) / (4 * total_measurements)
        
        print(f"  Observatory: {observatory_name}")
        print(f"  Energy range: {energy_range[0]:.2e} - {energy_range[1]:.2e} eV")
        print(f"  Lorentz violation: {detectable_lorentz}/{total_measurements} detectable")
        print(f"  Dispersion relation: {detectable_dispersion}/{total_measurements} detectable")
        print(f"  Threshold effects: {detectable_threshold}/{total_measurements} detectable")
        print(f"  Composition effects: {detectable_composition}/{total_measurements} detectable")
        print(f"  Overall sensitivity: {overall_sensitivity:.3f}")
        
        return {
            'observatory': observatory_name,
            'energy_range': energy_range,
            'observatory_info': observatory,
            'lorentz_results': lorentz_results,
            'dispersion_results': dispersion_results,
            'threshold_results': threshold_results,
            'composition_results': composition_results,
            'detectable_counts': {
                'lorentz_violation': detectable_lorentz,
                'dispersion_relation': detectable_dispersion,
                'threshold_effects': detectable_threshold,
                'composition_effects': detectable_composition
            },
            'overall_sensitivity': overall_sensitivity,
            'total_measurements': total_measurements
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive cosmic ray QG analysis.
        
        Returns:
        --------
        Dict
            Comprehensive analysis results
        """
        print("\n" + "="*60)
        print("COSMIC RAY QUANTUM GRAVITY ANALYSIS")
        print("="*60)
        
        # Analyze all observatories
        observatory_results = {}
        
        for observatory_name in self.observatories.keys():
            print(f"\nAnalyzing {observatory_name}...")
            observatory = self.observatories[observatory_name]
            energy_range = observatory['energy_range']
            
            result = self.evaluate_observatory_sensitivity(observatory_name, energy_range)
            observatory_results[observatory_name] = result
        
        # Summary statistics
        total_detectable = 0
        total_measurements = 0
        
        for observatory_name, result in observatory_results.items():
            for effect_type, count in result['detectable_counts'].items():
                total_detectable += count
                total_measurements += result['total_measurements']
        
        overall_detection_rate = total_detectable / total_measurements if total_measurements > 0 else 0
        
        print(f"\n" + "="*60)
        print("COSMIC RAY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total measurements analyzed: {total_measurements}")
        print(f"Total potentially detectable effects: {total_detectable}")
        print(f"Overall detection rate: {overall_detection_rate:.3f}")
        print(f"Observatories analyzed: {len(observatory_results)}")
        
        # Honest assessment
        print(f"\nHONEST ASSESSMENT:")
        print(f"✅ Cosmic ray analysis provides highest energy access to QG effects")
        print(f"✅ Natural particle accelerators reach 10²⁰+ eV energies")
        print(f"✅ Multiple observatories provide complementary measurements")
        print(f"❌ QG effects remain fundamentally undetectable at current precision")
        print(f"❌ Required precision improvements: 10⁶+ orders of magnitude")
        print(f"❌ No realistic path to detection with current technology")
        
        return {
            'observatory_results': observatory_results,
            'summary': {
                'total_measurements': total_measurements,
                'total_detectable': total_detectable,
                'overall_detection_rate': overall_detection_rate,
                'observatories_analyzed': len(observatory_results)
            },
            'honest_assessment': {
                'highest_energy_access': True,
                'natural_accelerators': True,
                'complementary_measurements': True,
                'fundamentally_undetectable': True,
                'precision_improvement_needed': 1e6,
                'no_realistic_path': True
            }
        }
    
    def generate_visualization(self, results: Dict, save_path: str = "cosmic_ray_qg_analysis.png"):
        """
        Generate visualization of cosmic ray QG analysis results.
        
        Parameters:
        -----------
        results : Dict
            Analysis results from run_comprehensive_analysis
        save_path : str
            Path to save the visualization
        """
        observatory_results = results['observatory_results']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Energy ranges of observatories
        observatory_names = list(observatory_results.keys())
        energy_ranges = [observatory_results[name]['energy_range'] for name in observatory_names]
        
        for i, (name, (min_e, max_e)) in enumerate(zip(observatory_names, energy_ranges)):
            axes[0, 0].semilogx([min_e, max_e], [i, i], 'o-', linewidth=3, markersize=8, label=name)
        
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Observatory')
        axes[0, 0].set_title('Cosmic Ray Observatory Energy Ranges')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Detection rates by effect type
        effect_types = ['lorentz_violation', 'dispersion_relation', 'threshold_effects', 'composition_effects']
        detection_rates = []
        
        for effect_type in effect_types:
            total_detectable = sum(observatory_results[name]['detectable_counts'][effect_type] 
                                 for name in observatory_names)
            total_measurements = sum(observatory_results[name]['total_measurements'] 
                                   for name in observatory_names)
            detection_rate = total_detectable / total_measurements if total_measurements > 0 else 0
            detection_rates.append(detection_rate)
        
        axes[0, 1].bar(effect_types, detection_rates, color=['red', 'blue', 'green', 'orange'])
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].set_title('QG Effect Detection Rates')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Observatory sensitivity comparison
        sensitivities = [observatory_results[name]['overall_sensitivity'] for name in observatory_names]
        axes[1, 0].bar(observatory_names, sensitivities, color='purple')
        axes[1, 0].set_ylabel('Overall Sensitivity')
        axes[1, 0].set_title('Observatory Sensitivity Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Energy vs QG effect size
        # Use first observatory's results for energy scaling
        first_obs = list(observatory_results.values())[0]
        energies = [r['energy'] for r in first_obs['lorentz_results']]
        lorentz_effects = [r['lorentz_violation'] for r in first_obs['lorentz_results']]
        dispersion_effects = [r['qg_correction'] for r in first_obs['dispersion_results']]
        
        axes[1, 1].loglog(energies, lorentz_effects, 'ro-', label='Lorentz Violation', linewidth=2)
        axes[1, 1].loglog(energies, dispersion_effects, 'bo-', label='Dispersion Relation', linewidth=2)
        axes[1, 1].set_xlabel('Energy (eV)')
        axes[1, 1].set_ylabel('QG Effect Size')
        axes[1, 1].set_title('Energy Scaling of QG Effects')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {save_path}")


def main():
    """Run cosmic ray QG analysis."""
    print("Cosmic Ray Quantum Gravity Detection Analysis")
    print("=" * 60)
    
    # Initialize cosmic ray detection
    cr_detection = CosmicRayDetection()
    
    # Run comprehensive analysis
    results = cr_detection.run_comprehensive_analysis()
    
    # Generate visualization
    cr_detection.generate_visualization(results)
    
    print("\n🎉 Cosmic ray QG analysis completed!")
    print("The framework provides theoretical predictions for cosmic ray QG detection.")


if __name__ == "__main__":
    main() 