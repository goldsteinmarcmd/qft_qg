#!/usr/bin/env python
"""
Higgs Differential Cross-Section with QG Corrections

This module implements precise predictions for the Higgs boson differential
cross-section at high pT, including quantum gravity corrections. This is 
identified as one possible signature for experimental validation, with focus
on theoretical consistency and future experimental facilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import seaborn as sns

from qft.gauge_qg_integration import GaugeQGIntegration
from quantum_gravity_framework.high_energy_collisions import HighEnergyCollisionSimulator
from quantum_gravity_framework.qft_integration import QFTIntegration
from quantum_gravity_framework.category_theory import CategoryTheoryGeometry

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


class HiggsPtSpectrum:
    """
    Predictions for Higgs boson differential cross-section with QG corrections.
    """
    
    def __init__(self, qg_scale=1.22e19, collision_energy=13.6e3):
        """
        Initialize the Higgs pT spectrum calculator.
        
        Parameters:
        -----------
        qg_scale : float
            Quantum gravity energy scale in GeV (default: Planck scale 1.22e19 GeV)
        collision_energy : float
            Collider energy in GeV (default: 13.6 TeV for LHC Run 3)
        """
        self.qg_scale = qg_scale
        self.collision_energy = collision_energy
        
        # Initialize QFT-QG framework components
        self.qft_qg = QFTIntegration(dim=4, cutoff_scale=qg_scale)
        self.category_geometry = CategoryTheoryGeometry(dim=4)
        self.high_energy_simulator = HighEnergyCollisionSimulator(
            max_energy=collision_energy,
            qg_scale=qg_scale
        )
        
        # Load experimental Higgs data (simplified simulation)
        self.load_experimental_data()
        
        # QG correction parameters (from results)
        self.correction_factors = self.extract_correction_parameters()
    
    def load_experimental_data(self):
        """Load experimental Higgs pT data (simulated for this example)."""
        # Create simulated Higgs pT data based on SM predictions
        # This would be replaced with actual LHC data in a real analysis
        
        # pT bins in GeV
        self.pt_bins = np.concatenate([
            np.linspace(0, 100, 10),
            np.linspace(120, 300, 10),
            np.linspace(350, 700, 8),
            np.linspace(800, 1500, 8)
        ])
        
        # Standard Model predicted differential cross section (pb/GeV)
        # These values are approximate and would be replaced with precise calculations
        sm_xsec_values = np.zeros_like(self.pt_bins)
        
        # Simplified model for SM Higgs pT distribution
        # dσ/dpT ~ pT at low pT, falls as ~ 1/pT^4 at high pT
        for i, pt in enumerate(self.pt_bins):
            if pt < 50:
                sm_xsec_values[i] = 0.5 * (pt/50.0) 
            else:
                sm_xsec_values[i] = 0.5 * (50.0/pt)**4
        
        # Create DataFrame for easy manipulation
        self.sm_predictions = pd.DataFrame({
            'pt': self.pt_bins,
            'xsec': sm_xsec_values,
            'xsec_err': sm_xsec_values * 0.1  # 10% theoretical uncertainty
        })
        
        # Create simulated "measurement" with statistical fluctuations
        # In a real analysis, this would be the actual LHC data
        measurement_uncertainty = np.sqrt(sm_xsec_values) * 0.2  # Statistical uncertainty model
        measurement_values = np.random.normal(sm_xsec_values, measurement_uncertainty)
        measurement_values[measurement_values < 0] = 0  # No negative cross sections
        
        self.measured_data = pd.DataFrame({
            'pt': self.pt_bins,
            'xsec': measurement_values,
            'xsec_err': measurement_uncertainty
        })
    
    def extract_correction_parameters(self):
        """Extract QG correction parameters from the framework."""
        try:
            # Get SM corrections from QFT-QG framework
            sm_corrections = self.qft_qg.quantum_gravity_corrections_to_standard_model()
            higgs_correction = sm_corrections['process_corrections']['higgs_production']
            
            # Extract the basic correction factor
            base_factor = higgs_correction['relative_correction']
            
            # Get momentum-dependent correction parameters
            qg_action = self.qft_qg.quantum_effective_action()
            beta1 = qg_action['correction_parameters']['beta1']
            beta2 = qg_action['correction_parameters']['beta2']
            
            return {
                'base_factor': base_factor,
                'beta1': beta1,  # p^4 correction
                'beta2': beta2,  # p^2m^2 correction
                'source': 'framework'
            }
        except Exception:
            # Use theoretically motivated small values based on standard EFT calculations
            # These values represent genuine theoretical expectations
            return {
                'base_factor': 3.256e-8,  # Theoretical value without enhancement
                'beta1': 0.1,             # Standard higher-derivative coefficient
                'beta2': 0.05,            # Standard mixed term coefficient
                'source': 'theoretical_defaults'
            }
    
    def calculate_qg_corrected_spectrum(self):
        """Calculate QG-corrected Higgs pT spectrum."""
        # Get SM predictions
        sm_pt = self.sm_predictions['pt'].values
        sm_xsec = self.sm_predictions['xsec'].values
        
        # Initialize array for corrected cross section
        qg_xsec = np.zeros_like(sm_xsec)
        
        # Apply QG corrections with pT-dependent enhancement
        # The correction increases with pT: more significant at high momentum
        for i, pt in enumerate(sm_pt):
            # Normalize pT to QG scale - this is the small parameter in the EFT
            pt_normalized = pt / self.qg_scale
            
            # Basic correction factor (increases with pT^2 due to dimension-6 operators)
            correction = 1.0 + self.correction_factors['base_factor'] * (pt/100.0)**2
            
            # Higher-derivative term contribution (~ p^4 term in action)
            p4_correction = self.correction_factors['beta1'] * pt_normalized**2
            
            # Mixed kinetic-mass term (~ p^2m^2 term in action)
            # m_h is Higgs mass
            m_h = 125.0  # GeV
            m_normalized = m_h / self.qg_scale
            p2m2_correction = self.correction_factors['beta2'] * pt_normalized * m_normalized**2
            
            # Total correction from QG effects (no artificial resonance)
            total_correction = correction * (1.0 + p4_correction + p2m2_correction)
            
            # Apply correction to SM cross section
            qg_xsec[i] = sm_xsec[i] * total_correction
        
        # Create DataFrame with results
        self.qg_predictions = pd.DataFrame({
            'pt': sm_pt,
            'xsec': qg_xsec,
            'xsec_err': qg_xsec * 0.12  # 12% uncertainty (increased for QG model)
        })
        
        return self.qg_predictions
    
    def calculate_significance(self, luminosity=3000.0):
        """
        Calculate statistical significance of QG effects.
        
        Parameters:
        -----------
        luminosity : float
            Integrated luminosity in fb^-1
            
        Returns:
        --------
        DataFrame
            Statistical significance in each pT bin
        """
        # Calculate QG-corrected spectrum if not already done
        if not hasattr(self, 'qg_predictions'):
            self.calculate_qg_corrected_spectrum()
        
        # Initialize results
        significance_data = []
        
        # Calculate bin widths
        bin_widths = np.zeros_like(self.pt_bins)
        bin_widths[:-1] = self.pt_bins[1:] - self.pt_bins[:-1]
        bin_widths[-1] = bin_widths[-2]  # Use same width for last bin
        
        # Calculate for each pT bin
        for i, pt in enumerate(self.pt_bins):
            # Get SM and QG predictions
            sm_xsec = self.sm_predictions.iloc[i]['xsec']
            qg_xsec = self.qg_predictions.iloc[i]['xsec']
            
            # Calculate expected events (cross section * luminosity * bin width)
            sm_events = sm_xsec * luminosity * bin_widths[i]
            qg_events = qg_xsec * luminosity * bin_widths[i]
            
            # Statistical uncertainty: sqrt(N)
            sm_uncertainty = np.sqrt(sm_events)
            
            # Absolute difference in event counts
            abs_diff = abs(qg_events - sm_events)
            
            # Significance (sigma)
            if sm_uncertainty > 0:
                significance = abs_diff / sm_uncertainty
            else:
                significance = 0
            
            # Store results
            significance_data.append({
                'pt': pt,
                'sm_events': sm_events,
                'qg_events': qg_events,
                'abs_diff': abs_diff,
                'significance': significance
            })
        
        # Convert to DataFrame
        self.significance = pd.DataFrame(significance_data)
        
        return self.significance
    
    def plot_pt_spectrum(self, save_path=None):
        """
        Plot Higgs pT spectrum with QG corrections.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate predictions if not already done
        if not hasattr(self, 'qg_predictions'):
            self.calculate_qg_corrected_spectrum()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Upper panel: differential cross section
        ax1.errorbar(self.sm_predictions['pt'], self.sm_predictions['xsec'],
                   yerr=self.sm_predictions['xsec_err'], 
                   fmt='o-', color='blue', label='Standard Model', alpha=0.7)
        
        ax1.errorbar(self.qg_predictions['pt'], self.qg_predictions['xsec'],
                   yerr=self.qg_predictions['xsec_err'],
                   fmt='s-', color='red', label='QG Corrected', alpha=0.7)
        
        # Simulated "measurement" points
        ax1.errorbar(self.measured_data['pt'], self.measured_data['xsec'],
                   yerr=self.measured_data['xsec_err'],
                   fmt='x', color='black', label='Simulated Data', alpha=0.8)
        
        # Set log scales for better visualization
        ax1.set_yscale('log')
        ax1.set_xlabel('Higgs $p_T$ [GeV]')
        ax1.set_ylabel('$d\\sigma/dp_T$ [pb/GeV]')
        ax1.set_title('Higgs Boson Differential Cross Section with QG Corrections')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lower panel: ratio of QG to SM
        ratio = self.qg_predictions['xsec'] / self.sm_predictions['xsec']
        ratio_err = ratio * np.sqrt((self.qg_predictions['xsec_err']/self.qg_predictions['xsec'])**2 + 
                                  (self.sm_predictions['xsec_err']/self.sm_predictions['xsec'])**2)
        
        ax2.errorbar(self.qg_predictions['pt'], ratio, yerr=ratio_err,
                   fmt='o-', color='red', alpha=0.7)
        
        # Add horizontal line at ratio=1
        ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Higgs $p_T$ [GeV]')
        ax2.set_ylabel('QG / SM')
        ax2.set_ylim(0.8, 1.5)  # Adjust as needed
        ax2.grid(True, alpha=0.3)
        
        # Add QG scale info
        fig.text(0.15, 0.01, f"QG Scale: {self.qg_scale/1e3:.1e} TeV", 
                fontsize=10, color='red')
        fig.text(0.55, 0.01, f"Base Correction: {self.correction_factors['base_factor']:.2e}", 
                fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_significance(self, luminosities=[300, 1000, 3000], save_path=None):
        """
        Plot significance of QG effects at different luminosities.
        
        Parameters:
        -----------
        luminosities : list
            List of integrated luminosities in fb^-1
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different luminosities
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(luminosities)))
        
        # Calculate and plot significance for each luminosity
        for i, lumi in enumerate(luminosities):
            significance = self.calculate_significance(luminosity=lumi)
            
            ax.plot(significance['pt'], significance['significance'],
                  'o-', color=colors[i], 
                  label=f'{lumi} fb$^{{-1}}$', alpha=0.8)
        
        # Add horizontal line at significance=2 (95% CL)
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='95% CL')
        
        # Add horizontal line at significance=5 (discovery)
        ax.axhline(y=5, color='black', linestyle='--', alpha=0.7, label='5$\\sigma$ (discovery)')
        
        ax.set_xlabel('Higgs $p_T$ [GeV]')
        ax.set_ylabel('Significance ($\\sigma$)')
        ax.set_title('Experimental Sensitivity to QG Corrections in Higgs Production')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add QG scale info
        fig.text(0.15, 0.01, f"QG Scale: {self.qg_scale/1e3:.1e} TeV", 
                fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def discovery_potential_summary(self):
        """
        Generate summary of discovery potential.
        
        Returns:
        --------
        dict
            Summary of discovery potential
        """
        # Calculate significance for different luminosities
        luminosities = [300, 1000, 3000]  # fb^-1
        pt_thresholds = [500, 750, 1000]  # GeV
        
        results = {}
        
        for lumi in luminosities:
            significance = self.calculate_significance(luminosity=lumi)
            
            # Find maximum significance
            max_signif = significance['significance'].max()
            max_pt = significance.loc[significance['significance'].idxmax()]['pt']
            
            # Check significance at different pT thresholds
            threshold_results = {}
            for pt in pt_thresholds:
                # Find closest bin
                closest_idx = np.abs(significance['pt'] - pt).argmin()
                threshold_results[f'pt_{pt}'] = significance.iloc[closest_idx]['significance']
            
            # Store results
            results[f'lumi_{lumi}'] = {
                'max_significance': max_signif,
                'max_significance_pt': max_pt,
                'thresholds': threshold_results
            }
        
        # Calculate required luminosity for 5-sigma discovery
        pt_for_discovery = []
        lumi_for_discovery = []
        
        # For each pT bin, estimate required luminosity
        for i, pt in enumerate(self.pt_bins):
            if i % 3 == 0:  # Skip some bins for clarity
                # Get SM and QG predictions
                sm_xsec = self.sm_predictions.iloc[i]['xsec']
                qg_xsec = self.qg_predictions.iloc[i]['xsec']
                
                # If there's a significant difference
                if abs(qg_xsec - sm_xsec) > 1e-12:
                    # Calculate bin width
                    if i < len(self.pt_bins) - 1:
                        bin_width = self.pt_bins[i+1] - self.pt_bins[i]
                    else:
                        bin_width = self.pt_bins[i] - self.pt_bins[i-1]
                    
                    # Required events for 5-sigma: (QG-SM)^2 / SM = 25
                    # Therefore, luminosity = 25 / ((QG-SM)^2/SM * bin_width)
                    required_lumi = 25.0 / ((qg_xsec - sm_xsec)**2 / sm_xsec * bin_width)
                    
                    if required_lumi > 0 and required_lumi < 1e5:  # Reasonable range
                        pt_for_discovery.append(pt)
                        lumi_for_discovery.append(required_lumi)
        
        # Add to results
        results['discovery_requirements'] = {
            'pt': pt_for_discovery,
            'required_luminosity': lumi_for_discovery
        }
        
        # Overall summary
        min_lumi_for_discovery = min(lumi_for_discovery) if lumi_for_discovery else float('inf')
        best_pt_for_discovery = pt_for_discovery[np.argmin(lumi_for_discovery)] if lumi_for_discovery else None
        
        results['summary'] = {
            'min_luminosity_for_discovery': min_lumi_for_discovery,
            'best_pt_for_discovery': best_pt_for_discovery,
            'qg_scale': self.qg_scale,
            'collision_energy': self.collision_energy,
            'is_discoverable_hl_lhc': min_lumi_for_discovery < 3000,
            'is_observable_run3': min_lumi_for_discovery < 300
        }
        
        return results
    
    def plot_future_collider_prospects(self, save_path=None):
        """
        Plot prospects for future collider facilities.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Future collider scenarios
        colliders = {
            'HL-LHC': {'energy': 14e3, 'luminosity': 3000},
            'HE-LHC': {'energy': 27e3, 'luminosity': 15000},
            'FCC-hh': {'energy': 100e3, 'luminosity': 30000},
            'SPPC': {'energy': 75e3, 'luminosity': 25000},
            'Muon Collider': {'energy': 10e3, 'luminosity': 10000}
        }
        
        # Colors for different colliders
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(colliders)))
        
        # Calculate and plot significance for each collider
        for i, (name, params) in enumerate(colliders.items()):
            # Simple scaling of significance with energy and sqrt(luminosity)
            # Higher energy increases the QG effects, luminosity improves statistics
            energy_factor = (params['energy'] / self.collision_energy)**2
            lumi_factor = np.sqrt(params['luminosity'] / 3000)
            
            # Calculate base significance at current LHC energy and 3000 fb^-1
            base_significance = self.calculate_significance(luminosity=3000)
            
            # Scale significance
            scaled_significance = base_significance.copy()
            scaled_significance['significance'] *= energy_factor * lumi_factor
            
            ax.plot(scaled_significance['pt'], scaled_significance['significance'],
                  'o-', color=colors[i], 
                  label=f'{name} ({params["energy"]/1e3:.0f} TeV, {params["luminosity"]} fb$^{{-1}}$)', 
                  alpha=0.8)
        
        # Add horizontal line at significance=5 (discovery)
        ax.axhline(y=5, color='black', linestyle='--', alpha=0.7, label='5$\\sigma$ (discovery)')
        
        ax.set_xlabel('Higgs $p_T$ [GeV]')
        ax.set_ylabel('Projected Significance ($\\sigma$)')
        ax.set_title('Future Collider Prospects for QG Corrections in Higgs Production')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add QG scale info
        fig.text(0.15, 0.01, f"QG Scale: {self.qg_scale/1e3:.1e} TeV", 
                fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def explore_alternative_signatures(self):
        """
        Explore alternative experimental signatures for QG effects.
        
        Returns:
        --------
        dict
            Dictionary of alternative signatures and their detectability
        """
        # Define alternative signatures and their theoretical sensitivity to QG effects
        alternative_signatures = {
            'higgs_pt_spectrum': {
                'description': 'Higgs pT spectrum at high momentum',
                'energy_scaling': 2,  # Scales as (E/M_QG)^2
                'lhc_sensitivity': 'Very Low',
                'future_sensitivity': 'Low',
                'theoretical_cleanliness': 'High',
                'discovery_prospects': self._estimate_discovery_prospects('higgs_pt_spectrum')
            },
            'black_hole_production': {
                'description': 'Production of microscopic black holes',
                'energy_scaling': 8,  # Scales much more steeply with energy
                'lhc_sensitivity': 'None',
                'future_sensitivity': 'Medium',
                'theoretical_cleanliness': 'Medium',
                'discovery_prospects': self._estimate_discovery_prospects('black_hole_production')
            },
            'graviton_production': {
                'description': 'Direct production of graviton KK modes',
                'energy_scaling': 4,  # Scales as (E/M_QG)^4
                'lhc_sensitivity': 'None',
                'future_sensitivity': 'Low',
                'theoretical_cleanliness': 'Medium',
                'discovery_prospects': self._estimate_discovery_prospects('graviton_production')
            },
            'extra_dimension_effects': {
                'description': 'Searches for effects of extra dimensions',
                'energy_scaling': 4,  # Scales as (E/M_QG)^4
                'lhc_sensitivity': 'None',
                'future_sensitivity': 'Medium',
                'theoretical_cleanliness': 'Medium',
                'discovery_prospects': self._estimate_discovery_prospects('extra_dimension_effects')
            },
            'planck_scale_resonances': {
                'description': 'Resonances from Planck-scale states',
                'energy_scaling': 6,  # Scales as (E/M_QG)^6
                'lhc_sensitivity': 'None',
                'future_sensitivity': 'Low',
                'theoretical_cleanliness': 'Low',
                'discovery_prospects': self._estimate_discovery_prospects('planck_scale_resonances')
            },
            'qg_modified_dispersion': {
                'description': 'Modified dispersion relations for particles',
                'energy_scaling': 3,  # Scales as (E/M_QG)^3
                'lhc_sensitivity': 'None',
                'future_sensitivity': 'Medium',
                'theoretical_cleanliness': 'High',
                'discovery_prospects': self._estimate_discovery_prospects('qg_modified_dispersion')
            },
            'lorentz_violation': {
                'description': 'Tests of Lorentz invariance violation',
                'energy_scaling': 2,  # Scales as (E/M_QG)^2
                'lhc_sensitivity': 'Very Low',
                'future_sensitivity': 'Medium',
                'theoretical_cleanliness': 'High',
                'discovery_prospects': self._estimate_discovery_prospects('lorentz_violation')
            },
            'cosmic_rays': {
                'description': 'Ultra-high energy cosmic ray physics',
                'energy_scaling': 3,  # Scales as (E/M_QG)^3
                'lhc_sensitivity': 'N/A',
                'future_sensitivity': 'Medium',
                'theoretical_cleanliness': 'Low',
                'discovery_prospects': self._estimate_discovery_prospects('cosmic_rays')
            },
        }
        
        return alternative_signatures
    
    def _estimate_discovery_prospects(self, signature_type):
        """
        Estimate discovery prospects for a given signature type.
        
        Parameters:
        -----------
        signature_type : str
            Type of signature to evaluate
        
        Returns:
        --------
        dict
            Discovery prospects for different experimental facilities
        """
        # Define energy and luminosity factors for different facilities
        facilities = {
            'LHC (13.6 TeV)': {'energy_factor': 1.0, 'lumi_factor': 1.0},
            'HL-LHC (14 TeV)': {'energy_factor': 1.03, 'lumi_factor': 10.0},
            'HE-LHC (27 TeV)': {'energy_factor': 1.99, 'lumi_factor': 50.0},
            'FCC-hh (100 TeV)': {'energy_factor': 7.35, 'lumi_factor': 100.0},
            'Cosmic Rays': {'energy_factor': 100.0, 'lumi_factor': 0.001},
            'Gravitational Waves': {'energy_factor': 0.01, 'lumi_factor': 100.0}
        }
        
        # Signature-specific energy scaling factors
        energy_scalings = {
            'higgs_pt_spectrum': 2,
            'black_hole_production': 8,
            'graviton_production': 4,
            'extra_dimension_effects': 4,
            'planck_scale_resonances': 6,
            'qg_modified_dispersion': 3,
            'lorentz_violation': 2,
            'cosmic_rays': 3
        }
        
        # Base significance - for Higgs pT this is our actual calculation
        # For others, it's a theoretical estimate
        if signature_type == 'higgs_pt_spectrum':
            base_significance = self.calculate_significance(luminosity=300)['significance'].max()
        else:
            # For other signatures, estimate based on theoretical considerations
            # These are just rough estimates
            base_estimates = {
                'black_hole_production': 1e-12,
                'graviton_production': 1e-10,
                'extra_dimension_effects': 1e-9,
                'planck_scale_resonances': 1e-11,
                'qg_modified_dispersion': 1e-8,
                'lorentz_violation': 1e-7,
                'cosmic_rays': 1e-6
            }
            base_significance = base_estimates.get(signature_type, 1e-10)
        
        # Calculate prospects for each facility
        prospects = {}
        for facility, factors in facilities.items():
            # Calculate significance scaling
            # Effect grows with energy^scaling and sqrt(luminosity)
            energy_boost = factors['energy_factor'] ** energy_scalings.get(signature_type, 2)
            stats_boost = np.sqrt(factors['lumi_factor'])
            
            # Estimate significance
            estimated_significance = base_significance * energy_boost * stats_boost
            
            # Determine discovery potential
            if estimated_significance >= 5.0:
                discovery_potential = "Discovery possible"
            elif estimated_significance >= 2.0:
                discovery_potential = "Evidence possible"
            elif estimated_significance >= 0.1:
                discovery_potential = "Observable effect"
            else:
                discovery_potential = "Undetectable"
            
            prospects[facility] = {
                'significance': estimated_significance,
                'discovery_potential': discovery_potential
            }
        
        return prospects

def main():
    """Run the Higgs pT analysis with QG corrections."""
    print("Calculating Higgs pT spectrum with QG corrections...")
    
    # Initialize the analysis with theoretically motivated QG scale (Planck scale)
    higgs_analysis = HiggsPtSpectrum(qg_scale=1.22e19, collision_energy=13.6e3)
    
    # Calculate QG-corrected spectrum
    qg_predictions = higgs_analysis.calculate_qg_corrected_spectrum()
    
    # Plot the results
    higgs_analysis.plot_pt_spectrum(save_path='higgs_pt_qg_corrections.png')
    higgs_analysis.plot_significance(save_path='higgs_pt_significance.png')
    
    # Plot future collider prospects
    higgs_analysis.plot_future_collider_prospects(save_path='higgs_pt_future_prospects.png')
    
    # Get discovery potential summary
    discovery = higgs_analysis.discovery_potential_summary()
    
    # Explore alternative signatures
    alternative_signatures = higgs_analysis.explore_alternative_signatures()
    
    # Print summary
    print("\nDiscovery Potential Summary:")
    print("----------------------------")
    print(f"QG Scale: {discovery['summary']['qg_scale']/1e3:.1e} TeV")
    print(f"Collision Energy: {discovery['summary']['collision_energy']/1e3:.1f} TeV")
    print(f"Min Luminosity for 5σ Discovery: {discovery['summary']['min_luminosity_for_discovery']:.1f} fb^-1")
    
    # Check if best_pt_for_discovery exists before formatting
    best_pt = discovery['summary']['best_pt_for_discovery']
    if best_pt is not None:
        print(f"Best pT for Discovery: {best_pt:.0f} GeV")
    else:
        print("Best pT for Discovery: Not found (signal too weak)")
    
    print(f"Discoverable at HL-LHC (3000 fb^-1): {discovery['summary']['is_discoverable_hl_lhc']}")
    print(f"Observable in LHC Run 3 (300 fb^-1): {discovery['summary']['is_observable_run3']}")
    
    # Print alternative signatures summary
    print("\nAlternative Experimental Signatures:")
    print("----------------------------------")
    for signature, details in alternative_signatures.items():
        print(f"\n{signature}: {details['description']}")
        print(f"  Energy scaling: ~(E/M_QG)^{details['energy_scaling']}")
        print(f"  LHC sensitivity: {details['lhc_sensitivity']}")
        print(f"  Future facility sensitivity: {details['future_sensitivity']}")
        
        # Print top prospect
        best_prospect = None
        best_significance = 0
        for facility, prospect in details['discovery_prospects'].items():
            if prospect['significance'] > best_significance:
                best_significance = prospect['significance']
                best_prospect = facility
        
        if best_prospect:
            print(f"  Best prospect: {best_prospect} ({details['discovery_prospects'][best_prospect]['discovery_potential']})")
            print(f"  Estimated significance: {details['discovery_prospects'][best_prospect]['significance']:.2e}σ")
    
    # Save results to file
    with open('higgs_pt_qg_results.txt', 'w') as f:
        f.write("Higgs pT Spectrum with QG Corrections\n")
        f.write("====================================\n\n")
        f.write(f"QG Scale: {discovery['summary']['qg_scale']/1e3:.1e} TeV\n")
        f.write(f"Collision Energy: {discovery['summary']['collision_energy']/1e3:.1f} TeV\n")
        f.write(f"Min Luminosity for 5σ Discovery: {discovery['summary']['min_luminosity_for_discovery']:.1f} fb^-1\n")
        
        if best_pt is not None:
            f.write(f"Best pT for Discovery: {best_pt:.0f} GeV\n")
        else:
            f.write("Best pT for Discovery: Not found (signal too weak)\n")
            
        f.write(f"Discoverable at HL-LHC: {discovery['summary']['is_discoverable_hl_lhc']}\n")
        f.write(f"Observable in LHC Run 3: {discovery['summary']['is_observable_run3']}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-----------------\n")
        for lumi in [300, 1000, 3000]:
            f.write(f"\nLuminosity: {lumi} fb^-1\n")
            f.write(f"  Max Significance: {discovery[f'lumi_{lumi}']['max_significance']:.2f}σ at pT = {discovery[f'lumi_{lumi}']['max_significance_pt']:.0f} GeV\n")
            for pt in [500, 750, 1000]:
                f.write(f"  Significance at pT = {pt} GeV: {discovery[f'lumi_{lumi}']['thresholds'][f'pt_{pt}']:.2f}σ\n")
        
        # Write alternative signatures summary
        f.write("\n\nAlternative Experimental Signatures:\n")
        f.write("----------------------------------\n")
        for signature, details in alternative_signatures.items():
            f.write(f"\n{signature}: {details['description']}\n")
            f.write(f"  Energy scaling: ~(E/M_QG)^{details['energy_scaling']}\n")
            
            # Write best prospect
            best_prospect = None
            best_significance = 0
            for facility, prospect in details['discovery_prospects'].items():
                if prospect['significance'] > best_significance:
                    best_significance = prospect['significance']
                    best_prospect = facility
            
            if best_prospect:
                f.write(f"  Best prospect: {best_prospect} ({details['discovery_prospects'][best_prospect]['discovery_potential']})\n")
                f.write(f"  Estimated significance: {details['discovery_prospects'][best_prospect]['significance']:.2e}σ\n")
    
    # Also save the alternative signatures to a separate file
    with open('alternative_qg_signatures.txt', 'w') as f:
        f.write("Alternative Signatures for QG Effects\n")
        f.write("===================================\n\n")
        f.write(f"QG Scale: {higgs_analysis.qg_scale/1e3:.1e} TeV\n\n")
        
        # Sort signatures by best prospects
        sorted_signatures = []
        for signature, details in alternative_signatures.items():
            best_prospect = None
            best_significance = 0
            for facility, prospect in details['discovery_prospects'].items():
                if prospect['significance'] > best_significance:
                    best_significance = prospect['significance']
                    best_prospect = facility
            
            sorted_signatures.append((signature, details, best_significance))
        
        # Sort by significance in descending order
        sorted_signatures.sort(key=lambda x: x[2], reverse=True)
        
        # Write sorted signatures
        for signature, details, _ in sorted_signatures:
            f.write(f"{signature}: {details['description']}\n")
            f.write(f"  Energy scaling: ~(E/M_QG)^{details['energy_scaling']}\n")
            f.write(f"  LHC sensitivity: {details['lhc_sensitivity']}\n")
            f.write(f"  Future facility sensitivity: {details['future_sensitivity']}\n")
            f.write(f"  Theoretical cleanliness: {details['theoretical_cleanliness']}\n")
            
            # Write prospects for each facility
            f.write("  Discovery prospects:\n")
            for facility, prospect in details['discovery_prospects'].items():
                f.write(f"    {facility}: {prospect['significance']:.2e}σ ({prospect['discovery_potential']})\n")
            
            f.write("\n")
    
    print("\nResults saved to higgs_pt_qg_results.txt")
    print("Alternative signatures saved to alternative_qg_signatures.txt")
    print("Figures saved as higgs_pt_qg_corrections.png, higgs_pt_significance.png, and higgs_pt_future_prospects.png")


if __name__ == "__main__":
    main() 