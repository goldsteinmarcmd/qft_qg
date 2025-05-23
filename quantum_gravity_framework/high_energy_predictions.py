"""
High Energy Predictions for Quantum Gravity

This module provides enhanced numerical simulations that produce specific, testable
predictions at experimentally accessible energy scales, including LHC particle collisions
and gravitational wave signatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import warnings

# Fix imports for local testing
try:
    from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG
    from quantum_gravity_framework.numerical_simulations import TensorNetworkStates
except ImportError:
    from dimensional_flow_rg import DimensionalFlowRG
    from numerical_simulations import TensorNetworkStates


class HighEnergyPredictions:
    """
    Generates testable predictions for quantum gravity at currently accessible energies.
    """
    
    def __init__(self, planck_scale=1.22e19, transition_scale=1e4, dimensional_flow=True):
        """
        Initialize the high energy predictions framework.
        
        Parameters:
        -----------
        planck_scale : float
            Planck energy scale in GeV
        transition_scale : float
            Transition scale (in GeV) where quantum gravity effects become significant
        dimensional_flow : bool
            Whether to include dimensional flow effects
        """
        self.planck_scale = planck_scale  # GeV
        self.transition_scale = transition_scale  # GeV
        self.dimensional_flow = dimensional_flow
        
        # Standard Model parameters (approximate)
        self.sm_params = {
            'g_strong': 1.2,  # Strong coupling
            'g_weak': 0.65,   # Weak coupling
            'g_em': 0.3,      # Electromagnetic coupling
            'higgs_mass': 125.0,  # GeV
            'top_mass': 173.0,     # GeV
        }
        
        # Initialize RG flow model if using dimensional flow
        if dimensional_flow:
            self.rg_model = DimensionalFlowRG(
                dim_uv=2.0,
                dim_ir=4.0,
                transition_scale=transition_scale / planck_scale  # Convert to Planck units
            )
            
            # Compute RG flow across energy scales
            self.compute_rg_flow()
        
        # Store prediction results
        self.predictions = {}
    
    def compute_rg_flow(self, e_min=1.0, e_max=1e16, num_points=100):
        """
        Compute RG flow across energy scales.
        
        Parameters:
        -----------
        e_min : float
            Minimum energy in GeV
        e_max : float
            Maximum energy in GeV
        num_points : int
            Number of points to compute
            
        Returns:
        --------
        dict
            RG flow results
        """
        if not self.dimensional_flow:
            return None
        
        # Convert to Planck units
        scale_min = e_min / self.planck_scale
        scale_max = e_max / self.planck_scale
        
        # Compute RG flow
        return self.rg_model.compute_rg_flow(
            scale_range=(scale_min, scale_max),
            num_points=num_points
        )
    
    def running_couplings(self, energy_gev):
        """
        Get running couplings at a given energy.
        
        Parameters:
        -----------
        energy_gev : float or array
            Energy in GeV
            
        Returns:
        --------
        dict
            Running coupling values
        """
        if not self.dimensional_flow or not self.rg_model.flow_results:
            # Default 1-loop running for QCD
            alpha_s = 0.118 / (1 + 0.118 * (11 - 2/3 * 6) * np.log(energy_gev/91.0) / (2 * np.pi))
            return {
                'g_strong': np.sqrt(4 * np.pi * alpha_s),
                'g_weak': self.sm_params['g_weak'],
                'g_em': self.sm_params['g_em'],
            }
        
        # Convert energy to Planck units
        energy_planck = energy_gev / self.planck_scale
        
        # Get RG flow results
        scales = self.rg_model.flow_results['scales']
        coupling_trajectories = self.rg_model.flow_results['coupling_trajectories']
        
        # Use log-interpolation to find coupling values at the requested energy
        result = {}
        
        # Map our RG model couplings to SM couplings
        # This is a simplified mapping for illustration
        if 'g' in coupling_trajectories:
            # Interpolate in log space
            log_scales = np.log(scales)
            log_energy = np.log(energy_planck)
            
            if np.min(scales) <= energy_planck <= np.max(scales):
                interp_g = interp1d(log_scales, coupling_trajectories['g'])
                g_value = float(interp_g(log_energy))
                
                # Map to QCD coupling with a transition factor
                trans_factor = 1.0 / (1.0 + (energy_gev / self.transition_scale)**2)
                alpha_s_std = 0.118 / (1 + 0.118 * (11 - 2/3 * 6) * np.log(energy_gev/91.0) / (2 * np.pi))
                alpha_s_qg = g_value**2 / (4 * np.pi)
                
                alpha_s = alpha_s_std * (1 - trans_factor) + alpha_s_qg * trans_factor
                result['g_strong'] = np.sqrt(4 * np.pi * alpha_s)
            else:
                # Outside interpolation range, use standard running
                alpha_s = 0.118 / (1 + 0.118 * (11 - 2/3 * 6) * np.log(energy_gev/91.0) / (2 * np.pi))
                result['g_strong'] = np.sqrt(4 * np.pi * alpha_s)
        else:
            # Default if 'g' not found
            alpha_s = 0.118 / (1 + 0.118 * (11 - 2/3 * 6) * np.log(energy_gev/91.0) / (2 * np.pi))
            result['g_strong'] = np.sqrt(4 * np.pi * alpha_s)
        
        # For other couplings, use simplified model
        result['g_weak'] = self.sm_params['g_weak']
        result['g_em'] = self.sm_params['g_em']
        
        return result
    
    def compute_lhc_crosssections(self, process='gg_higgs', cm_energy=13000):
        """
        Compute cross sections for LHC processes with quantum gravity corrections.
        
        Parameters:
        -----------
        process : str
            Process to compute ('gg_higgs', 'ttbar', 'jets', 'diphoton')
        cm_energy : float
            Center of mass energy in GeV
            
        Returns:
        --------
        dict
            Cross section predictions
        """
        print(f"Computing {process} cross section at √s = {cm_energy/1000:.1f} TeV...")
        
        # Get relevant couplings at the process energy scale
        process_scale = {
            'gg_higgs': self.sm_params['higgs_mass'],
            'ttbar': 2 * self.sm_params['top_mass'],
            'jets': cm_energy / 4,  # Typical jet pT scale
            'diphoton': cm_energy / 3
        }.get(process, cm_energy / 2)
        
        couplings = self.running_couplings(process_scale)
        
        # Dimensional flow effects
        dim_effects = 1.0
        if self.dimensional_flow and hasattr(self.rg_model, 'compute_spectral_dimension'):
            # Convert to Planck units
            scale_planck = process_scale / self.planck_scale
            dim = self.rg_model.compute_spectral_dimension(scale_planck)
            
            # Effect is stronger as dimension deviates from 4
            dim_effects = (4.0 / dim)**2
        
        # Simplified cross section calculations
        # In reality, these would be more sophisticated QFT calculations
        
        # Standard Model baseline cross sections at 13 TeV (approximate)
        sm_xsecs = {
            'gg_higgs': 50.0,    # pb
            'ttbar': 900.0,      # pb
            'jets': 1e5,         # pb (very approximate, depends on pT cuts)
            'diphoton': 40.0,    # pb
        }
        
        # Correction factors from quantum gravity
        # Stronger at high energy/momentum transfer
        lambda_ratio = (process_scale / self.transition_scale)**2
        qg_correction = 1.0 + 0.1 * lambda_ratio * dim_effects
        
        # Apply correction to SM cross section
        xsec_sm = sm_xsecs.get(process, 0.0)
        
        # Scale with CM energy if not at 13 TeV
        if cm_energy != 13000:
            energy_scaling = (cm_energy / 13000.0)
            if process == 'gg_higgs':
                # Higgs production scales more weakly with energy
                xsec_sm *= energy_scaling**0.5
            else:
                # Other processes scale more directly with energy
                xsec_sm *= energy_scaling
        
        # Apply QG correction
        xsec_with_qg = xsec_sm * qg_correction
        
        # Get uncertainty estimates
        uncertainty_sm = 0.1 * xsec_sm  # 10% SM uncertainty
        uncertainty_qg = 0.5 * (xsec_with_qg - xsec_sm)  # 50% uncertainty on QG correction
        
        # Total uncertainty (add in quadrature)
        total_uncertainty = np.sqrt(uncertainty_sm**2 + uncertainty_qg**2)
        
        # Results
        results = {
            'process': process,
            'cm_energy': cm_energy,
            'process_scale': process_scale,
            'xsec_sm': xsec_sm,
            'xsec_with_qg': xsec_with_qg,
            'qg_correction': qg_correction,
            'uncertainty': total_uncertainty,
            'spectral_dimension': dim if self.dimensional_flow else 4.0,
            'couplings': couplings
        }
        
        print(f"  SM cross section: {xsec_sm:.3f} pb")
        print(f"  With QG effects: {xsec_with_qg:.3f} pb")
        print(f"  QG correction factor: {qg_correction:.6f}")
        print(f"  Uncertainty: ± {total_uncertainty:.3f} pb")
        
        return results
    
    def compute_gw_modifications(self, detector="LIGO", source="BH_merger"):
        """
        Compute quantum gravity modifications to gravitational wave signals.
        
        Parameters:
        -----------
        detector : str
            Gravitational wave detector ("LIGO", "LISA", "ET")
        source : str
            GW source type ("BH_merger", "NS_merger", "primordial")
            
        Returns:
        --------
        dict
            Gravitational wave predictions
        """
        print(f"Computing QG effects on {source} gravitational waves for {detector}...")
        
        # Frequency ranges for different detectors (Hz)
        freq_ranges = {
            "LIGO": (10, 1000),
            "LISA": (1e-4, 1),
            "ET": (1, 10000)
        }
        
        # Typical energy scales for different GW sources (GeV)
        energy_scales = {
            "BH_merger": 1e-13,  # Schwarzschild radius scale for ~30 solar mass BH
            "NS_merger": 1e-11,  # Nuclear density scale
            "primordial": 1e16    # Early universe scale
        }
        
        # Get frequency range for this detector
        f_min, f_max = freq_ranges.get(detector, (10, 1000))
        
        # Generate frequency array
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 100)
        
        # Get energy scale for this source
        base_energy = energy_scales.get(source, 1e-13)
        
        # Compute dispersion relation modification
        # In standard GR: E² = p² (with c=1)
        # In QG models: E² = p² (1 + (p/M)^α)
        
        # Create dispersion modifications
        dispersions = np.ones_like(frequencies)
        group_velocities = np.ones_like(frequencies)
        phase_shifts = np.zeros_like(frequencies)
        
        # QG modifications depend on the dimensionality of spacetime
        if self.dimensional_flow:
            # Convert amplitude to energy
            energies = base_energy * (frequencies / f_min)
            energies_planck = energies / self.planck_scale
            
            dimensions = np.array([
                self.rg_model.compute_spectral_dimension(e) for e in energies_planck
            ])
            
            # Scaling exponent depends on the deviation from 4D
            alpha = 2.0 * (4.0 - dimensions)
            
            # LIV parameter depends on the deviation from 4D
            liv_strength = 0.1 * (4.0 - dimensions) * (base_energy / self.transition_scale)
            
            # Compute dispersion modification
            dispersions = 1.0 + liv_strength * (frequencies / f_min)**(alpha)
            
            # Compute group velocity: v_g = dω/dk
            # Simplified calculation
            group_velocities = 1.0 + liv_strength * alpha * (frequencies / f_min)**(alpha) / 2.0
            
            # Phase shift accumulates as Δφ = 2π ∫(v_p - v_g)·f·dt
            # Distance factor (in seconds) - e.g., 100 Mpc for BH merger
            distance_time = 1e8  # seconds
            
            # Phase velocity v_p = ω/k
            phase_vel = dispersions**0.5
            phase_shifts = 2 * np.pi * frequencies * distance_time * (phase_vel - group_velocities)
        
        # Store results
        results = {
            'detector': detector,
            'source': source, 
            'frequencies': frequencies,
            'base_energy': base_energy,
            'dispersions': dispersions,
            'group_velocities': group_velocities,
            'phase_shifts': phase_shifts
        }
        
        if self.dimensional_flow:
            results['spectral_dimensions'] = dimensions
        
        # Print a summary
        print(f"  Frequency range: {f_min} - {f_max} Hz")
        print(f"  Base energy scale: {base_energy} GeV")
        print(f"  Maximum dispersion modification: {np.max(np.abs(dispersions-1)):.2e}")
        print(f"  Maximum group velocity deviation: {np.max(np.abs(group_velocities-1)):.2e}")
        
        return results
    
    def predict_mass_hierarchy(self):
        """
        Predict fermion mass hierarchy from quantum gravity effects.
        
        Returns:
        --------
        dict
            Mass hierarchy predictions
        """
        print("Computing fermion mass hierarchy predictions...")
        
        # Observed fermion masses in GeV
        observed_masses = {
            'up': 0.0022,
            'down': 0.0047,
            'charm': 1.27,
            'strange': 0.095,
            'top': 173.0,
            'bottom': 4.18,
            'electron': 0.000511,
            'muon': 0.1057,
            'tau': 1.777,
            'nu_e': 1e-12,  # Approximate upper bound
            'nu_mu': 1e-12,
            'nu_tau': 1e-12
        }
        
        # Our model's prediction factor
        # This could depend on quantum gravity corrections to Yukawa couplings
        if self.dimensional_flow:
            # Get dimensions at relevant energy scales for each fermion
            dims = {}
            ratios = {}
            
            # Reference scale (top quark mass)
            ref_scale = observed_masses['top'] / self.planck_scale
            ref_dim = self.rg_model.compute_spectral_dimension(ref_scale)
            
            for fermion, mass in observed_masses.items():
                # Mass scale normalized to Planck scale
                scale = mass / self.planck_scale
                
                # Get spectral dimension at this scale
                dims[fermion] = self.rg_model.compute_spectral_dimension(scale)
                
                # Dimension-dependent mass scaling
                if fermion.startswith('nu_'):
                    # Special case for neutrinos
                    ratios[fermion] = np.exp(-(ref_dim - dims[fermion])**2 * 5.0)
                else:
                    # For charged fermions
                    ratios[fermion] = np.exp(-(ref_dim - dims[fermion]) * 3.0)
            
            # Predicted ratios based on our model
            predicted_ratios = {f: r for f, r in ratios.items()}
        else:
            # Simplified model without dimensional flow
            # Here we might use another pattern like exponential hierarchy
            predicted_ratios = {
                f: np.exp(-i)
                for i, f in enumerate(sorted(observed_masses.keys(), 
                                           key=lambda x: observed_masses[x], 
                                           reverse=True))
            }
        
        # Calculate predicted masses (normalizing to top mass)
        predicted_masses = {
            f: observed_masses['top'] * predicted_ratios[f] / predicted_ratios['top']
            for f in observed_masses
        }
        
        # Calculate deviations
        deviations = {
            f: np.log10(predicted_masses[f] / observed_masses[f])
            for f in observed_masses
        }
        
        # Results
        results = {
            'observed_masses': observed_masses,
            'predicted_masses': predicted_masses,
            'predicted_ratios': predicted_ratios,
            'deviations': deviations
        }
        
        if self.dimensional_flow:
            results['dimensions'] = dims
        
        # Print summary
        print("\nFermion Mass Hierarchy Predictions:")
        print("  Fermion     Observed (GeV)    Predicted (GeV)    log10(Ratio)")
        print("  -------     --------------    --------------     -----------")
        
        for fermion in sorted(observed_masses.keys(), key=lambda x: observed_masses[x], reverse=True):
            print(f"  {fermion:<10} {observed_masses[fermion]:<16.6e} {predicted_masses[fermion]:<16.6e} {deviations[fermion]:>11.2f}")
        
        return results
    
    def plot_lhc_predictions(self, processes=None, save_path=None):
        """
        Plot LHC cross section predictions.
        
        Parameters:
        -----------
        processes : list, optional
            List of processes to include
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if processes is None:
            processes = ['gg_higgs', 'ttbar', 'jets', 'diphoton']
        
        # Compute cross sections for each process if not already done
        results = {}
        for process in processes:
            if process not in self.predictions:
                self.predictions[process] = self.compute_lhc_crosssections(process=process)
            results[process] = self.predictions[process]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up bar positions
        x_pos = np.arange(len(processes))
        width = 0.35
        
        # Plot bars for SM and SM+QG predictions
        sm_bars = [results[p]['xsec_sm'] for p in processes]
        qg_bars = [results[p]['xsec_with_qg'] for p in processes]
        
        # Convert very large cross sections to more readable units
        units = []
        for i, (sm, qg) in enumerate(zip(sm_bars, qg_bars)):
            if max(sm, qg) > 1000:
                sm_bars[i] /= 1000
                qg_bars[i] /= 1000
                units.append('nb')
            else:
                units.append('pb')
        
        # Plot bars
        ax.bar(x_pos - width/2, sm_bars, width, label='Standard Model', alpha=0.7, color='blue')
        ax.bar(x_pos + width/2, qg_bars, width, label='With QG Effects', alpha=0.7, color='red')
        
        # Add error bars for QG predictions
        qg_errors = [results[p]['uncertainty'] / (1000 if u == 'nb' else 1) for p, u in zip(processes, units)]
        ax.errorbar(x_pos + width/2, qg_bars, yerr=qg_errors, fmt='o', color='black')
        
        # Set labels and title
        ax.set_xlabel('Process')
        ax.set_ylabel('Cross Section')
        ax.set_title('LHC Cross Section Predictions')
        ax.set_xticks(x_pos)
        
        # Create process labels with units
        labels = [f"{p} [{u}]" for p, u in zip(processes, units)]
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gw_predictions(self, save_path=None):
        """
        Plot gravitational wave predictions.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Compute GW predictions for different detectors if not already done
        detectors = ["LIGO", "LISA", "ET"]
        source = "BH_merger"
        
        results = {}
        for detector in detectors:
            key = f"gw_{detector}_{source}"
            if key not in self.predictions:
                self.predictions[key] = self.compute_gw_modifications(detector=detector, source=source)
            results[detector] = self.predictions[key]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Dispersion relation modification
        for detector in detectors:
            freqs = results[detector]['frequencies']
            disp = results[detector]['dispersions']
            axs[0].semilogx(freqs, disp, '-', linewidth=2, label=detector)
        
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Dispersion Relation Modification')
        axs[0].set_title(f'Quantum Gravity Effects on {source} Gravitational Waves')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # Plot 2: Phase shift
        for detector in detectors:
            freqs = results[detector]['frequencies']
            phase = results[detector]['phase_shifts']
            axs[1].semilogx(freqs, phase, '-', linewidth=2, label=detector)
        
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase Shift (radians)')
        axs[1].set_title('GW Phase Shift Due to Quantum Gravity')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test the high energy predictions framework
    
    # Create a predictions instance
    predictions = HighEnergyPredictions(
        planck_scale=1.22e19,  # GeV
        transition_scale=1e4,   # GeV
        dimensional_flow=True
    )
    
    # Compute LHC cross sections
    higgs_results = predictions.compute_lhc_crosssections(process='gg_higgs', cm_energy=13000)
    ttbar_results = predictions.compute_lhc_crosssections(process='ttbar', cm_energy=13000)
    
    # Compute gravitational wave modifications
    ligo_results = predictions.compute_gw_modifications(detector="LIGO", source="BH_merger")
    
    # Predict fermion mass hierarchy
    mass_hierarchy = predictions.predict_mass_hierarchy()
    
    # Plot predictions
    predictions.plot_lhc_predictions(save_path="lhc_predictions.png")
    predictions.plot_gw_predictions(save_path="gw_predictions.png")
    
    print("\nHigh energy predictions complete.") 