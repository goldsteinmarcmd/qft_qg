"""
Experimental Predictions from Quantum Gravity

This module implements concrete experimental predictions from our quantum gravity framework,
focusing on potentially observable signatures at various energy scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.stats as stats

from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG
from quantum_gravity_framework.gauge_field_integration import GaugeFieldIntegration
from quantum_gravity_framework.fermion_treatment import FermionTreatment
from quantum_gravity_framework.unified_framework import UnifiedFramework


class ExperimentalPredictions:
    """
    Generates concrete experimental predictions from the quantum gravity framework.
    """
    
    def __init__(self, dim_uv=2.0, dim_ir=4.0, transition_scale=1.0):
        """
        Initialize the experimental predictions generator.
        
        Parameters:
        -----------
        dim_uv : float
            UV (high energy) spectral dimension
        dim_ir : float
            IR (low energy) spectral dimension
        transition_scale : float
            Scale of dimension transition (in Planck units)
        """
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        
        # Initialize dimensional flow RG
        self.rg = DimensionalFlowRG(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize gauge field integration
        self.gauge = GaugeFieldIntegration(
            gauge_group="SU3",  # Default to QCD for most predictions
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize fermion treatment
        self.fermion = FermionTreatment(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize unified framework for QFT-QG interface
        self.unified = UnifiedFramework(
            dim_profile=lambda E: dim_ir - (dim_ir - dim_uv) / (1.0 + (E / transition_scale)**(-2))
        )
        
        # Store predictions
        self.predictions = {}
    
    def compute_running_couplings(self, energy_range=None, num_points=100):
        """
        Compute running couplings across a range of energies.
        
        Parameters:
        -----------
        energy_range : tuple, optional
            (min_energy, max_energy) in Planck units
        num_points : int
            Number of energy points to compute
            
        Returns:
        --------
        dict
            Running coupling predictions
        """
        print("Computing running coupling predictions...")
        
        if energy_range is None:
            # Default: from LHC energies to Planck scale
            # LHC ~ 10^-15 Planck units, Planck = 1
            energy_range = (1e-15, 1.0)
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), num_points)
        
        # Compute RG flow
        self.rg.compute_rg_flow(scale_range=energy_range)
        
        # Collect results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'coupling_trajectories': {}
        }
        
        # Get couplings at each scale
        for scale in energy_scales:
            # Get dimension at this scale
            dimension = self.rg.compute_spectral_dimension(scale)
            results['dimensions'].append(dimension)
        
        # Store coupling trajectories
        results['coupling_trajectories'] = self.rg.flow_results['coupling_trajectories']
        
        # Store complete results
        self.predictions['running_couplings'] = results
        return results
    
    def predict_lhc_deviations(self):
        """
        Predict deviations from Standard Model that might be observable at LHC energies.
        
        Returns:
        --------
        dict
            LHC deviation predictions
        """
        print("Computing LHC-scale deviation predictions...")
        
        # LHC energy ~ 10^4 GeV ~ 10^-15 Planck units
        lhc_energy = 1e-15
        
        # Compute RG flow if not already computed
        if not self.rg.flow_results:
            self.compute_running_couplings()
        
        # Get dimension at LHC energy
        dimension = self.rg.compute_spectral_dimension(lhc_energy)
        
        # Small deviation from 4D at LHC scale
        dim_deviation = dimension - 4.0
        
        # Estimate coupling deviations
        coupling_deviations = {}
        
        # Extract coupling at LHC scale
        scales = self.rg.flow_results['scales']
        idx = np.abs(scales - lhc_energy).argmin()
        
        for coupling_name, trajectory in self.rg.flow_results['coupling_trajectories'].items():
            # Compare to standard power-law running in exactly 4D
            # This is a simplified approximation
            standard_running = trajectory[-1] * (lhc_energy / scales[-1])**(0.1)  # Example: 0.1 is a typical anomalous dimension
            
            # Get actual coupling from dimensional flow
            actual_coupling = trajectory[idx]
            
            # Compute relative deviation
            deviation = (actual_coupling - standard_running) / standard_running
            coupling_deviations[coupling_name] = deviation
        
        # Predict cross-section deviations
        # In reality, these would be computed from detailed QFT calculations
        # using the modified couplings and propagators
        
        # Here's a simplified approximation:
        # Cross-section deviation ~ dimension deviation + coupling deviations
        xsec_deviation = abs(dim_deviation)
        for dev in coupling_deviations.values():
            xsec_deviation += abs(dev)
        
        # Calculate uncertainties for experimental predictions
        # Simplified uncertainty model: statistical + systematic
        statistical_uncertainty = 0.0005  # 0.05% statistical
        systematic_uncertainty = 0.001    # 0.1% systematic
        
        # Combined uncertainty (quadrature sum)
        total_uncertainty = np.sqrt(statistical_uncertainty**2 + systematic_uncertainty**2)
        
        # Check if deviation is within experimental sensitivity
        is_detectable = xsec_deviation > 3 * total_uncertainty
        
        # Calculate required luminosity for 3σ detection
        luminosity_needed = (3 * total_uncertainty / xsec_deviation)**2 * 300  # fb^-1
        
        # Store results
        results = {
            'energy_scale': lhc_energy,
            'dimension': dimension,
            'dimension_deviation': dim_deviation,
            'coupling_deviations': coupling_deviations,
            'cross_section_deviation': xsec_deviation,
            'uncertainty': {
                'statistical': statistical_uncertainty,
                'systematic': systematic_uncertainty,
                'total': total_uncertainty
            },
            'detectability': {
                'is_detectable': is_detectable,
                'significance': xsec_deviation / total_uncertainty,
                'required_luminosity_fb': luminosity_needed
            }
        }
        
        print(f"  Dimension at LHC scale: {dimension:.8f}")
        print(f"  Predicted cross-section deviation: {xsec_deviation:.8f}")
        
        # Store in predictions
        self.predictions['lhc_deviations'] = results
        return results
    
    def predict_high_energy_deviations(self, collider_energy_tev=100, confidence_level=0.95):
        """
        Predict specific deviations from Standard Model in future high-energy accelerator experiments,
        with detailed numerical estimates and error bars.
        
        Parameters:
        -----------
        collider_energy_tev : float
            Collider energy in TeV (e.g., 100 TeV for FCC)
        confidence_level : float
            Confidence level for error bars (e.g., 0.95 for 95% confidence)
            
        Returns:
        --------
        dict
            High-energy deviation predictions with error bars
        """
        print(f"Computing predictions for {collider_energy_tev} TeV collider...")
        
        # Convert TeV to Planck units
        # 1 TeV ~ 10^-16 M_Planck
        collider_energy = collider_energy_tev * 1e-16
        
        # Get dimension at this energy
        dimension = self.rg.compute_spectral_dimension(collider_energy)
        
        # Set energy scale in the unified framework
        self.unified.set_energy_scale(collider_energy)
        
        # 1. Compute cross-section modifications for standard processes
        processes = ['2to2_scalar', '2to2_fermion', 'diphoton', 'dilepton']
        xsec_predictions = {}
        
        for process in processes:
            # Base cross-section in standard model (arbitrary units for illustration)
            sm_xsec = 1.0
            
            # QG modification factor
            # In practice, this would be computed using modified propagators,
            # vertices, etc. from the unified framework
            
            # Simplified approximation based on dimension deviation
            dim_factor = (dimension / 4.0)**2
            
            # Add momentum-dependent modifications from propagators
            test_momentum = collider_energy
            propagator_factor = self.unified.compute_propagator(
                'scalar', test_momentum, include_qg_corrections=True
            ) / self.unified.compute_propagator(
                'scalar', test_momentum, include_qg_corrections=False
            )
            
            # Combined modification factor
            mod_factor = dim_factor * propagator_factor
            
            # Compute modified cross-section
            qg_xsec = sm_xsec * mod_factor
            
            # Compute statistical uncertainty (simplified model)
            # In reality, this would come from a full analysis with systematics
            
            # Uncertainty grows as we approach Planck scale
            rel_uncertainty = 0.01 * (collider_energy * 1e16)**0.5
            
            # For processes sensitive to QG effects, we scale uncertainty
            process_sensitivity = {
                '2to2_scalar': 1.0,
                '2to2_fermion': 1.2,
                'diphoton': 1.5,
                'dilepton': 1.3
            }
            rel_uncertainty *= process_sensitivity.get(process, 1.0)
            
            # Absolute uncertainty
            uncertainty = qg_xsec * rel_uncertainty
            
            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            lower_bound = qg_xsec - z_score * uncertainty
            upper_bound = qg_xsec + z_score * uncertainty
            
            # Store results
            xsec_predictions[process] = {
                'sm_prediction': sm_xsec,
                'qg_prediction': qg_xsec,
                'relative_deviation': (qg_xsec - sm_xsec) / sm_xsec,
                'uncertainty': uncertainty,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': confidence_level
            }
        
        # 2. Compute energy-dependent effects
        energy_range = np.linspace(collider_energy * 0.1, collider_energy, 10)
        energy_dependence = {
            'energies': energy_range,
            'dimensions': [self.rg.compute_spectral_dimension(e) for e in energy_range],
            'cross_sections': {}
        }
        
        # Compute cross-section trends with energy
        for process in processes:
            xsecs = []
            uncertainties = []
            
            for energy in energy_range:
                # Set energy in unified framework
                self.unified.set_energy_scale(energy)
                
                # Get dimension
                dim = self.rg.compute_spectral_dimension(energy)
                
                # Compute cross-section (simplified)
                sm_xsec = 1.0 * (energy / collider_energy)**(-2)  # Standard scaling
                dim_factor = (dim / 4.0)**2
                propagator_factor = self.unified.compute_propagator(
                    'scalar', energy, include_qg_corrections=True
                ) / self.unified.compute_propagator(
                    'scalar', energy, include_qg_corrections=False
                )
                
                mod_factor = dim_factor * propagator_factor
                qg_xsec = sm_xsec * mod_factor
                
                # Uncertainty
                rel_uncertainty = 0.01 * (energy * 1e16)**0.5
                rel_uncertainty *= process_sensitivity.get(process, 1.0)
                uncertainty = qg_xsec * rel_uncertainty
                
                xsecs.append(qg_xsec)
                uncertainties.append(uncertainty)
            
            energy_dependence['cross_sections'][process] = {
                'values': xsecs,
                'uncertainties': uncertainties
            }
        
        # Combine results
        results = {
            'collider_energy_tev': collider_energy_tev,
            'collider_energy_planck': collider_energy,
            'dimension': dimension,
            'dimension_deviation': dimension - 4.0,
            'cross_sections': xsec_predictions,
            'energy_dependence': energy_dependence,
            'detection_significance': {},
            'experimental_sensitivity': {}
        }
        
        # Estimate detection significance
        # S/sqrt(B) where S is signal (deviation) and B is background (SM)
        for process, data in xsec_predictions.items():
            signal = abs(data['qg_prediction'] - data['sm_prediction'])
            background = data['sm_prediction']
            
            # Assume an integrated luminosity (arbitrary units)
            luminosity = 1000  # Higher luminosity → better sensitivity
            
            # Calculate events
            signal_events = signal * luminosity
            background_events = background * luminosity
            
            # Significance
            if background_events > 0:
                significance = signal_events / np.sqrt(background_events)
            else:
                significance = 0
                
            # Experimental sensitivity (how much luminosity needed for 5σ discovery)
            if signal > 0:
                lumi_for_5sigma = 25 * background / (signal**2)
            else:
                lumi_for_5sigma = float('inf')
                
            results['detection_significance'][process] = significance
            results['experimental_sensitivity'][process] = {
                'luminosity_for_5sigma': lumi_for_5sigma,
                'luminosity_units': 'arbitrary'
            }
        
        # Store the prediction
        self.predictions['high_energy_deviations'] = results
        
        # Print summary
        print(f"Predictions for {collider_energy_tev} TeV collider:")
        print(f"  Effective dimension: {dimension:.6f} (deviation: {dimension-4.0:.6e})")
        
        for process, data in xsec_predictions.items():
            rel_dev = data['relative_deviation'] * 100
            print(f"  {process}: deviation = {rel_dev:.4f}% ± {data['uncertainty']/data['sm_prediction']*100:.4f}%")
            print(f"    Significance with L=1000: {results['detection_significance'][process]:.2f}σ")
        
        return results
    
    def predict_cmb_signatures(self, multipole_range=(2, 2500)):
        """
        Predict quantum gravity signatures in the cosmic microwave background power spectrum,
        focusing on deviations at high multipoles and polarization effects.
        
        Parameters:
        -----------
        multipole_range : tuple
            Range of multipoles (l values) to analyze
            
        Returns:
        --------
        dict
            CMB signature predictions
        """
        print("Computing quantum gravity signatures in CMB...")
        
        # Generate multipole range
        multipoles = np.arange(multipole_range[0], multipole_range[1] + 1)
        
        # Standard ΛCDM TT power spectrum (simplified approximation)
        # In practice, this would come from a proper Boltzmann code
        def lambda_cdm_tt(l):
            # Simplified model that captures main features of TT spectrum
            return 5000 * (1 + l/10)**(-1) * np.exp(-l**2 / 1.4e6) + 20
        
        # Standard values
        tt_standard = lambda_cdm_tt(multipoles)
        
        # Energy scale corresponding to each multipole
        # Rough correspondence: l ~ η₀k where η₀ is conformal time today
        # For the early universe, high-l modes exited horizon at higher energies
        # Approximate relation: E ~ 10^16 GeV * (l/3000)
        energies = 1e-3 * (multipoles / 3000)  # In Planck units
        
        # Get dimensions for each energy scale
        dimensions = np.array([self.rg.compute_spectral_dimension(e) for e in energies])
        
        # Compute quantum gravity corrections to CMB spectra
        
        # 1. Primordial power spectrum modification
        # Δ_s^2(k) = Δ_s^2(k)_std * (1 + δ_QG)
        def primordial_correction(dim, energy):
            # Simplified model: correction grows with energy and dimension deviation
            return (dim - 4.0) * energy * 20
        
        # 2. Transfer function modification
        # Small effects from modified dispersion relations
        def transfer_correction(dim, l):
            # Simplified model
            return (dim - 4.0) * np.log(l + 10) * 0.01
        
        # Combine corrections
        primordial_mods = np.array([primordial_correction(dim, e) 
                                 for dim, e in zip(dimensions, energies)])
        transfer_mods = np.array([transfer_correction(dim, l) 
                                for dim, l in zip(dimensions, multipoles)])
        
        # Total modification to TT spectrum
        total_mod = primordial_mods + transfer_mods
        
        # Apply modifications
        tt_modified = tt_standard * (1 + total_mod)
        
        # Compute TE and EE spectra (simplified)
        te_standard = tt_standard * 0.1 * np.sin(multipoles / 100)
        ee_standard = tt_standard * 0.05 * (1 - np.cos(multipoles / 80))
        
        # QG modifications to polarization
        te_modified = te_standard * (1 + total_mod * 1.2)  # Slightly larger effect on TE
        ee_modified = ee_standard * (1 + total_mod * 1.5)  # Larger effect on EE
        
        # Compute BB spectrum - very sensitive to quantum gravity
        # Standard inflationary gravitational waves (r = 0.01)
        bb_standard = tt_standard * 0.01 * np.exp(-multipoles / 80)
        
        # QG modifications potentially much larger for BB
        bb_modified = bb_standard * (1 + total_mod * 3.0)
        
        # Calculate uncertainties (cosmic variance + instrumental)
        # Cosmic variance: ΔC_l/C_l = sqrt(2/(2l+1))
        cosmic_variance = np.sqrt(2 / (2 * multipoles + 1))
        
        # Instrumental noise (decreases with l, simplified model)
        instr_noise_fraction = 0.01 * (1 + (multipoles / 1000)**2)
        
        # Combined uncertainty
        total_uncertainty_fraction = np.sqrt(cosmic_variance**2 + instr_noise_fraction**2)
        
        tt_uncertainty = tt_standard * total_uncertainty_fraction
        te_uncertainty = te_standard * total_uncertainty_fraction * 1.5  # Higher for TE
        ee_uncertainty = ee_standard * total_uncertainty_fraction * 2.0  # Higher for EE
        bb_uncertainty = bb_standard * total_uncertainty_fraction * 5.0  # Much higher for BB
        
        # Calculate significance of deviation
        tt_significance = np.abs(tt_modified - tt_standard) / tt_uncertainty
        te_significance = np.abs(te_modified - te_standard) / te_uncertainty
        ee_significance = np.abs(ee_modified - ee_standard) / ee_uncertainty
        bb_significance = np.abs(bb_modified - bb_standard) / bb_uncertainty
        
        # Assemble results
        results = {
            'multipoles': multipoles,
            'energy_scales': energies,
            'dimensions': dimensions,
            'spectra': {
                'TT': {
                    'standard': tt_standard,
                    'modified': tt_modified,
                    'uncertainty': tt_uncertainty,
                    'significance': tt_significance
                },
                'TE': {
                    'standard': te_standard,
                    'modified': te_modified,
                    'uncertainty': te_uncertainty,
                    'significance': te_significance
                },
                'EE': {
                    'standard': ee_standard,
                    'modified': ee_modified,
                    'uncertainty': ee_uncertainty,
                    'significance': ee_significance
                },
                'BB': {
                    'standard': bb_standard,
                    'modified': bb_modified,
                    'uncertainty': bb_uncertainty,
                    'significance': bb_significance
                }
            },
            'optimal_detection': {}
        }
        
        # Determine optimal multipole ranges for detection
        for spectrum in ['TT', 'TE', 'EE', 'BB']:
            significance = results['spectra'][spectrum]['significance']
            
            # Find regions with significance > 2
            high_signif_regions = []
            current_region = None
            
            for l, s in zip(multipoles, significance):
                if s > 2 and current_region is None:
                    # Start new region
                    current_region = {'start': l, 'end': l, 'max_signif': s}
                elif s > 2 and current_region is not None:
                    # Continue region
                    current_region['end'] = l
                    current_region['max_signif'] = max(current_region['max_signif'], s)
                elif s <= 2 and current_region is not None:
                    # End region
                    high_signif_regions.append(current_region)
                    current_region = None
            
            # Add last region if open
            if current_region is not None:
                high_signif_regions.append(current_region)
            
            results['optimal_detection'][spectrum] = high_signif_regions
        
        # Store predictions
        self.predictions['cmb_signatures'] = results
        
        # Print summary
        print("CMB quantum gravity signatures summary:")
        for spectrum in ['TT', 'TE', 'EE', 'BB']:
            max_signif = np.max(results['spectra'][spectrum]['significance'])
            max_l = multipoles[np.argmax(results['spectra'][spectrum]['significance'])]
            print(f"  {spectrum}: Max significance {max_signif:.2f}σ at l={max_l}")
            
            if results['optimal_detection'][spectrum]:
                best_region = max(results['optimal_detection'][spectrum], 
                                 key=lambda r: r['max_signif'])
                print(f"    Best detection in l={best_region['start']}-{best_region['end']} "
                      f"with up to {best_region['max_signif']:.2f}σ")
            else:
                print("    No statistically significant detection regions")
        
        return results
    
    def predict_astrophysical_signatures(self):
        """
        Predict astrophysical signatures of quantum gravity.
        
        Returns:
        --------
        dict
            Astrophysical predictions
        """
        print("Computing astrophysical signature predictions...")
        
        # Range of energies relevant for astrophysical observations
        # From GeV (cosmic rays) to near-Planckian (early universe)
        energy_scales = np.logspace(-18, -1, 20)  # In Planck units
        
        # Compute RG flow if not already computed
        if not self.rg.flow_results:
            self.compute_running_couplings()
        
        # Get dimensions at each scale
        dimensions = [self.rg.compute_spectral_dimension(e) for e in energy_scales]
        
        # Estimate GZK cutoff modification
        # GZK cutoff occurs when cosmic ray protons interact with CMB photons
        # In standard physics, this is around 5×10^19 eV, or ~4×10^-9 Planck units
        gzk_energy = 4e-9
        gzk_dimension = self.rg.compute_spectral_dimension(gzk_energy)
        
        # Modification from dimensional flow
        # This would involve actual cross-section calculations in variable dimension
        # Simplified approximation: cutoff energy scales with dimension deviation
        dim_factor = (gzk_dimension / 4.0)**2
        modified_gzk = gzk_energy * dim_factor
        
        # Compute Lorentz violation effects
        # Modified dispersion relation: E^2 = p^2(1 + ξ(E/E_p)^n)
        # where ξ is a parameter, E_p is Planck energy, n is power
        # In our model, n is related to dimensional flow
        n_lorentz = abs(self.dim_uv - self.dim_ir)
        xi_lorentz = 1.0 - self.dim_uv / self.dim_ir
        
        # Compute time-of-flight differences for photons of different energies
        # For a GRB at distance L, time delay ~ ξ·L·(E/E_p)^n
        # Use a typical GRB distance of 10^28 cm ~ 10^9 light-years
        grb_distance = 1e9  # in light-years
        
        # Calculate time delays for photons of different energies
        # Reference: 1 GeV photon
        ref_energy = 1e-19  # in Planck units
        
        # Higher energy photons
        test_energies = np.logspace(-18, -12, 7)  # From 10 GeV to 10^6 GeV
        time_delays = []
        
        for energy in test_energies:
            # Time delay ~ distance * (E^n - E_ref^n)
            delay = xi_lorentz * grb_distance * (energy**n_lorentz - ref_energy**n_lorentz)
            # Convert to seconds (approximation)
            delay_seconds = delay * 3.15e7  # light-years to seconds
            time_delays.append(delay_seconds)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': dimensions,
            'gzk_cutoff': {
                'standard_energy': gzk_energy,
                'modified_energy': modified_gzk,
                'dimension': gzk_dimension,
                'modification_factor': dim_factor
            },
            'lorentz_violation': {
                'power_n': n_lorentz,
                'parameter_xi': xi_lorentz,
                'test_energies': test_energies,
                'time_delays': time_delays,
                'grb_distance': grb_distance
            }
        }
        
        print(f"  Modified GZK cutoff: {modified_gzk:.6e} Planck units")
        print(f"  Lorentz violation parameter: n = {n_lorentz:.4f}, ξ = {xi_lorentz:.4f}")
        
        # Store in predictions
        self.predictions['astrophysical_signatures'] = results
        return results
    
    def predict_black_hole_properties(self):
        """
        Predict modified black hole properties from quantum gravity.
        
        Returns:
        --------
        dict
            Black hole predictions
        """
        print("Computing black hole property predictions...")
        
        # Range of black hole masses (in Planck masses)
        masses = np.logspace(0, 10, 20)
        
        # Classical Schwarzschild radius (in Planck length)
        r_classical = 2 * masses
        
        # Compute quantum-corrected radius
        # In dimensional flow, the radius scales differently with mass
        # R ~ M^(1/(d-3)) in d dimensions
        r_quantum = []
        entropies = []
        temperatures = []
        
        for mass in masses:
            # Estimate black hole size
            # (determines the effective energy scale)
            r_sch = 2 * mass
            energy_scale = 1.0 / r_sch
            
            # Get dimension at this energy scale
            dimension = self.rg.compute_spectral_dimension(energy_scale)
            
            # Apply quantum corrections
            if dimension > 3:
                # Modified radius-mass relation
                radius = r_sch * (dimension / 4.0)**(dimension / (dimension - 3))
            else:
                # Near dimension 3, behavior changes dramatically
                # Use a regularized expression
                radius = r_sch * np.exp(1.0 / (dimension - 3 + 1e-10))
            
            r_quantum.append(radius)
            
            # Modified entropy (Bekenstein-Hawking in variable dimension)
            # S ~ Area ~ R^(d-2)
            entropy = np.pi * radius**(dimension - 2)
            entropies.append(entropy)
            
            # Modified temperature
            # T ~ 1/R in all dimensions
            temperature = 1.0 / (4 * np.pi * radius)
            temperatures.append(temperature)
        
        # Store results
        results = {
            'masses': masses,
            'r_classical': r_classical,
            'r_quantum': r_quantum,
            'entropies': entropies,
            'temperatures': temperatures
        }
        
        print(f"  Black hole radius correction at M=10^10 M_p: " 
              f"{r_quantum[-1]/r_classical[-1]:.6f}")
        
        # Store in predictions
        self.predictions['black_holes'] = results
        return results
    
    def predict_primordial_fluctuations(self):
        """
        Predict primordial fluctuation spectra from inflation with quantum gravity.
        
        Returns:
        --------
        dict
            Primordial fluctuation predictions
        """
        print("Computing primordial fluctuation predictions...")
        
        # Range of scales (k-values) for primordial fluctuations
        # From cosmic horizon to small scales
        k_values = np.logspace(-4, 4, 100)
        
        # Standard nearly scale-invariant spectrum
        # P(k) ~ k^(n_s - 1) with n_s ~ 0.96
        ns_standard = 0.96
        p_standard = k_values**(ns_standard - 1)
        
        # Compute quantum gravity corrections
        # Energy scale during inflation ~ 10^-5 Planck
        inflation_energy = 1e-5
        
        # Get dimension at inflation energy
        dimension = self.rg.compute_spectral_dimension(inflation_energy)
        
        # Modified spectral index due to dimensional flow
        # Simplified model: correction ~ (dimension - 4)
        ns_modified = ns_standard + 0.01 * (dimension - 4.0)
        
        # For large k, additional modifications from trans-Planckian effects
        # Use a transition function
        def transition(k, k0, width):
            return 0.5 * (1 + np.tanh((k - k0) / width))
        
        # Transition scale
        k0 = 100  # in relative units
        width = 10
        
        # Compute modified spectrum
        p_modified = p_standard * (1 + 0.1 * (dimension - 4.0) * transition(k_values, k0, width))
        
        # Store results
        results = {
            'k_values': k_values,
            'p_standard': p_standard,
            'p_modified': p_modified,
            'ns_standard': ns_standard,
            'ns_modified': ns_modified,
            'dimension': dimension,
            'inflation_energy': inflation_energy
        }
        
        print(f"  Dimension at inflation scale: {dimension:.6f}")
        print(f"  Modified spectral index: {ns_modified:.6f}")
        
        # Store in predictions
        self.predictions['primordial_fluctuations'] = results
        return results
    
    def predict_gravitational_wave_modifications(self):
        """
        Predict modifications to gravitational wave propagation.
        
        Returns:
        --------
        dict
            Gravitational wave predictions
        """
        print("Computing gravitational wave modification predictions...")
        
        # Standard GW dispersion relation: ω^2 = k^2
        # QG-modified: ω^2 = k^2(1 + A(k/k_p)^α)
        # where k_p is Planck momentum and α depends on dimensional flow
        
        # Range of frequencies (in Hz)
        frequencies = np.logspace(-8, 3, 100)  # From nHz to kHz
        
        # Convert to dimensionless units (ratio to Planck frequency)
        # Planck frequency ~ 10^43 Hz
        f_planck = 1e43
        f_ratios = frequencies / f_planck
        
        # Get spectral dimension at different frequencies
        # Estimate energy scale ~ 2π * frequency
        dimensions = [self.rg.compute_spectral_dimension(2 * np.pi * f) for f in f_ratios]
        
        # Power α from dimensional flow model
        alpha = abs(self.dim_uv - self.dim_ir)
        
        # Amplitude A estimated from dimensional deviation
        amplitude = 0.1 * (1 - self.dim_uv / self.dim_ir)
        
        # Compute phase speed modifications
        # v_phase = ω/k = c * sqrt(1 + A(k/k_p)^α)
        v_modifications = np.sqrt(1 + amplitude * (f_ratios**alpha))
        
        # Compute GW time-of-flight differences
        # For a source at distance L
        # Δt = L/c * A/α * (f_high^α - f_low^α) [for α ≠ 0]
        
        # Use a binary neutron star merger at 100 Mpc
        bns_distance = 100e6  # in parsec
        parsec_to_meters = 3.086e16
        distance_m = bns_distance * parsec_to_meters
        
        # Compare arrival times for different frequencies
        # Reference: 100 Hz
        f_ref = 100  # Hz
        f_ref_ratio = f_ref / f_planck
        
        time_shifts = []
        for f_ratio in f_ratios:
            if abs(alpha) > 1e-10:
                # Time shift formula (in seconds)
                shift = distance_m / 3e8 * amplitude / alpha * (f_ratio**alpha - f_ref_ratio**alpha)
            else:
                # For α ≈ 0, use limit expression
                shift = distance_m / 3e8 * amplitude * np.log(f_ratio / f_ref_ratio)
            
            time_shifts.append(shift)
        
        # Store results
        results = {
            'frequencies': frequencies,
            'f_ratios': f_ratios,
            'dimensions': dimensions,
            'alpha': alpha,
            'amplitude': amplitude,
            'v_modifications': v_modifications,
            'time_shifts': time_shifts,
            'reference_frequency': f_ref,
            'bns_distance': bns_distance
        }
        
        print(f"  Dispersion parameter: α = {alpha:.4f}, A = {amplitude:.4f}")
        print(f"  Maximum time shift: {np.max(np.abs(time_shifts)):.6e} seconds")
        
        # Store in predictions
        self.predictions['gravitational_waves'] = results
        return results
    
    def plot_running_couplings(self, save_path=None):
        """
        Plot running coupling predictions.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'running_couplings' not in self.predictions:
            raise ValueError("No running coupling predictions available. Run compute_running_couplings first.")
            
        results = self.predictions['running_couplings']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Spectral dimension vs energy scale
        axs[0].semilogx(results['energy_scales'], results['dimensions'], 'r-', linewidth=2)
        axs[0].set_xlabel('Energy Scale (Planck units)')
        axs[0].set_ylabel('Spectral Dimension')
        axs[0].set_title('Running Spectral Dimension')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Running couplings
        for name, trajectory in results['coupling_trajectories'].items():
            axs[1].loglog(
                self.rg.flow_results['scales'],
                trajectory,
                '-',
                linewidth=2,
                label=f'{name}'
            )
        
        axs[1].set_xlabel('Energy Scale (Planck units)')
        axs[1].set_ylabel('Coupling Strength')
        axs[1].set_title('Running Couplings')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_black_hole_properties(self, save_path=None):
        """
        Plot black hole property predictions.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'black_holes' not in self.predictions:
            raise ValueError("No black hole predictions available. Run predict_black_hole_properties first.")
            
        results = self.predictions['black_holes']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Radius vs mass
        axs[0, 0].loglog(
            results['masses'],
            results['r_classical'],
            'k--',
            linewidth=2,
            label='Classical'
        )
        axs[0, 0].loglog(
            results['masses'],
            results['r_quantum'],
            'r-',
            linewidth=2,
            label='Quantum corrected'
        )
        axs[0, 0].set_xlabel('Black Hole Mass (Planck masses)')
        axs[0, 0].set_ylabel('Radius (Planck lengths)')
        axs[0, 0].set_title('Black Hole Radius-Mass Relation')
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        axs[0, 0].legend()
        
        # Plot 2: Ratio of quantum to classical radius
        axs[0, 1].semilogx(
            results['masses'],
            np.array(results['r_quantum']) / np.array(results['r_classical']),
            'b-',
            linewidth=2
        )
        axs[0, 1].set_xlabel('Black Hole Mass (Planck masses)')
        axs[0, 1].set_ylabel('Ratio R_quantum / R_classical')
        axs[0, 1].set_title('Quantum Correction Factor')
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Entropy vs mass
        axs[1, 0].loglog(
            results['masses'],
            results['entropies'],
            'g-',
            linewidth=2
        )
        # Classical Bekenstein-Hawking entropy (S ~ M^2)
        S_classical = 4 * np.pi * results['masses']**2
        axs[1, 0].loglog(
            results['masses'],
            S_classical,
            'k--',
            linewidth=2,
            label='Classical S ~ M²'
        )
        axs[1, 0].set_xlabel('Black Hole Mass (Planck masses)')
        axs[1, 0].set_ylabel('Entropy (Planck units)')
        axs[1, 0].set_title('Black Hole Entropy')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        axs[1, 0].legend()
        
        # Plot 4: Temperature vs mass
        axs[1, 1].loglog(
            results['masses'],
            results['temperatures'],
            'r-',
            linewidth=2
        )
        # Classical Hawking temperature (T ~ 1/M)
        T_classical = 1.0 / (8 * np.pi * results['masses'])
        axs[1, 1].loglog(
            results['masses'],
            T_classical,
            'k--',
            linewidth=2,
            label='Classical T ~ 1/M'
        )
        axs[1, 1].set_xlabel('Black Hole Mass (Planck masses)')
        axs[1, 1].set_ylabel('Temperature (Planck units)')
        axs[1, 1].set_title('Black Hole Temperature')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_gravitational_wave_modifications(self, save_path=None):
        """
        Plot gravitational wave modification predictions.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'gravitational_waves' not in self.predictions:
            raise ValueError("No gravitational wave predictions available. Run predict_gravitational_wave_modifications first.")
            
        results = self.predictions['gravitational_waves']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Phase velocity modification vs frequency
        axs[0].semilogx(
            results['frequencies'],
            (results['v_modifications'] - 1) * 3e8,  # Convert to m/s deviation from c
            'r-',
            linewidth=2
        )
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Phase Speed Deviation (m/s)')
        axs[0].set_title('Gravitational Wave Speed vs Frequency')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Time shifts vs frequency
        axs[1].loglog(
            results['frequencies'],
            np.abs(results['time_shifts']),
            'b-',
            linewidth=2
        )
        # Add horizontal line at 1 microsecond (approximate LIGO timing precision)
        axs[1].axhline(1e-6, color='k', linestyle='--', alpha=0.7)
        axs[1].text(
            1e-5,
            1.1e-6,
            'LIGO Timing Precision (~1 μs)',
            fontsize=8,
            verticalalignment='bottom'
        )
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Arrival Time Shift (s)')
        axs[1].set_title(f'Time Shifts for Source at {results["bns_distance"]} Mpc')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # Test the experimental predictions
    
    # Create an experimental predictions instance
    predictions = ExperimentalPredictions(
        dim_uv=2.0,
        dim_ir=4.0,
        transition_scale=1.0
    )
    
    # Compute running couplings
    predictions.compute_running_couplings()
    
    # Predict LHC deviations
    predictions.predict_lhc_deviations()
    
    # Predict astrophysical signatures
    predictions.predict_astrophysical_signatures()
    
    # Predict black hole properties
    predictions.predict_black_hole_properties()
    
    # Predict gravitational wave modifications
    predictions.predict_gravitational_wave_modifications()
    
    # Plot results
    predictions.plot_running_couplings(save_path="running_couplings.png")
    predictions.plot_black_hole_properties(save_path="black_hole_properties.png")
    predictions.plot_gravitational_wave_modifications(save_path="gw_modifications.png")
    
    print("\nExperimental predictions test complete.") 