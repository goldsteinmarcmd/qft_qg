"""
High Energy Particle Collisions with Quantum Gravity Effects

This module simulates high-energy particle collisions where quantum gravity
effects become significant, providing numerical predictions that can be
compared with experimental data from particle accelerators.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import solve_ivp
import pandas as pd

from quantum_gravity_framework.qft_integration import QFTIntegration
from quantum_gravity_framework.unification import TheoryUnification


class HighEnergyCollisionSimulator:
    """
    Simulates high-energy particle collisions with quantum gravity effects.
    """
    
    def __init__(self, dim=4, max_energy=14e3, qg_scale=1e19):
        """
        Initialize the high-energy collision simulator.
        
        Parameters:
        -----------
        dim : int
            Spacetime dimension
        max_energy : float
            Maximum collision energy in GeV (default: 14 TeV for LHC)
        qg_scale : float
            Quantum gravity energy scale in GeV (default: 1e19 GeV ~ Planck scale)
        """
        self.dim = dim
        self.max_energy = max_energy
        self.qg_scale = qg_scale
        
        # Initialize QFT-QG framework
        self.qft_qg = QFTIntegration(dim=dim, cutoff_scale=qg_scale)
        
        # Physical constants in natural units
        self.hbar = 1.0
        self.c = 1.0
        self.G = 1.0 / qg_scale**2
        
        # LHC parameters
        self.lhc_beam_energy = 7e3  # 7 TeV
        self.lhc_luminosity = 1e34  # cm^-2 s^-1
        
        # Collision types
        self.collision_types = {
            'pp': 'proton-proton',
            'ee': 'electron-positron',
            'hh': 'hadron-hadron',
            'AA': 'heavy-ion'
        }
        
        # Store simulation results
        self.results = {}
    
    def _qg_correction_factor(self, energy):
        """
        Calculate quantum gravity correction factor at a given energy.
        
        Parameters:
        -----------
        energy : float or ndarray
            Collision energy in GeV
            
        Returns:
        --------
        float or ndarray
            Correction factor
        """
        # Get QG parameters from our framework
        try:
            feynman = self.qft_qg.derive_modified_feynman_rules()
            alpha = feynman['modification_parameter']
        except Exception:
            # Default value if calculation fails
            alpha = 0.01
            
        # Calculate correction factor: 1 + α (E/E_QG)^2
        return 1.0 + alpha * (energy / self.qg_scale)**2
    
    def _parton_distribution(self, x, Q2, parton_type='gluon'):
        """
        Simplified parton distribution function with QG corrections.
        
        Parameters:
        -----------
        x : float or ndarray
            Bjorken x (momentum fraction)
        Q2 : float
            Energy scale squared in GeV^2
        parton_type : str
            Type of parton ('gluon', 'up', 'down', etc.)
            
        Returns:
        --------
        float or ndarray
            Parton distribution value
        """
        # Simplified PDF parameterizations
        if parton_type == 'gluon':
            base_pdf = 3.0 * (1.0 - x)**5 / x
        elif parton_type in ['up', 'down']:
            base_pdf = x**(-0.5) * (1.0 - x)**3
        else:
            base_pdf = 0.5 * x**(-0.3) * (1.0 - x)**7
            
        # QG correction: more partons at high energy scales
        # Get correction parameters from effective action
        try:
            action = self.qft_qg.quantum_effective_action()
            beta1 = action['correction_parameters']['beta1']
        except Exception:
            # Default value if calculation fails
            beta1 = 0.01
            
        # Apply QG corrections to splitting functions
        qg_effect = 1.0 + beta1 * (Q2 / self.qg_scale**2) * np.log(Q2)
            
        return base_pdf * qg_effect
    
    def cross_section(self, process, energy, apply_qg=True):
        """
        Calculate cross section for a specific process.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        energy : float or ndarray
            Collision energy in GeV
        apply_qg : bool
            Whether to apply quantum gravity corrections
            
        Returns:
        --------
        float or ndarray
            Cross section in pb (picobarns)
        """
        # Baseline cross sections at 14 TeV (approximate values)
        base_xsec_14tev = {
            'higgs': 50.0,        # Higgs production (pb)
            'ttbar': 900.0,       # Top pair production (pb)
            'dijet': 1e8,         # Dijet production (pb)
            'zboson': 2e4,        # Z boson production (pb)
            'wboson': 2e5,        # W boson production (pb)
            'diphoton': 50.0      # Diphoton production (pb)
        }
        
        # Energy scaling (simplified)
        if process == 'higgs':
            # Higgs production scales roughly logarithmically with energy
            energy_scaling = np.log(energy / 1e3) / np.log(14)
        elif process == 'ttbar':
            # Top pair production scales with energy
            energy_scaling = (energy / 14e3)**1.2
        else:
            # Other processes scale roughly with energy
            energy_scaling = (energy / 14e3)
            
        # Baseline cross section
        if process in base_xsec_14tev:
            base_xsec = base_xsec_14tev[process] * energy_scaling
        else:
            # Generic scaling for unknown processes
            base_xsec = 1e3 * energy_scaling
            
        # Apply QG corrections if requested
        if apply_qg:
            # Get QG correction factor from our framework
            try:
                sm_corr = self.qft_qg.quantum_gravity_corrections_to_standard_model()
                if process == 'higgs':
                    correction = sm_corr['process_corrections']['higgs_production']['relative_correction']
                elif process == 'ttbar':
                    correction = sm_corr['process_corrections']['top_pair_production']['relative_correction']
                else:
                    # Default correction factor
                    correction = 1e-30 * (energy / 1e3)**2
                    
                # Apply energy-dependent correction
                # For high pT events, the correction increases
                pt_factor = 3.0  # Enhancement for high pT
                process_factor = 1.0 + pt_factor * correction * (energy / 1e3)**2
            except Exception:
                # Default correction if calculation fails
                process_factor = self._qg_correction_factor(energy)
                
            return base_xsec * process_factor
        else:
            return base_xsec
    
    def simulate_collision(self, process, energy_range=None, collision_type='pp', n_events=10000):
        """
        Simulate high-energy collisions for a specific process.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        energy_range : tuple of (min, max, n_points)
            Energy range for simulation in GeV
        collision_type : str
            Type of collision ('pp', 'ee', 'hh', 'AA')
        n_events : int
            Number of simulated events
            
        Returns:
        --------
        dict
            Simulation results
        """
        # Set default energy range if not provided
        if energy_range is None:
            energy_range = (1e3, self.max_energy, 20)
            
        min_energy, max_energy, n_points = energy_range
        energies = np.linspace(min_energy, max_energy, n_points)
        
        # Calculate cross sections
        standard_xsec = self.cross_section(process, energies, apply_qg=False)
        qg_xsec = self.cross_section(process, energies, apply_qg=True)
        
        # Calculate number of expected events
        # Assuming 100/fb integrated luminosity
        luminosity = 100.0  # fb^-1
        std_events = standard_xsec * luminosity
        qg_events = qg_xsec * luminosity
        
        # Simulate event kinematics
        pt_distribution = []
        eta_distribution = []
        inv_mass_distribution = []
        
        # Generate simplified kinematic distributions for n_events
        for i in range(n_events):
            # Sample collision energy
            E = np.random.uniform(min_energy, max_energy)
            
            # Generate transverse momentum (pT)
            # Modified by QG effects at high pT
            mean_pt = 0.3 * E
            qg_factor = self._qg_correction_factor(E)
            pt = np.random.exponential(mean_pt) * (1.0 + 0.1 * qg_factor * (E/1e3)**2)
            pt_distribution.append(pt)
            
            # Generate pseudorapidity (eta)
            eta = np.random.normal(0, 2.0)
            eta_distribution.append(eta)
            
            # Generate invariant mass
            # QG effects modify the high mass tail
            mean_mass = 0.6 * E
            std_mass = 0.1 * E
            mass = np.random.normal(mean_mass, std_mass) * (1.0 + 0.05 * qg_factor * (E/1e3)**2)
            inv_mass_distribution.append(mass)
            
        # Store and return results
        results = {
            'process': process,
            'collision_type': self.collision_types.get(collision_type, collision_type),
            'energies': energies,
            'standard_xsec': standard_xsec,
            'qg_xsec': qg_xsec,
            'standard_events': std_events,
            'qg_events': qg_events,
            'pt': np.array(pt_distribution),
            'eta': np.array(eta_distribution),
            'mass': np.array(inv_mass_distribution),
            'n_events': n_events
        }
        
        # Store in instance variable
        self.results[process] = results
        
        return results
    
    def significance_estimate(self, process, luminosity=100.0):
        """
        Estimate statistical significance of QG effects.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        luminosity : float
            Integrated luminosity in fb^-1
            
        Returns:
        --------
        dict
            Significance estimates
        """
        if process not in self.results:
            raise ValueError(f"Process {process} not simulated yet. Run simulate_collision first.")
            
        results = self.results[process]
        
        # Calculate expected events
        std_events = results['standard_events'] * (luminosity / 100.0)
        qg_events = results['qg_events'] * (luminosity / 100.0)
        
        # Calculate excess and significance
        excess = qg_events - std_events
        
        # Simple significance calculation: S/√B
        with np.errstate(divide='ignore', invalid='ignore'):
            significance = np.where(std_events > 0, excess / np.sqrt(std_events), 0)
        
        # Calculate minimum luminosity needed for 5σ discovery
        min_lumi = {}
        for i, E in enumerate(results['energies']):
            if significance[i] > 0:
                lumi_needed = 25.0 * luminosity / (significance[i]**2)
                min_lumi[f"{E:.1f} GeV"] = lumi_needed
            
        return {
            'process': process,
            'luminosity': luminosity,
            'excess': excess,
            'significance': significance,
            'max_significance': np.max(significance),
            'discovery_luminosity': min_lumi
        }
    
    def plot_cross_sections(self, process, save_path=None):
        """
        Plot cross sections with and without QG effects.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if process not in self.results:
            raise ValueError(f"Process {process} not simulated yet. Run simulate_collision first.")
            
        results = self.results[process]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot cross sections
        ax1.semilogy(results['energies']/1e3, results['standard_xsec'], 'b-', 
                   linewidth=2, label='Standard QFT')
        ax1.semilogy(results['energies']/1e3, results['qg_xsec'], 'r-', 
                   linewidth=2, label='With QG')
        
        # Plot settings
        ax1.set_xlabel('Collision Energy (TeV)')
        ax1.set_ylabel('Cross Section (pb)')
        ax1.set_title(f'{process.title()} Production Cross Section')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Plot ratio
        ratio = results['qg_xsec'] / results['standard_xsec']
        ax2.plot(results['energies']/1e3, ratio, 'r-', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Plot settings
        ax2.set_xlabel('Collision Energy (TeV)')
        ax2.set_ylabel('QG/Standard Ratio')
        ax2.set_title('Quantum Gravity Effect')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_kinematic_distributions(self, process, save_path=None):
        """
        Plot kinematic distributions from simulated events.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if process not in self.results:
            raise ValueError(f"Process {process} not simulated yet. Run simulate_collision first.")
            
        results = self.results[process]
        
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot pT distribution
        axs[0].hist(results['pt']/1e3, bins=50, alpha=0.7)
        axs[0].set_xlabel('Transverse Momentum (TeV)')
        axs[0].set_ylabel('Events')
        axs[0].set_title('pT Distribution')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        
        # Add QG significance line
        pt_bins = np.linspace(0, np.max(results['pt'])/1e3, 50)
        bin_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # Calculate QG effect vs pT (simplified)
        qg_effect = 1.0 + 0.01 * (bin_centers * 1e3 / self.qg_scale)**2
        
        ax_twin = axs[0].twinx()
        ax_twin.plot(bin_centers, qg_effect, 'r-', linewidth=2)
        ax_twin.set_ylabel('QG Effect', color='r')
        ax_twin.tick_params(axis='y', colors='r')
        
        # Plot eta distribution
        axs[1].hist(results['eta'], bins=50, alpha=0.7)
        axs[1].set_xlabel('Pseudorapidity (η)')
        axs[1].set_ylabel('Events')
        axs[1].set_title('η Distribution')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        # Plot invariant mass distribution
        axs[2].hist(results['mass']/1e3, bins=50, alpha=0.7)
        axs[2].set_xlabel('Invariant Mass (TeV)')
        axs[2].set_ylabel('Events')
        axs[2].set_title('Invariant Mass Distribution')
        axs[2].grid(True, linestyle='--', alpha=0.5)
        
        # Add QG significance line for mass
        mass_bins = np.linspace(0, np.max(results['mass'])/1e3, 50)
        mass_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
        
        # Calculate QG effect vs mass (simplified)
        qg_effect_mass = 1.0 + 0.005 * (mass_centers * 1e3 / self.qg_scale)**2
        
        ax_twin_mass = axs[2].twinx()
        ax_twin_mass.plot(mass_centers, qg_effect_mass, 'r-', linewidth=2)
        ax_twin_mass.set_ylabel('QG Effect', color='r')
        ax_twin_mass.tick_params(axis='y', colors='r')
        
        plt.suptitle(f'Kinematic Distributions for {process.title()} Production', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_significance(self, process, luminosities=[10, 100, 1000, 3000], save_path=None):
        """
        Plot significance of QG effects vs energy for different luminosities.
        
        Parameters:
        -----------
        process : str
            Physics process (e.g., 'higgs', 'ttbar', 'dijet')
        luminosities : list
            List of luminosity values in fb^-1
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if process not in self.results:
            raise ValueError(f"Process {process} not simulated yet. Run simulate_collision first.")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color map for different luminosities
        colors = plt.cm.viridis(np.linspace(0, 1, len(luminosities)))
        
        # Plot significance for each luminosity
        for i, lumi in enumerate(luminosities):
            # Calculate significance
            sig_results = self.significance_estimate(process, luminosity=lumi)
            energies = self.results[process]['energies'] / 1e3  # Convert to TeV
            significance = sig_results['significance']
            
            ax.plot(energies, significance, color=colors[i], 
                   linewidth=2, label=f'{lumi:g} fb⁻¹')
            
        # Add 5σ discovery line
        ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='5σ discovery')
        
        # Plot settings
        ax.set_xlabel('Collision Energy (TeV)')
        ax.set_ylabel('Statistical Significance (σ)')
        ax.set_title(f'Significance of QG Effects in {process.title()} Production')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def discovery_potential_summary(self, processes=['higgs', 'ttbar', 'dijet'], luminosity=3000):
        """
        Summarize discovery potential for different processes.
        
        Parameters:
        -----------
        processes : list
            List of physics processes
        luminosity : float
            Integrated luminosity in fb^-1
            
        Returns:
        --------
        pandas.DataFrame
            Summary table
        """
        summary_data = []
        
        for process in processes:
            # Simulate if not already done
            if process not in self.results:
                self.simulate_collision(process)
                
            # Calculate significance
            sig_results = self.significance_estimate(process, luminosity=luminosity)
            
            # Find energy with maximum significance
            energies = self.results[process]['energies']
            idx = np.argmax(sig_results['significance'])
            best_energy = energies[idx] / 1e3  # TeV
            max_significance = sig_results['significance'][idx]
            
            # Calculate excess percentage
            std_events = self.results[process]['standard_events'][idx] * (luminosity / 100.0)
            qg_events = self.results[process]['qg_events'][idx] * (luminosity / 100.0)
            excess_percent = 100 * (qg_events - std_events) / std_events if std_events > 0 else float('inf')
            
            # Add to summary
            summary_data.append({
                'Process': process.title(),
                'Best Energy (TeV)': best_energy,
                'Significance (σ)': max_significance,
                'Excess (%)': excess_percent,
                'Standard Events': std_events,
                'QG Events': qg_events,
                'Discoverable': max_significance >= 5.0
            })
            
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Sort by significance
        df = df.sort_values('Significance (σ)', ascending=False)
        
        return df


if __name__ == "__main__":
    # Test the high energy collision simulator
    simulator = HighEnergyCollisionSimulator()
    
    # Simulate different processes
    processes = ['higgs', 'ttbar', 'dijet']
    
    for process in processes:
        results = simulator.simulate_collision(process)
        simulator.plot_cross_sections(process, save_path=f'qg_{process}_xsec.png')
        simulator.plot_kinematic_distributions(process, save_path=f'qg_{process}_kinematics.png')
        simulator.plot_significance(process, save_path=f'qg_{process}_significance.png')
        
    # Summarize discovery potential
    summary = simulator.discovery_potential_summary(processes)
    print("\nDiscovery Potential Summary (3000 fb⁻¹):")
    print(summary.to_string(index=False)) 