#!/usr/bin/env python
"""
Black Hole Information Paradox in Categorical Quantum Gravity

This module applies the categorical QG framework to the black hole information
paradox, providing predictions for information preservation and Hawking radiation
modifications based on QFT-QG integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
import scipy.special as special
import networkx as nx
import seaborn as sns

from quantum_gravity_framework.quantum_black_hole import QuantumBlackHole
from quantum_gravity_framework.category_theory import CategoryTheoryGeometry
from quantum_gravity_framework.backreaction import QuantumBackreaction
from quantum_gravity_framework.qft_integration import QFTIntegration

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


class BlackHoleInformationSolver:
    """
    Analysis of black hole information paradox using the categorical QG framework.
    """
    
    def __init__(self, mass_solar=10.0, qg_scale=1e19, spectral_dim_uv=2.0, use_categories=True):
        """
        Initialize the black hole information solver.
        
        Parameters:
        -----------
        mass_solar : float
            Black hole mass in solar masses
        qg_scale : float
            Quantum gravity energy scale in GeV
        spectral_dim_uv : float
            UV spectral dimension
        use_categories : bool
            Whether to use categorical structures for black hole microstates
        """
        self.mass_solar = mass_solar
        self.qg_scale = qg_scale
        self.spectral_dim_uv = spectral_dim_uv
        self.use_categories = use_categories
        
        # Physical constants in natural units (G = c = ħ = 1)
        self.G = 1.0
        self.c = 1.0
        self.hbar = 1.0
        
        # Convert to natural units (GeV scale)
        # 1 solar mass ≈ 1.989e30 kg ≈ 1.116e57 GeV/c²
        self.mass_GeV = mass_solar * 1.116e57
        
        # Planck mass in GeV
        self.m_planck = 1.22e19  # GeV
        
        # Initialize quantum black hole model
        self.qbh = QuantumBlackHole(mass_solar=mass_solar, qg_scale=qg_scale)
        
        # Initialize QG framework components
        self.category_geometry = CategoryTheoryGeometry(dim=4)
        self.backreaction = QuantumBackreaction(qg_scale=qg_scale)
        self.qft_qg = QFTIntegration(dim=4, cutoff_scale=qg_scale)
        
        # Set up black hole parameters
        self.setup_black_hole_parameters()
        
        # Microstate graph representation
        self.microstate_graph = nx.DiGraph()
    
    def setup_black_hole_parameters(self):
        """Calculate standard black hole parameters."""
        # Schwarzschild radius (in natural units)
        self.r_s = 2.0 * self.G * self.mass_GeV
        
        # Hawking temperature (in GeV)
        self.T_H = self.hbar * self.c**3 / (8 * np.pi * self.G * self.mass_GeV * self.qg_scale)
        
        # Black hole entropy (dimensionless)
        self.S_BH = np.pi * self.r_s**2 / (self.G * self.hbar)
        
        # Black hole lifetime (in seconds)
        # τ ≈ 5120 * π * G²M³/ħc⁴
        self.lifetime = 5120 * np.pi * self.G**2 * self.mass_GeV**3 / (self.hbar * self.c**4)
        
        # Apply QG corrections
        self.apply_qg_corrections()
    
    def apply_qg_corrections(self):
        """Apply quantum gravity corrections to black hole parameters."""
        # Use the qbh model to get corrected parameters
        corrected_params = self.qbh.calculate_qg_corrections()
        
        # Extract corrections
        self.entropy_correction = corrected_params.get('entropy_correction', 1.0)
        self.temperature_correction = corrected_params.get('temperature_correction', 1.0)
        self.evaporation_correction = corrected_params.get('evaporation_rate_correction', 1.0)
        
        # Apply the corrections
        self.S_BH_corrected = self.S_BH * self.entropy_correction
        self.T_H_corrected = self.T_H * self.temperature_correction
        
        # Categorical corrections if enabled
        if self.use_categories:
            self.apply_categorical_corrections()
    
    def apply_categorical_corrections(self):
        """Apply categorical structure corrections to black hole."""
        # Get categorical structure
        obj_count = self.category_geometry.count_objects()
        morph_count = self.category_geometry.count_morphisms()
        
        # Generate categorical correction factor
        # Based on the principle that categories provide additional structure
        # to black hole microstates, modifying the entropy counting
        cat_factor = np.log(morph_count / obj_count) / np.log(self.mass_GeV / self.m_planck)
        
        # Apply correction to entropy (additional structure reduces effective entropy)
        cat_entropy_factor = 1.0 - 0.1 * cat_factor
        self.S_BH_corrected *= max(0.5, cat_entropy_factor)
        
        # Generate microstate network based on categorical structure
        self.generate_microstate_network()
    
    def generate_microstate_network(self):
        """Generate a network representation of black hole microstates."""
        # Clear previous graph
        self.microstate_graph.clear()
        
        # Number of microstates to represent (much smaller than actual number)
        n_states = min(100, int(np.exp(np.log(self.S_BH_corrected) / 10)))
        
        # Create nodes for each microstate
        for i in range(n_states):
            # Node attributes based on microstate properties
            energy = self.T_H_corrected * (1.0 + 0.1 * np.random.normal())
            self.microstate_graph.add_node(i, energy=energy, weight=np.exp(-energy/self.T_H_corrected))
        
        # Create edges representing transitions between microstates
        for i in range(n_states):
            # Each microstate connects to several others
            n_connections = np.random.poisson(5)
            targets = np.random.choice(n_states, min(n_connections, n_states), replace=False)
            
            for j in targets:
                if i != j:
                    # Transition probability follows detailed balance
                    prob = min(1.0, np.exp(-(self.microstate_graph.nodes[j]['energy'] - 
                                          self.microstate_graph.nodes[i]['energy'])/self.T_H_corrected))
                    self.microstate_graph.add_edge(i, j, probability=prob)
    
    def calculate_hawking_spectrum(self, omega_range=None, particle_type='scalar'):
        """
        Calculate the Hawking radiation spectrum with QG corrections.
        
        Parameters:
        -----------
        omega_range : array-like, optional
            Range of radiation frequencies (in GeV)
        particle_type : str
            Type of particle ('scalar', 'fermion', 'photon')
            
        Returns:
        --------
        tuple
            (frequencies, standard_spectrum, corrected_spectrum)
        """
        # Default frequency range if not provided
        if omega_range is None:
            # Use a range around the peak of the black body spectrum
            omega_max = 2.82 * self.T_H_corrected  # Peak of black body spectrum
            omega_range = np.linspace(0.1 * omega_max, 5.0 * omega_max, 100)
        
        # Initialize arrays for spectra
        standard_spectrum = np.zeros_like(omega_range, dtype=float)
        corrected_spectrum = np.zeros_like(omega_range, dtype=float)
        
        # Calculate the standard Hawking spectrum
        for i, omega in enumerate(omega_range):
            # Greybody factor (simplified)
            greybody = self.calculate_greybody_factor(omega, particle_type)
            
            # Standard Hawking spectrum: Plank distribution with greybody factor
            if particle_type in ['scalar', 'photon']:
                # Bosonic statistics
                n_omega = 1.0 / (np.exp(omega / self.T_H) - 1.0)
            else:
                # Fermionic statistics
                n_omega = 1.0 / (np.exp(omega / self.T_H) + 1.0)
            
            standard_spectrum[i] = greybody * n_omega * omega**2 / (2.0 * np.pi**2)
            
            # QG corrected spectrum
            # 1. Modified temperature
            # 2. Modified statistics (non-thermal corrections)
            # 3. Modified greybody factor from backreaction
            
            # Get QG corrections
            qg_corrections = self.calculate_qg_spectral_corrections(omega, particle_type)
            
            # Apply all corrections
            corrected_spectrum[i] = standard_spectrum[i] * qg_corrections
        
        return omega_range, standard_spectrum, corrected_spectrum
    
    def calculate_greybody_factor(self, omega, particle_type):
        """
        Calculate greybody factor for Hawking radiation.
        
        This is a simplified model of the transmission coefficient for radiation
        escaping from the black hole.
        
        Parameters:
        -----------
        omega : float
            Radiation frequency in GeV
        particle_type : str
            Type of particle
            
        Returns:
        --------
        float
            Greybody factor
        """
        # Simplified greybody factor model
        # In reality, this involves solving wave equations in Schwarzschild background
        
        # Dimensionless parameter
        x = omega * self.r_s
        
        if particle_type == 'scalar':
            # Scalar field greybody (simplified)
            if x < 1.0:
                return 4.0 * x**2  # Low energy limit
            else:
                return 1.0 - np.exp(-x)  # High energy approximation
        
        elif particle_type == 'fermion':
            # Fermion greybody (simplified)
            if x < 1.0:
                return 2.0 * x**2  # Different low energy behavior
            else:
                return 1.0 - np.exp(-0.8*x)  # High energy approximation
        
        elif particle_type == 'photon':
            # Photon greybody (simplified)
            if x < 1.0:
                return x**2  # Low energy limit suppressed more
            else:
                return 1.0 - np.exp(-1.2*x)  # High energy approximation
        
        else:
            # Default
            return 1.0 - np.exp(-x)
    
    def calculate_qg_spectral_corrections(self, omega, particle_type):
        """
        Calculate QG corrections to Hawking spectrum.
        
        Parameters:
        -----------
        omega : float
            Radiation frequency in GeV
        particle_type : str
            Type of particle
            
        Returns:
        --------
        float
            Total correction factor
        """
        # 1. Temperature correction
        temp_factor = self.temperature_correction
        
        # 2. Non-thermal corrections from QG
        # Higher order terms in the expansion of the radiation formula
        energy_ratio = omega / self.qg_scale
        non_thermal = 1.0 + 0.1 * energy_ratio**2 - 0.05 * energy_ratio**4
        
        # 3. Modified dispersion from QG
        # Get dispersion modification from QFT-QG framework
        try:
            qg_action = self.qft_qg.quantum_effective_action()
            beta1 = qg_action['correction_parameters']['beta1']
            dispersion_mod = 1.0 + beta1 * (omega / self.qg_scale)**2
        except:
            # Default if framework calculation fails
            dispersion_mod = 1.0 + 0.1 * (omega / self.qg_scale)**2
        
        # 4. Microstate effects if using categorical structure
        if self.use_categories:
            # Microstate factor varies with both energy and particle type
            if particle_type == 'scalar':
                microstate_factor = 1.0 + 0.15 * np.sin(10.0 * energy_ratio)
            elif particle_type == 'fermion':
                microstate_factor = 1.0 + 0.12 * np.cos(12.0 * energy_ratio)
            else:
                microstate_factor = 1.0 + 0.08 * np.sin(8.0 * energy_ratio)
        else:
            microstate_factor = 1.0
        
        # Combine all corrections
        total_correction = temp_factor * non_thermal * dispersion_mod * microstate_factor
        
        # Ensure physically reasonable
        return max(0.0, total_correction)
    
    def solve_evaporation_dynamics(self, t_max=None, with_remnant=True):
        """
        Solve black hole evaporation dynamics including QG corrections.
        
        Parameters:
        -----------
        t_max : float, optional
            Maximum integration time
        with_remnant : bool
            Whether to include black hole remnant effects
            
        Returns:
        --------
        dict
            Solution data
        """
        # Default maximum time based on standard lifetime
        if t_max is None:
            t_max = 2.0 * self.lifetime
        
        # Function for black hole mass evolution
        def mass_evolution(t, M):
            """
            dM/dt = -α / M² with QG corrections
            
            α is a constant depending on degrees of freedom
            """
            # Standard evaporation rate
            # Including factor for degrees of freedom (simplified)
            dof = 30.0  # Approximate standard model DOF
            alpha = dof * self.hbar * self.c**4 / (15360 * np.pi * self.G**2)
            
            std_rate = -alpha / M**2
            
            # QG corrections to evaporation rate
            # 1. Modified temperature changes emission rate
            # 2. Near Planck scale, evaporation slows dramatically
            # 3. Final remnant state may exist
            
            # Get mass ratio to Planck mass
            m_ratio = M / self.m_planck
            
            # Temperature correction factor
            temp_corr = self.temperature_correction
            
            # Near-Planck scale suppression
            if with_remnant and m_ratio < this_solver.get_remnant_mass() * 2.0:
                # Approaching remnant mass - evaporation slows dramatically
                remnant_mass = this_solver.get_remnant_mass()
                remnant_factor = (m_ratio - remnant_mass)**2 / (m_ratio**2) if m_ratio > remnant_mass else 0.0
            else:
                # Standard suppression from QG corrections
                remnant_factor = 1.0 - 1.0 / (1.0 + 100.0 * m_ratio**4)
            
            # Apply all corrections
            return std_rate * temp_corr * remnant_factor
        
        # Solver needs reference to self inside the ODE function
        this_solver = self
        
        # Solve the ODE
        t_eval = np.logspace(0, np.log10(t_max), 1000)
        solution = solve_ivp(
            mass_evolution,
            [0, t_max],
            [self.mass_GeV],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract solution
        t = solution.t
        M = solution.y[0]
        
        # Calculate other quantities along the solution
        temperature = np.zeros_like(t)
        entropy = np.zeros_like(t)
        radius = np.zeros_like(t)
        
        for i in range(len(t)):
            # Schwarzschild radius
            radius[i] = 2.0 * self.G * M[i]
            
            # Temperature
            temp = self.hbar * self.c**3 / (8 * np.pi * self.G * M[i] * self.qg_scale)
            temperature[i] = temp * self.temperature_correction
            
            # Entropy
            s_bh = np.pi * radius[i]**2 / (self.G * self.hbar)
            entropy[i] = s_bh * self.entropy_correction
        
        # Store solution
        solution_data = {
            'time': t,
            'mass': M,
            'temperature': temperature,
            'entropy': entropy,
            'radius': radius,
            'final_mass': M[-1],
            'evaporation_time': t[-1] if M[-1] < 0.01 * self.mass_GeV else None
        }
        
        return solution_data
    
    def get_remnant_mass(self):
        """
        Calculate the black hole remnant mass based on QG model.
        
        Returns:
        --------
        float
            Remnant mass in Planck units
        """
        # Default remnant mass from minimal length considerations
        default_remnant = 2.0
        
        try:
            # Try to get remnant mass from QFT-QG framework
            qbh_params = self.qbh.calculate_remnant_parameters()
            return qbh_params.get('remnant_mass_planck_units', default_remnant)
        except:
            # Fall back to default if framework calculation fails
            return default_remnant
    
    def calculate_information_dynamics(self, evaporation_data):
        """
        Calculate information dynamics during black hole evaporation.
        
        Parameters:
        -----------
        evaporation_data : dict
            Data from solve_evaporation_dynamics
            
        Returns:
        --------
        dict
            Information dynamics data
        """
        # Extract evaporation data
        t = evaporation_data['time']
        M = evaporation_data['mass']
        S = evaporation_data['entropy']
        
        # Initialize information measures
        # 1. Cumulative emitted information
        # 2. Radiation entanglement entropy
        # 3. Mutual information between radiation and remaining black hole
        # 4. Page curve
        
        # Number of time steps
        n_steps = len(t)
        
        # Information quantities
        emitted_info = np.zeros(n_steps)
        radiation_entropy = np.zeros(n_steps)
        mutual_info = np.zeros(n_steps)
        
        # Initial black hole entropy
        S_initial = S[0]
        
        # Calculate information dynamics
        for i in range(1, n_steps):
            # Change in black hole entropy
            dS = S[i-1] - S[i]
            
            # Emitted entropy is approximately thermal for standard Hawking radiation
            # With QG corrections, information can escape earlier
            
            # Mass ratio to Planck mass
            m_ratio = M[i] / self.m_planck
            
            # QG correction to information escape
            # Standard scenario: information only in correlations, visible after Page time
            # QG scenario: direct information escape possible due to non-locality
            
            # Information escape parameter (increases as black hole evaporates)
            info_escape = 0.0 if m_ratio > 10.0 else 0.5 * (1.0 - np.exp(-1.0 / m_ratio))
            
            # Emitted information in this step
            # Portion of entropy change is carried as actual information
            emitted_step = dS * info_escape
            
            # Cumulative emitted information
            emitted_info[i] = emitted_info[i-1] + emitted_step
            
            # Radiation entropy follows Page curve
            # Early: mostly thermal (entropy increases)
            # Late: information recovery (entropy decreases)
            
            # Standard Page curve calculation
            page_fraction = (S_initial - S[i]) / S_initial
            
            if page_fraction <= 0.5:
                # Early radiation is thermal
                radiation_entropy[i] = (S_initial - S[i])
            else:
                # After Page time, entropy follows black hole
                radiation_entropy[i] = S[i]
            
            # QG corrections to Page curve
            if self.use_categories:
                # Categorical structure allows information to escape earlier
                # Modified radiation entropy
                qg_factor = min(1.0, 0.8 + info_escape)
                radiation_entropy[i] *= qg_factor
            
            # Mutual information between radiation and black hole
            # I(R:BH) = S(R) + S(BH) - S(R,BH) = S(R) + S(BH) - S_initial
            mutual_info[i] = radiation_entropy[i] + S[i] - S_initial
        
        return {
            'time': t,
            'emitted_information': emitted_info,
            'radiation_entropy': radiation_entropy,
            'mutual_information': mutual_info,
            'initial_entropy': S_initial,
            'page_time': t[np.argmax(radiation_entropy)]
        }
    
    def visualize_information_paradox_resolution(self, evaporation_data, info_dynamics):
        """
        Visualize the resolution of the black hole information paradox.
        
        Parameters:
        -----------
        evaporation_data : dict
            Data from solve_evaporation_dynamics
        info_dynamics : dict
            Data from calculate_information_dynamics
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Black hole mass evolution
        ax = axs[0, 0]
        
        # Normalize time by lifetime
        t_norm = evaporation_data['time'] / self.lifetime
        
        # Normalize mass by initial mass
        M_norm = evaporation_data['mass'] / self.mass_GeV
        
        ax.semilogy(t_norm, M_norm, 'b-', linewidth=2)
        
        # Add marker for Page time
        page_time = info_dynamics['page_time'] / self.lifetime
        page_mass = np.interp(page_time * self.lifetime, 
                             evaporation_data['time'],
                             evaporation_data['mass']) / self.mass_GeV
        
        ax.plot(page_time, page_mass, 'ro', markersize=8, label='Page time')
        
        # Add remnant mass
        if M_norm[-1] > 1e-3:
            ax.axhline(y=M_norm[-1], color='r', linestyle='--', 
                     label=f'Remnant: {M_norm[-1]:.1e} M_initial')
        
        ax.set_xlabel('Time / Lifetime')
        ax.set_ylabel('Mass / Initial Mass')
        ax.set_title('Black Hole Evaporation with QG Corrections')
        ax.grid(True)
        ax.legend()
        
        # Plot 2: Page curve
        ax = axs[0, 1]
        
        # Normalize entropy by initial entropy
        S_norm = info_dynamics['radiation_entropy'] / info_dynamics['initial_entropy']
        
        ax.plot(t_norm, S_norm, 'g-', linewidth=2, label='QG Page Curve')
        
        # Add standard Page curve for comparison
        # Early: S_rad = S_BH_initial - S_BH
        # Late: S_rad = S_BH
        S_BH_norm = evaporation_data['entropy'] / info_dynamics['initial_entropy']
        S_rad_std = np.minimum(1.0 - S_BH_norm, S_BH_norm)
        
        ax.plot(t_norm, S_rad_std, 'k--', linewidth=1.5, label='Standard Page Curve')
        
        # Add Page time marker
        ax.axvline(x=page_time, color='r', linestyle=':', label='Page time')
        
        ax.set_xlabel('Time / Lifetime')
        ax.set_ylabel('S_radiation / S_initial')
        ax.set_title('Black Hole Page Curve: Information Recovery')
        ax.grid(True)
        ax.legend()
        
        # Plot 3: Information escape
        ax = axs[1, 0]
        
        # Normalize information by initial entropy
        I_norm = info_dynamics['emitted_information'] / info_dynamics['initial_entropy']
        
        ax.plot(t_norm, I_norm, 'r-', linewidth=2)
        
        # Add standard scenario (no info until very late)
        std_info = np.zeros_like(t_norm)
        std_transition = 0.9  # Very late phase
        late_phase = t_norm > std_transition
        
        if np.any(late_phase):
            # Information only in late phase
            std_info[late_phase] = (t_norm[late_phase] - std_transition) / (1.0 - std_transition)
            std_info = np.minimum(std_info, 1.0)
            
            ax.plot(t_norm, std_info, 'k--', linewidth=1.5, label='Standard (no QG)')
        
        ax.set_xlabel('Time / Lifetime')
        ax.set_ylabel('Information / Initial Entropy')
        ax.set_title('Information Escape from Black Hole')
        ax.grid(True)
        ax.legend()
        
        # Plot 4: Mutual information
        ax = axs[1, 1]
        
        # Normalize mutual information
        MI_norm = info_dynamics['mutual_information'] / info_dynamics['initial_entropy']
        
        ax.plot(t_norm, MI_norm, 'purple', linewidth=2)
        
        # Add reference lines
        ax.axhline(y=0, color='k', linestyle=':')
        ax.axhline(y=1, color='r', linestyle='--', 
                 label='Complete information recovery')
        
        ax.set_xlabel('Time / Lifetime')
        ax.set_ylabel('I(BH:R) / S_initial')
        ax.set_title('Mutual Information: Radiation and Black Hole')
        ax.grid(True)
        ax.legend()
        
        plt.suptitle('Black Hole Information Paradox Resolution in QG Framework', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return fig
    
    def visualize_microstates(self):
        """
        Visualize black hole microstate structure.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.use_categories or len(self.microstate_graph) == 0:
            self.generate_microstate_network()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a layout for the graph
        pos = nx.spring_layout(self.microstate_graph, seed=42)
        
        # Get node weights for sizing
        weights = [self.microstate_graph.nodes[n]['weight'] * 500 for n in self.microstate_graph.nodes]
        
        # Get edge probabilities for coloring
        edge_probs = [self.microstate_graph[u][v]['probability'] for u, v in self.microstate_graph.edges]
        
        # Node energies for coloring
        energies = [self.microstate_graph.nodes[n]['energy'] for n in self.microstate_graph.nodes]
        energies_norm = [(e - min(energies)) / (max(energies) - min(energies)) for e in energies]
        
        # Draw the graph
        nodes = nx.draw_networkx_nodes(
            self.microstate_graph, pos, ax=ax,
            node_size=weights,
            node_color=energies_norm,
            cmap=plt.cm.viridis,
            alpha=0.8
        )
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(energies), vmax=max(energies)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Microstate Energy')
        
        # Draw edges with colors based on transition probability
        edges = nx.draw_networkx_edges(
            self.microstate_graph, pos, ax=ax,
            width=2,
            edge_color=edge_probs,
            edge_cmap=plt.cm.coolwarm,
            alpha=0.6,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )
        
        # Add labels to a subset of nodes
        # Select a few important nodes based on weight
        top_nodes = sorted(self.microstate_graph.nodes, 
                         key=lambda n: self.microstate_graph.nodes[n]['weight'], 
                         reverse=True)[:10]
        
        node_labels = {n: f"{n}" for n in top_nodes}
        nx.draw_networkx_labels(self.microstate_graph, pos, labels=node_labels, font_size=12)
        
        ax.set_title('Black Hole Microstate Network', fontsize=16)
        ax.set_axis_off()
        
        # Add key information
        plt.figtext(0.02, 0.02, f"Black Hole Mass: {self.mass_solar} M_sun", fontsize=12)
        plt.figtext(0.02, 0.97, f"Entropy: {self.S_BH_corrected:.2e}", fontsize=12, va='top')
        plt.figtext(0.5, 0.02, f"Temperature: {self.T_H_corrected:.2e} GeV", fontsize=12)
        
        return fig


def run_black_hole_analysis():
    """Run a complete black hole information analysis."""
    print("Starting Black Hole Information Analysis with QG Framework")
    
    # Initialize black hole solver
    print("Initializing black hole (10 solar masses)...")
    bh_solver = BlackHoleInformationSolver(mass_solar=10.0, qg_scale=1e19, use_categories=True)
    
    # Calculate Hawking spectrum
    print("Calculating Hawking radiation spectrum with QG corrections...")
    omega, std_spectrum, qg_spectrum = bh_solver.calculate_hawking_spectrum()
    
    # Plot Hawking spectrum
    fig_spectrum, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(omega, std_spectrum, 'b--', label='Standard Hawking')
    ax.plot(omega, qg_spectrum, 'r-', label='QG Corrected')
    
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('Emission rate (d²N/dtdω)')
    ax.set_title('Hawking Radiation Spectrum with QG Corrections')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    fig_spectrum.savefig('black_hole_spectrum_qg.png', dpi=300, bbox_inches='tight')
    
    # Solve evaporation dynamics
    print("Solving black hole evaporation dynamics...")
    evaporation_with_remnant = bh_solver.solve_evaporation_dynamics(with_remnant=True)
    evaporation_no_remnant = bh_solver.solve_evaporation_dynamics(with_remnant=False)
    
    # Calculate information dynamics
    print("Calculating information dynamics during evaporation...")
    info_dynamics = bh_solver.calculate_information_dynamics(evaporation_with_remnant)
    
    # Visualize information paradox resolution
    print("Generating visualization of information paradox resolution...")
    fig_info = bh_solver.visualize_information_paradox_resolution(
        evaporation_with_remnant, info_dynamics
    )
    
    # Save figure
    fig_info.savefig('black_hole_information_qg.png', dpi=300, bbox_inches='tight')
    
    # Visualize black hole microstates
    print("Visualizing black hole microstate structure...")
    fig_microstates = bh_solver.visualize_microstates()
    
    # Save figure
    fig_microstates.savefig('black_hole_microstates_qg.png', dpi=300, bbox_inches='tight')
    
    # Compare remnant vs no-remnant scenarios
    fig_remnant, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize time
    t1_norm = evaporation_with_remnant['time'] / bh_solver.lifetime
    t2_norm = evaporation_no_remnant['time'] / bh_solver.lifetime
    
    # Normalize mass
    m1_norm = evaporation_with_remnant['mass'] / bh_solver.mass_GeV
    m2_norm = evaporation_no_remnant['mass'] / bh_solver.mass_GeV
    
    ax.semilogy(t1_norm, m1_norm, 'r-', linewidth=2, label='With Remnant')
    ax.semilogy(t2_norm, m2_norm, 'b--', linewidth=2, label='Complete Evaporation')
    
    ax.set_xlabel('Time / Lifetime')
    ax.set_ylabel('Mass / Initial Mass')
    ax.set_title('Black Hole Evaporation: Remnant vs Complete Evaporation')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    fig_remnant.savefig('black_hole_remnant_qg.png', dpi=300, bbox_inches='tight')
    
    # Print summary of results
    print("\nSummary of Black Hole Analysis with QG Framework:")
    print("------------------------------------------------")
    print(f"Black Hole Mass: {bh_solver.mass_solar} solar masses")
    print(f"Schwarzschild Radius: {bh_solver.r_s:.2e} natural units")
    print(f"Standard Entropy: {bh_solver.S_BH:.2e}")
    print(f"QG-Corrected Entropy: {bh_solver.S_BH_corrected:.2e}")
    print(f"Standard Hawking Temperature: {bh_solver.T_H:.2e} GeV")
    print(f"QG-Corrected Temperature: {bh_solver.T_H_corrected:.2e} GeV")
    
    if evaporation_with_remnant['final_mass'] > 1e-10:
        print(f"Remnant Mass: {evaporation_with_remnant['final_mass'] / bh_solver.m_planck:.2f} Planck masses")
    
    page_time_fraction = info_dynamics['page_time'] / bh_solver.lifetime
    print(f"Page Time: {page_time_fraction:.2f} of lifetime")
    
    # Information recovery
    final_info = info_dynamics['emitted_information'][-1] / info_dynamics['initial_entropy']
    print(f"Information Recovery: {final_info:.1%}")
    
    print("\nFigures saved:")
    print("- black_hole_spectrum_qg.png")
    print("- black_hole_information_qg.png")
    print("- black_hole_microstates_qg.png")
    print("- black_hole_remnant_qg.png")


if __name__ == "__main__":
    run_black_hole_analysis() 