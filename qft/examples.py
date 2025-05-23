"""Examples demonstrating the QFT library usage.

This module provides examples of how to use the various QFT calculation
modules together.
"""

import numpy as np
import matplotlib.pyplot as plt

from qft.path_integral import (
    generate_field_configurations,
    calculate_path_integral,
    action_free_scalar
)
from qft.feynman import (
    Particle,
    Vertex,
    Edge,
    FeynmanDiagram,
    electron_positron_scattering
)
from qft.scattering import (
    mandelstam_variables,
    compton_scattering_amplitude,
    cross_section
)
from qft.renormalization import (
    one_loop_correction,
    running_coupling,
    plot_running_coupling
)
from qft.gauge_theory import (
    U1GaugeField,
    SUNGaugeField,
    initialize_lattice_gauge_field,
    lattice_action
)
from qft.effective_field_theory import (
    Field,
    Operator,
    EffectiveTheory,
    create_sm_eft,
    create_heavy_quark_eft,
    plot_operator_scaling
)
from qft.non_perturbative import (
    SchwingerDysonSolver,
    instanton_profile_kink,
    instanton_action,
    borel_resum,
    stochastic_quantization,
    FunctionalRG
)
from qft.quantum_gravity import (
    CausalTriangulation,
    EmergentSpacetime,
    SpinNetwork
)
from qft.lattice_field_theory import (
    LatticeScalarField,
    HybridMonteCarlo,
    critical_exponents_fss,
    lattice_to_continuum_extrapolation
)
from qft.tensor_networks import (
    MatrixProductState,
    TensorRenormalizationGroup,
    ising_initial_tensor
)

def example_path_integral():
    """Demonstrate a path integral calculation."""
    print("\n===== Path Integral Example =====")
    
    # Generate field configurations
    grid_size = 100
    num_configs = 500
    print(f"Generating {num_configs} field configurations on a grid of size {grid_size}...")
    configs = generate_field_configurations(grid_size, num_configs, sigma=1.0)
    
    # Define an action for a free scalar field with mass = 1
    def example_action(phi):
        # Approximate derivative using finite differences
        d_phi = np.gradient(phi)
        return action_free_scalar(phi, mass=1.0, d_phi=d_phi)
    
    # Calculate path integral
    print("Calculating path integral...")
    result = calculate_path_integral(example_action, configs)
    print(f"Path integral result: {result}")

def example_feynman_diagram():
    """Demonstrate Feynman diagram creation and amplitude calculation."""
    print("\n===== Feynman Diagram Example =====")
    
    print("Creating e⁻e⁺ scattering Feynman diagram...")
    diagram = electron_positron_scattering()
    
    # Calculate amplitude
    amplitude = diagram.calculate_amplitude()
    print(f"Scattering amplitude: {amplitude}")
    
    # Visualize diagram
    print("Plotting Feynman diagram (will be displayed if run in interactive mode)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    diagram.visualize(ax=ax, show=False)
    plt.title("Electron-Positron Scattering")
    # plt.show()  # Uncomment to display if running interactively

def example_scattering_amplitude():
    """Demonstrate scattering amplitude calculation."""
    print("\n===== Scattering Amplitude Example =====")
    
    # Compton scattering
    print("Calculating Compton scattering amplitude...")
    
    # Define momenta in lab frame (MeV)
    electron_mass = 0.511  # MeV
    
    # Incoming electron at rest
    p_e_in = np.array([electron_mass, 0, 0, 0])
    
    # Incoming photon with energy 10 MeV in z-direction
    photon_energy = 10.0  # MeV
    p_gamma_in = np.array([photon_energy, 0, 0, photon_energy])
    
    # Scattering angle (30 degrees)
    angle = np.pi / 6
    
    # Calculate outgoing momenta (simplified)
    # In a real calculation, we would use conservation laws and kinematics
    p_e_out = np.array([5.0, 1.0, 0, 3.0])
    p_gamma_out = p_e_in + p_gamma_in - p_e_out
    
    # Calculate amplitude squared
    amplitude_squared = compton_scattering_amplitude(
        p_e_in, p_gamma_in, p_e_out, p_gamma_out
    )
    
    print(f"Compton scattering amplitude squared: {amplitude_squared} MeV⁻²")
    
    # Calculate Mandelstam variables
    s, t, u = mandelstam_variables(p_e_in, p_gamma_in, p_e_out, p_gamma_out)
    print(f"Mandelstam s: {s} MeV²")
    print(f"Mandelstam t: {t} MeV²")
    print(f"Mandelstam u: {u} MeV²")

def example_renormalization():
    """Demonstrate renormalization calculations."""
    print("\n===== Renormalization Example =====")
    
    # Example parameters
    coupling = 0.1  # Coupling constant at 1 GeV
    mu = 1.0  # Reference scale (1 GeV)
    
    # Calculate running coupling at different energies
    energies = [0.1, 1.0, 10.0, 100.0]  # GeV
    print("Running coupling values:")
    for energy in energies:
        running_coup = running_coupling(coupling, mu, energy)
        print(f"  Coupling at {energy} GeV: {running_coup}")
    
    # Calculate one-loop correction
    print("\nCalculating one-loop correction...")
    external_momentum = np.array([10.0, 0, 0, 0])  # 10 GeV
    mass = 1.0  # 1 GeV
    cutoff = 1000.0  # 1 TeV cutoff
    
    div_part, fin_part = one_loop_correction(external_momentum, mass, coupling, cutoff=cutoff)
    print(f"One-loop correction (cutoff reg.):")
    print(f"  Divergent part = {div_part}")
    print(f"  Finite part = {fin_part}")

def example_gauge_theory():
    """Demonstrate gauge theory calculations."""
    print("\n===== Gauge Theory Example =====")
    
    # Example 1: U(1) gauge field (QED)
    print("QED Field Strength Calculation:")
    photon = U1GaugeField(name="Photon", coupling=1/137.036)
    
    # Create a simple electromagnetic field
    A_mu = np.array([0.0, 0.1, 0.0, 0.0])  # Only A_x is non-zero
    
    # Create derivatives (simple constant E field in z direction)
    partial_mu_A_nu = np.zeros((4, 4))
    partial_mu_A_nu[0, 2] = 0.01  # ∂A_z/∂t = E_z
    
    # Calculate field strength
    F_munu = photon.field_strength(A_mu, partial_mu_A_nu)
    
    # Display components (E and B fields)
    print("Electric field:")
    for i in range(3):
        print(f"  E_{i+1} = F_{0}{i+1} = {F_munu[0, i+1]}")
    
    print("Magnetic field:")
    print(f"  B_x = F_{2}{3} = {F_munu[2, 3]}")
    print(f"  B_y = F_{3}{1} = {F_munu[3, 1]}")
    print(f"  B_z = F_{1}{2} = {F_munu[1, 2]}")
    
    # Example 2: SU(3) gauge field (QCD)
    print("\nQCD Field Strength Calculation:")
    gluon = SUNGaugeField(name="Gluon", group="SU(3)", coupling=0.1)
    
    # Calculate action for a random field configuration
    A_mu = np.random.rand(4, 8) * 0.1  # Small random gluon field
    partial_mu_A_nu = np.random.rand(4, 4, 8) * 0.01
    
    F_munu = gluon.field_strength(A_mu, partial_mu_A_nu)
    action_value = gluon.action(F_munu)
    
    print(f"SU(3) Yang-Mills action: {action_value}")
    
    # Example 3: Lattice gauge theory
    print("\nLattice Gauge Theory Example:")
    lattice_size = (4, 4, 4, 4)
    gauge_links = initialize_lattice_gauge_field(lattice_size, gauge_group="U(1)")
    
    # Calculate Wilson action
    beta = 2.0  # β = 2N/g²
    action_value = lattice_action(gauge_links, lattice_size, beta, gauge_group="U(1)")
    
    print(f"U(1) lattice action with β={beta}: {action_value}")
    print(f"Average plaquette: {action_value / (6 * 4**4)}")

def example_effective_field_theory():
    """Demonstrate effective field theory calculations."""
    print("\n===== Effective Field Theory Example =====")
    
    # Create the Standard Model EFT
    print("Standard Model Effective Field Theory (SMEFT):")
    smeft = create_sm_eft()
    print(smeft.lagrangian(max_dimension=6))
    
    # Power counting example
    print("\nPower counting for dimension-6 operators:")
    dim6_ops = ["O_h6", "O_le", "O_4q"]
    
    for op_name in dim6_ops:
        dim = smeft.operators[op_name].dimension()
        print(f"\nOperator {op_name} has dimension {dim}")
        
        for energy in [10, 100, 500, 1000]:  # GeV
            contribution = smeft.power_counting(op_name, energy)
            print(f"  Contribution at {energy} GeV: {contribution:.6g}")
            
            # Show how suppressed this is compared to SM
            suppression = contribution / smeft.power_counting("O_h", energy)
            print(f"  Suppression vs. SM: {suppression:.6g}")
    
    # Heavy Quark EFT example
    print("\nHeavy Quark Effective Theory (HQET):")
    hqet = create_heavy_quark_eft()
    print(hqet.lagrangian(max_dimension=5))
    
    # Display scaling plot info
    print("\nOperator scaling with energy:")
    energy_range = np.logspace(1, 3, 100)  # 10 GeV to 1 TeV
    print("Calculated scaling for dimension-4 and dimension-6 operators")
    print("The plot would show dimension-6 operators scaling as (E/Λ)²")

def example_non_perturbative():
    """Demonstrate non-perturbative QFT calculations."""
    print("\n===== Non-Perturbative QFT Example =====")
    
    # Example 1: Schwinger-Dyson equations
    print("Schwinger-Dyson Equation Solver:")
    sd_solver = SchwingerDysonSolver()
    
    # Set up a range of momenta
    momenta = [np.array([p, 0, 0, 0]) for p in np.linspace(0.1, 10, 10)]
    
    # Solve for the propagator
    propagator = sd_solver.solve_scalar_propagator(mass=1.0, coupling=0.1, momenta=momenta)
    
    # Compare with free propagator
    p_squared = np.array([np.sum(p**2) for p in momenta])
    free_propagator = 1.0 / (p_squared + 1.0)
    
    print(f"Free vs. interacting propagator at p²=1:")
    for i, p2 in enumerate(p_squared):
        if 0.9 < p2 < 1.1:
            print(f"  Free: {free_propagator[i]:.6f}, Interacting: {propagator[i]:.6f}")
            print(f"  Self-energy correction: {1/propagator[i] - 1/free_propagator[i]:.6f}")
            break
    
    # Example 2: Instantons
    print("\nInstanton Calculation:")
    
    # Generate a kink instanton profile
    x_values = np.linspace(-5, 5, 100)
    instanton = np.array([instanton_profile_kink(x, mass=1.0) for x in x_values])
    
    # Define simplified operators for the instanton action
    def kinetic(field):
        # Approximate ∫dx (dφ/dx)² using finite differences
        gradient_squared = np.sum(np.diff(field)**2) / np.diff(x_values)[0]
        return gradient_squared
    
    def phi4_potential(field):
        # V(φ) = λ/4 (φ² - v²)²
        v = 1.0 / np.sqrt(2)
        return 0.25 * np.sum((field**2 - v**2)**2) * (x_values[1] - x_values[0])
    
    # Calculate instanton action
    action = instanton_action(instanton, phi4_potential, kinetic)
    print(f"Instanton action: {action:.6f}")
    print(f"Instanton density (g=0.1): {np.exp(-action/0.1**2):.6g}")
    
    # Example 3: Resummation of divergent series
    print("\nBorel Resummation:")
    
    # Example of a divergent perturbation series
    # These coefficients grow factorially, typical of QFT perturbation series
    coefficients = [1, 1, 2, 6, 24, 120, 720]
    
    # Compute resummed value at different coupling strengths
    couplings = [0.1, 0.2, 0.3, 0.4]
    for g in couplings:
        resummed = borel_resum(coefficients, g)
        # Compare with naive sum of the first few terms
        partial_sum = sum(coefficients[i] * g**(i+1) for i in range(min(4, len(coefficients))))
        
        print(f"  Coupling g={g:.1f}:")
        print(f"    Partial sum (4 terms): {partial_sum:.6f}")
        print(f"    Borel resummed: {resummed:.6f}")
    
    # Example 4: Functional Renormalization Group
    print("\nFunctional Renormalization Group:")
    
    # Define a simple φ⁴ potential
    def initial_potential(phi):
        return 0.5 * phi**2 + 0.1 * phi**4 / 24
    
    # Create and solve the FRG flow
    frg = FunctionalRG(initial_potential, grid_points=20, k_max=10.0, k_min=0.1)
    potentials, scales = frg.solve_flow(k_steps=10)
    
    # Show potential at UV and IR scales
    phi_index = 10  # Middle of the grid (φ=0)
    print(f"  Potential at φ=0:")
    print(f"    UV scale (k={scales[0]:.1f}): U={potentials[scales[0]][phi_index]:.6f}")
    print(f"    IR scale (k={scales[-1]:.1f}): U={potentials[scales[-1]][phi_index]:.6f}")
    
    # Compare effective mass (curvature at minimum)
    # This is proportional to d²U/dφ² at φ=0
    d2U_UV = (potentials[scales[0]][phi_index+1] - 2*potentials[scales[0]][phi_index] + 
              potentials[scales[0]][phi_index-1]) / frg.d_phi**2
    d2U_IR = (potentials[scales[-1]][phi_index+1] - 2*potentials[scales[-1]][phi_index] + 
              potentials[scales[-1]][phi_index-1]) / frg.d_phi**2
              
    print(f"  Effective mass squared (d²U/dφ²):")
    print(f"    UV: {d2U_UV:.6f}")
    print(f"    IR: {d2U_IR:.6f}")

def example_quantum_gravity():
    """Demonstrate quantum gravity approaches."""
    print("\n===== Quantum Gravity Example =====")
    
    # Example 1: Causal Dynamical Triangulations
    print("Causal Dynamical Triangulations:")
    
    # Create a small causal triangulation
    cdt = CausalTriangulation(time_slices=3, vertices_per_slice=5)
    graph = cdt.generate_triangulation()
    
    # Compute the action
    action = cdt.compute_action(newton_g=1.0, lambda_cc=0.1)
    
    print(f"  CDT triangulation:")
    print(f"    Vertices: {graph.number_of_nodes()}")
    print(f"    Edges: {graph.number_of_edges()}")
    print(f"    Regge action: {action:.6f}")
    
    # Perform a few Monte Carlo updates
    print("\n  Monte Carlo evolution:")
    accepted = 0
    steps = 10
    for i in range(steps):
        accepted += int(cdt.metropolis_update(beta=10.0))
        if i % 5 == 4:
            action = cdt.compute_action()
            print(f"    Step {i+1}, Action: {action:.6f}, Acceptance rate: {accepted/(i+1):.2f}")
    
    # Example 2: Emergent Spacetime
    print("\nEmergent Spacetime:")
    
    # Create a small emergent spacetime model
    es = EmergentSpacetime(n_nodes=8, dim=2, connectivity=0.3)
    es.generate_random_state()
    
    # Embed in 2D space
    pos = es.embed_in_space(iterations=50)
    
    # Compute entanglement entropies for different regions
    entropies = []
    
    for size in range(1, es.n_nodes):
        region = set(range(size))
        entropy = es.compute_entanglement_entropy(region)
        entropies.append(entropy)
    
    # Check if entanglement entropy follows area law
    print(f"  Entanglement graph with {es.n_nodes} nodes and {es.entanglement_graph.number_of_edges()} edges")
    print(f"  Entanglement entropies for regions of increasing size:")
    for i, entropy in enumerate(entropies):
        print(f"    Region size {i+1}: entropy = {entropy:.4f}")
    
    # Example 3: Spin Networks (Loop Quantum Gravity)
    print("\nLoop Quantum Gravity (Spin Networks):")
    
    # Create a spin network
    sn = SpinNetwork()
    sn.create_simple_network(n_nodes=6)
    
    # Calculate geometric quantities
    areas = []
    volumes = []
    
    for i in range(6):
        # Calculate area of the edge connecting to next node
        j = (i + 1) % 6
        area = sn.area_operator((i, j))
        areas.append(area)
        
        # Calculate volume of node
        volume = sn.volume_operator(i)
        volumes.append(volume)
    
    print(f"  Spin network with {sn.graph.number_of_nodes()} nodes and {sn.graph.number_of_edges()} edges")
    print(f"  Edge spin labels: {[sn.spin_labels.get((i, (i+1)%6)) for i in range(6)]}")
    print(f"  Areas: {[f'{a:.2f}' for a in areas]}")
    print(f"  Volumes: {[f'{v:.2f}' for v in volumes]}")
    print(f"  Total area: {sum(areas):.4f}")
    print(f"  Total volume: {sum(volumes):.4f}")

def example_lattice_field_theory():
    """Demonstrate lattice field theory calculations."""
    print("\n===== Lattice Field Theory Example =====")
    
    # Example 1: Phi^4 theory in 2D
    print("Example 1: Monte Carlo simulation of Phi^4 theory")
    
    # Create a small 2D lattice
    lattice_size = (16, 16)
    mass_squared = -0.2  # Negative for symmetry breaking
    coupling = 0.5
    
    # Create simulation object
    sim = LatticeScalarField(
        lattice_size,
        mass_squared=mass_squared,
        coupling=coupling,
        dimension=2
    )
    
    # Run a short simulation
    print("Running simulation with Metropolis algorithm...")
    results = sim.run_simulation(
        num_thermalization=500,   # In a real study, use 5000+
        num_configurations=200,   # In a real study, use 10000+
        measurements_interval=5,
        beta=1.0
    )
    
    # Print results
    print(f"Action: {results['action_mean']:.4f} ± {results['action_error']:.4f}")
    print(f"Field expectation value: {results['field_mean']:.4f}")
    print(f"Field magnitude: {results['field_abs_mean']:.4f}")
    print(f"Susceptibility: {results['susceptibility']:.4f}")
    print(f"Binder cumulant: {results['binder_cumulant']:.4f}")
    
    # Calculate correlation function
    print("\nCalculating correlation function...")
    distances, correlations = sim.calculate_correlation_function(
        num_samples=50,
        thermalization=100
    )
    
    # Extract mass gap
    mass, mass_error = sim.calculate_mass_gap((distances, correlations))
    print(f"Mass gap: {mass:.4f} ± {mass_error:.4f}")
    
    # Example 2: Hybrid Monte Carlo
    print("\nExample 2: Hybrid Monte Carlo algorithm")
    
    # Define action and gradient for phi^4 theory
    def phi4_action(field):
        # Kinetic term
        kinetic = 0
        for i in range(len(field.shape)):
            shifted = np.roll(field, -1, axis=i)
            kinetic += np.sum((shifted - field)**2)
        
        kinetic *= 0.5
        
        # Mass and interaction terms
        mass_term = mass_squared * np.sum(field**2) * 0.5
        interaction = coupling * np.sum(field**4) / 24.0
        
        return kinetic + mass_term + interaction
    
    def phi4_gradient(field):
        # Gradient of the action with respect to the field
        grad = np.zeros_like(field)
        
        # Gradient of kinetic term
        for i in range(len(field.shape)):
            forward = np.roll(field, -1, axis=i)
            backward = np.roll(field, 1, axis=i)
            grad += -1 * (forward + backward - 2 * field)
        
        # Gradient of mass term
        grad += mass_squared * field
        
        # Gradient of interaction term
        grad += coupling * field**3 / 6.0
        
        return grad
    
    # Create HMC simulator
    hmc = HybridMonteCarlo(
        action_func=phi4_action,
        grad_func=phi4_gradient,
        field_shape=(12, 12),
        step_size=0.1,
        n_steps=10
    )
    
    # Run short HMC simulation
    print("Running HMC simulation...")
    
    # Function to calculate magnetization
    def measure_magnetization(field):
        return np.abs(np.mean(field))
    
    measurements = hmc.run_simulation(
        n_thermalization=50,
        n_samples=100,
        measurement_func=measure_magnetization
    )
    
    # Calculate results
    mean_m = np.mean(measurements)
    std_m = np.std(measurements)
    error_m = std_m / np.sqrt(len(measurements))
    
    print(f"Magnetization: {mean_m:.4f} ± {error_m:.4f}")
    
    # Example 3: Continuum limit extrapolation
    print("\nExample 3: Continuum limit extrapolation")
    
    # Simulated data: lattice spacings and corresponding observable values
    # In a real study, these would come from actual simulations
    lattice_spacings = [0.2, 0.1, 0.05, 0.025]
    mass_values = [0.987, 0.956, 0.942, 0.936]
    mass_errors = [0.010, 0.008, 0.007, 0.006]
    
    # Extrapolate to continuum
    continuum_mass, continuum_error, fit_info = lattice_to_continuum_extrapolation(
        lattice_spacings, mass_values, mass_errors
    )
    
    print(f"Lattice mass values at different spacings:")
    for i, a in enumerate(lattice_spacings):
        print(f"  a = {a:.3f}: mass = {mass_values[i]:.4f} ± {mass_errors[i]:.4f}")
    
    print(f"Extrapolated continuum mass: {continuum_mass:.4f} ± {continuum_error:.4f}")
    print(f"Fit type: {fit_info['type']}")
    
    if fit_info['type'] != 'failed':
        print(f"Chi-squared per DoF: {fit_info['chi2_dof']:.4f}")

def example_tensor_networks():
    """Demonstrate tensor network methods for QFT."""
    print("\n===== Tensor Networks Example =====")
    
    # Example 1: Matrix Product States for a spin chain
    print("Example 1: Matrix Product States")
    
    # Create a simple 8-site spin-1/2 chain
    physical_dims = [2] * 8
    bond_dim = 8
    mps = MatrixProductState(physical_dims, bond_dim=bond_dim)
    
    # Define Pauli matrices as operators
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Identity operator
    identity = np.eye(2)
    
    # Calculate single-site expectation values
    mag_z = []
    mag_x = []
    
    for i in range(len(physical_dims)):
        mag_z.append(mps.expectation_value(sigma_z, i).real)
        mag_x.append(mps.expectation_value(sigma_x, i).real)
    
    # Calculate nearest-neighbor correlation
    corr_zz = []
    for i in range(len(physical_dims) - 1):
        # Apply sigma_z to sites i and i+1
        mps_z_i = mps.apply_operator(sigma_z, i)
        mps_zz = mps_z_i.apply_operator(sigma_z, i+1)
        
        # Calculate expectation value
        corr_zz.append(mps.overlap(mps_zz).real)
    
    print(f"Average Z magnetization: {np.mean(mag_z):.6f}")
    print(f"Average X magnetization: {np.mean(mag_x):.6f}")
    print(f"Average ZZ correlation: {np.mean(corr_zz):.6f}")
    
    # Example 2: Tensor Renormalization Group for 2D Ising model
    print("\nExample 2: Tensor Renormalization Group")
    
    # Critical temperature for 2D Ising model
    T_c = 2.0 / np.log(1 + np.sqrt(2))
    
    # Calculate partition function at different temperatures
    temperatures = [0.8 * T_c, 0.9 * T_c, T_c, 1.1 * T_c, 1.2 * T_c]
    log_Z_per_site = []
    
    print("Computing Ising model partition function...")
    for T in temperatures:
        beta = 1.0 / T
        initial_tensor = ising_initial_tensor(beta)
        
        # Create TRG with truncation bond dimension 12
        trg = TensorRenormalizationGroup(initial_tensor, bond_dim=12)
        
        # Approximate partition function with 3 iterations (8x8 lattice)
        Z = trg.compute_partition_function(3)
        log_Z_per_site.append(np.log(abs(Z)) / 64)  # 8x8 = 64 sites
    
    # Print results
    print("Ising model free energy density (log(Z)/N):")
    for i, T in enumerate(temperatures):
        print(f"  T/T_c = {T/T_c:.2f}: {log_Z_per_site[i]:.6f}")
    
    # Look for maximum heat capacity (d²(logZ)/dβ²) near T_c
    if len(log_Z_per_site) >= 3:
        # Numerical second derivative
        betas = [1.0 / T for T in temperatures]
        d2_logZ = []
        
        for i in range(1, len(betas) - 1):
            # Simple finite difference for second derivative
            delta1 = (log_Z_per_site[i] - log_Z_per_site[i-1]) / (betas[i] - betas[i-1])
            delta2 = (log_Z_per_site[i+1] - log_Z_per_site[i]) / (betas[i+1] - betas[i])
            d2_logZ.append((delta2 - delta1) / ((betas[i+1] - betas[i-1]) / 2))
        
        # Find maximum (approximately the critical point)
        max_idx = np.argmax(np.abs(d2_logZ))
        T_approx = temperatures[max_idx + 1]
        
        print(f"\nEstimated critical temperature: T ≈ {T_approx:.4f}")
        print(f"Exact critical temperature: T_c = {T_c:.4f}")
        print(f"Relative error: {abs(T_approx - T_c)/T_c*100:.2f}%")

def run_all_examples():
    """Run all QFT examples."""
    print("====================================")
    print("Quantum Field Theory Examples")
    print("====================================")
    
    example_path_integral()
    example_feynman_diagram()
    example_scattering_amplitude()
    example_renormalization()
    example_gauge_theory()
    example_effective_field_theory()
    example_non_perturbative()
    example_quantum_gravity()
    example_lattice_field_theory()
    example_tensor_networks()
    
    print("\n====================================")
    print("All examples completed")
    print("====================================")

if __name__ == "__main__":
    run_all_examples() 