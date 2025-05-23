"""
Fermion Treatment with Dimensional Flow

This module implements proper fermion treatment with dimensional flow effects,
allowing for consistent handling of fermionic fields across energy scales
where quantum gravity effects become important.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, linalg as splinalg

from quantum_gravity_framework.dimensional_flow_rg import DimensionalFlowRG
from quantum_gravity_framework.unified_approach import UnifiedQGApproach
from quantum_gravity_framework.gauge_field_integration import GaugeFieldIntegration


class DiracOperator:
    """
    Implements dimension-dependent Dirac operator for fermions.
    """
    
    def __init__(self, dimension=4, lattice_size=10, mass=0.1):
        """
        Initialize the Dirac operator.
        
        Parameters:
        -----------
        dimension : float
            Spacetime dimension (can be non-integer for dimensional flow)
        lattice_size : int
            Size of the lattice
        mass : float
            Fermion mass
        """
        self.dimension = dimension
        self.lattice_size = lattice_size
        self.mass = mass
        
        # Integer part of dimension
        self.dim_int = int(np.round(dimension))
        
        # Initialize gamma matrices
        self.gamma_matrices = self._initialize_gamma_matrices(self.dim_int)
        
        # Total number of lattice sites
        self.total_sites = lattice_size ** self.dim_int
        
        # Spinor dimension
        self.spinor_dim = 2 ** (self.dim_int // 2)
        if self.dim_int % 2 == 1:
            self.spinor_dim = 2 ** ((self.dim_int + 1) // 2)
            
        # Build Dirac operator matrix
        self.dirac_matrix = self._build_dirac_matrix()
    
    def _initialize_gamma_matrices(self, dimension):
        """
        Initialize gamma matrices for the given dimension.
        
        Parameters:
        -----------
        dimension : int
            Integer spacetime dimension
            
        Returns:
        --------
        list
            List of gamma matrices
        """
        if dimension == 2:
            # 2D gamma matrices (2x2)
            gamma0 = np.array([[0, 1], [1, 0]])
            gamma1 = np.array([[0, -1j], [1j, 0]])
            return [gamma0, gamma1]
            
        elif dimension == 3:
            # 3D gamma matrices (2x2)
            gamma0 = np.array([[0, 1], [1, 0]])
            gamma1 = np.array([[0, -1j], [1j, 0]])
            gamma2 = np.array([[1, 0], [0, -1]])
            return [gamma0, gamma1, gamma2]
            
        elif dimension == 4:
            # 4D gamma matrices (4x4)
            # Dirac representation
            sigma1 = np.array([[0, 1], [1, 0]])
            sigma2 = np.array([[0, -1j], [1j, 0]])
            sigma3 = np.array([[1, 0], [0, -1]])
            
            gamma0 = np.block([
                [np.eye(2), np.zeros((2, 2))],
                [np.zeros((2, 2)), -np.eye(2)]
            ])
            
            gamma1 = np.block([
                [np.zeros((2, 2)), sigma1],
                [-sigma1, np.zeros((2, 2))]
            ])
            
            gamma2 = np.block([
                [np.zeros((2, 2)), sigma2],
                [-sigma2, np.zeros((2, 2))]
            ])
            
            gamma3 = np.block([
                [np.zeros((2, 2)), sigma3],
                [-sigma3, np.zeros((2, 2))]
            ])
            
            return [gamma0, gamma1, gamma2, gamma3]
            
        elif dimension == 5:
            # 5D gamma matrices (4x4)
            # First get 4D matrices
            gamma_4d = self._initialize_gamma_matrices(4)
            
            # Add gamma5 = i * gamma0 * gamma1 * gamma2 * gamma3
            gamma5 = np.array([[0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0]])
            
            return gamma_4d + [gamma5]
            
        else:
            raise ValueError(f"Gamma matrices for dimension {dimension} not implemented")
    
    def _build_dirac_matrix(self):
        """
        Build the sparse Dirac operator matrix.
        
        Returns:
        --------
        scipy.sparse.csr_matrix
            Sparse matrix representation of the Dirac operator
        """
        print(f"Building Dirac operator for dimension {self.dimension:.3f}...")
        
        # Size of the matrix
        N = self.total_sites * self.spinor_dim
        
        # Lists for sparse matrix construction
        row_indices = []
        col_indices = []
        values = []
        
        # Add mass term (diagonal)
        for i in range(N):
            row_indices.append(i)
            col_indices.append(i)
            values.append(self.mass)
        
        # Add derivative terms (hopping terms)
        for site_idx in range(self.total_sites):
            # Get site coordinates
            coords = self._index_to_coord(site_idx)
            
            # For each direction
            for mu in range(self.dim_int):
                # Forward and backward neighbors
                fwd_coords = list(coords)
                fwd_coords[mu] = (fwd_coords[mu] + 1) % self.lattice_size
                fwd_idx = self._coord_to_index(fwd_coords)
                
                bwd_coords = list(coords)
                bwd_coords[mu] = (bwd_coords[mu] - 1) % self.lattice_size
                bwd_idx = self._coord_to_index(bwd_coords)
                
                # Gamma matrix for this direction
                gamma = self.gamma_matrices[mu]
                
                # Add hopping terms for each spinor component
                for a in range(self.spinor_dim):
                    for b in range(self.spinor_dim):
                        # Forward hop with +gamma/2
                        row_indices.append(site_idx * self.spinor_dim + a)
                        col_indices.append(fwd_idx * self.spinor_dim + b)
                        values.append(0.5 * gamma[a, b])
                        
                        # Backward hop with -gamma/2
                        row_indices.append(site_idx * self.spinor_dim + a)
                        col_indices.append(bwd_idx * self.spinor_dim + b)
                        values.append(-0.5 * gamma[a, b])
        
        # Build sparse matrix
        dirac_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(N, N))
        
        # Apply dimension correction for non-integer dimensions
        # This is where the dimensional flow effects come in
        dim_correction = (self.dimension - self.dim_int)
        if abs(dim_correction) > 1e-6:
            # For fractional dimensions, we modify the Dirac operator
            # by including a dimension-dependent scaling factor
            
            # Scale the off-diagonal elements (derivative terms)
            # This mimics how derivatives behave in fractional dimensions
            dirac_matrix_diag = csr_matrix((dirac_matrix.diagonal(), 
                                          (np.arange(N), np.arange(N))), 
                                         shape=(N, N))
            
            dirac_matrix_offdiag = dirac_matrix - dirac_matrix_diag
            
            # Apply power-law scaling to off-diagonal elements
            # This represents how the derivative operator changes with dimension
            scale_factor = 1.0 + 0.5 * dim_correction
            dirac_matrix = dirac_matrix_diag + scale_factor * dirac_matrix_offdiag
        
        return dirac_matrix
    
    def _index_to_coord(self, index):
        """
        Convert flat index to lattice coordinates.
        
        Parameters:
        -----------
        index : int
            Flat index
            
        Returns:
        --------
        tuple
            Lattice coordinates
        """
        coords = []
        for d in range(self.dim_int):
            coords.append(index % self.lattice_size)
            index //= self.lattice_size
        return tuple(coords)
    
    def _coord_to_index(self, coords):
        """
        Convert lattice coordinates to flat index.
        
        Parameters:
        -----------
        coords : tuple or list
            Lattice coordinates
            
        Returns:
        --------
        int
            Flat index
        """
        index = 0
        for d in range(self.dim_int):
            index += coords[d] * (self.lattice_size ** d)
        return index
    
    def apply(self, spinor):
        """
        Apply the Dirac operator to a spinor field.
        
        Parameters:
        -----------
        spinor : ndarray
            Spinor field
            
        Returns:
        --------
        ndarray
            Result of D*spinor
        """
        if spinor.shape[0] != self.total_sites * self.spinor_dim:
            raise ValueError(f"Spinor dimension mismatch. Expected {self.total_sites * self.spinor_dim}, got {spinor.shape[0]}")
            
        return self.dirac_matrix.dot(spinor)
    
    def eigenvalues(self, k=10):
        """
        Compute the smallest eigenvalues of the Dirac operator.
        
        Parameters:
        -----------
        k : int
            Number of eigenvalues to compute
            
        Returns:
        --------
        ndarray
            Eigenvalues
        """
        # Use sparse eigenvalue solver for efficiency
        # Compute the smallest magnitude eigenvalues
        eigenvalues = splinalg.eigsh(self.dirac_matrix, k=k, which='SM', return_eigenvectors=False)
        return eigenvalues


class FermionTreatment:
    """
    Implements fermion treatment with dimensional flow effects.
    """
    
    def __init__(self, dim_uv=2.0, dim_ir=4.0, transition_scale=1.0, 
                 lattice_size=10, mass=0.1, gauge_group="U1"):
        """
        Initialize fermion treatment.
        
        Parameters:
        -----------
        dim_uv : float
            UV (high energy) spectral dimension
        dim_ir : float
            IR (low energy) spectral dimension
        transition_scale : float
            Scale of dimension transition (in Planck units)
        lattice_size : int
            Size of the lattice for discretization
        mass : float
            Fermion mass
        gauge_group : str
            Gauge group for gauge-fermion interactions
        """
        self.dim_uv = dim_uv
        self.dim_ir = dim_ir
        self.transition_scale = transition_scale
        self.lattice_size = lattice_size
        self.mass = mass
        self.gauge_group = gauge_group
        
        # Initialize dimensional flow RG
        self.rg = DimensionalFlowRG(
            dim_uv=dim_uv,
            dim_ir=dim_ir,
            transition_scale=transition_scale
        )
        
        # Initialize gauge field integration if needed
        self.gauge = None
        if gauge_group:
            self.gauge = GaugeFieldIntegration(
                gauge_group=gauge_group,
                dim_uv=dim_uv,
                dim_ir=dim_ir,
                transition_scale=transition_scale,
                lattice_size=lattice_size
            )
        
        # Store Dirac operators at different dimensions
        self.dirac_operators = {}
        
        # Store results
        self.results = {}
    
    def get_dirac_operator(self, energy_scale, recompute=False):
        """
        Get Dirac operator at a specific energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        recompute : bool
            Whether to recompute the operator if it already exists
            
        Returns:
        --------
        DiracOperator
            Dirac operator at the given scale
        """
        # Get dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Check if we already have this operator
        key = f"{dimension:.3f}"
        if key in self.dirac_operators and not recompute:
            return self.dirac_operators[key]
        
        # Get running mass at this scale
        running_mass = self._compute_running_mass(energy_scale)
        
        # Create new Dirac operator
        dirac = DiracOperator(
            dimension=dimension,
            lattice_size=self.lattice_size,
            mass=running_mass
        )
        
        self.dirac_operators[key] = dirac
        return dirac
    
    def _compute_running_mass(self, energy_scale):
        """
        Compute the running fermion mass at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Running mass
        """
        # Compute RG flow if not already computed
        if not self.rg.flow_results:
            self.rg.compute_rg_flow(scale_range=(energy_scale*0.1, energy_scale*10))
        
        # Get dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Extract running couplings
        scales = self.rg.flow_results['scales']
        idx = np.abs(scales - energy_scale).argmin()
        
        # If we have Yukawa coupling 'y' in RG results, use it to compute mass
        if 'y' in self.rg.flow_results['coupling_trajectories']:
            y_coupling = self.rg.flow_results['coupling_trajectories']['y'][idx]
            # Running mass m ~ y * v, where v is the VEV
            # Use simplified calculation
            running_mass = self.mass * y_coupling / 0.5  # Normalize to y₀ = 0.5
        else:
            # Default scaling behavior if RG doesn't have 'y'
            # Canonical scaling for fermion mass: [m] = 1 in any dimension
            # So the mass runs primarily due to anomalous dimension
            
            # Simple estimate of anomalous dimension
            if 'g' in self.rg.flow_results['coupling_trajectories']:
                g_coupling = self.rg.flow_results['coupling_trajectories']['g'][idx]
                gamma_m = 0.05 * g_coupling**2  # Simplified anomalous dimension
            else:
                gamma_m = 0.02 * (dimension - self.dim_ir)  # Simple estimate
                
            # Running mass with anomalous dimension
            running_mass = self.mass * (energy_scale / self.transition_scale)**gamma_m
        
        return running_mass
    
    def compute_dirac_spectrum(self, energy_scale, num_eigenvalues=10):
        """
        Compute the spectrum of the Dirac operator at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        num_eigenvalues : int
            Number of eigenvalues to compute
            
        Returns:
        --------
        dict
            Dirac spectrum results
        """
        print(f"Computing Dirac spectrum at scale {energy_scale:.2e}...")
        
        # Get dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Get Dirac operator
        dirac = self.get_dirac_operator(energy_scale)
        
        # Compute eigenvalues
        eigenvalues = dirac.eigenvalues(k=num_eigenvalues)
        
        # Sort by magnitude
        eigenvalues = sorted(eigenvalues, key=abs)
        
        print(f"  Dimension: {dimension:.3f}")
        print(f"  Smallest eigenvalues: {eigenvalues[:5]}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'mass': dirac.mass,
            'eigenvalues': eigenvalues
        }
        
        return result
    
    def compute_multiscale_dirac_spectrum(self, num_scales=5, num_eigenvalues=10):
        """
        Compute Dirac spectrum across multiple energy scales.
        
        Parameters:
        -----------
        num_scales : int
            Number of energy scales to compute
        num_eigenvalues : int
            Number of eigenvalues to compute at each scale
            
        Returns:
        --------
        dict
            Multiscale Dirac spectrum results
        """
        print(f"Computing Dirac spectrum across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-3, 3, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'masses': [],
            'spectra': []
        }
        
        # Compute spectrum at each scale
        for scale in energy_scales:
            spectrum = self.compute_dirac_spectrum(
                energy_scale=scale,
                num_eigenvalues=num_eigenvalues
            )
            
            # Store results
            results['dimensions'].append(spectrum['dimension'])
            results['masses'].append(spectrum['mass'])
            results['spectra'].append(spectrum['eigenvalues'])
        
        # Store complete results
        self.results['multiscale_dirac_spectrum'] = results
        return results
    
    def compute_chiral_condensate(self, energy_scale, num_samples=10):
        """
        Compute the chiral condensate at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        num_samples : int
            Number of samples for stochastic estimation
            
        Returns:
        --------
        dict
            Chiral condensate results
        """
        print(f"Computing chiral condensate at scale {energy_scale:.2e}...")
        
        # Get dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Get Dirac operator
        dirac = self.get_dirac_operator(energy_scale)
        
        # Compute chiral condensate
        # <ψ̄ψ> = Tr(D^-1)
        # We'll use a stochastic estimator for the trace
        
        # Matrix size
        N = dirac.total_sites * dirac.spinor_dim
        
        # Stochastic estimation of the trace
        condensate_samples = []
        
        for _ in range(num_samples):
            # Random vector with +/-1 elements
            eta = np.random.choice([-1, 1], size=N)
            
            # Solve D*x = η
            # For simplicity, use a direct solver
            # In practice, would use conjugate gradient or other iterative method
            x = splinalg.spsolve(dirac.dirac_matrix, eta)
            
            # Estimate tr(D^-1) = η† · x
            condensate = np.dot(eta, x) / N
            condensate_samples.append(condensate)
        
        # Compute mean and error
        condensate_mean = np.mean(condensate_samples)
        condensate_error = np.std(condensate_samples) / np.sqrt(num_samples)
        
        print(f"  Dimension: {dimension:.3f}")
        print(f"  Chiral condensate: {condensate_mean:.6f} ± {condensate_error:.6f}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'mass': dirac.mass,
            'condensate_mean': condensate_mean,
            'condensate_error': condensate_error,
            'condensate_samples': condensate_samples
        }
        
        return result
    
    def compute_multiscale_chiral_condensate(self, num_scales=5, num_samples=10):
        """
        Compute chiral condensate across multiple energy scales.
        
        Parameters:
        -----------
        num_scales : int
            Number of energy scales to compute
        num_samples : int
            Number of samples for stochastic estimation
            
        Returns:
        --------
        dict
            Multiscale chiral condensate results
        """
        print(f"Computing chiral condensate across {num_scales} energy scales...")
        
        # Generate logarithmically spaced energy scales
        energy_scales = np.logspace(-3, 3, num_scales)
        
        # Store results
        results = {
            'energy_scales': energy_scales,
            'dimensions': [],
            'masses': [],
            'condensates': [],
            'errors': []
        }
        
        # Compute condensate at each scale
        for scale in energy_scales:
            condensate = self.compute_chiral_condensate(
                energy_scale=scale,
                num_samples=num_samples
            )
            
            # Store results
            results['dimensions'].append(condensate['dimension'])
            results['masses'].append(condensate['mass'])
            results['condensates'].append(condensate['condensate_mean'])
            results['errors'].append(condensate['condensate_error'])
        
        # Store complete results
        self.results['multiscale_chiral_condensate'] = results
        return results
    
    def compute_fermion_determinant(self, energy_scale, num_samples=5):
        """
        Compute the fermion determinant at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        num_samples : int
            Number of samples for stochastic estimation
            
        Returns:
        --------
        dict
            Fermion determinant results
        """
        print(f"Computing fermion determinant at scale {energy_scale:.2e}...")
        
        # Get dimension at this scale
        dimension = self.rg.compute_spectral_dimension(energy_scale)
        
        # Get Dirac operator
        dirac = self.get_dirac_operator(energy_scale)
        
        # For a full determinant, we'd need eigenvalues
        # For demonstration purposes, compute a few smallest eigenvalues
        eigenvalues = dirac.eigenvalues(k=20)
        
        # Estimate log(det(D)) = Tr(log(D))
        # For positive definite matrices, this is sum of log of eigenvalues
        # For Dirac operator, need to handle complex eigenvalues
        log_det_est = np.sum(np.log(np.abs(eigenvalues)))
        
        # Scale appropriately for full determinant
        # This is a very rough estimate
        N = dirac.total_sites * dirac.spinor_dim
        scaling_factor = N / len(eigenvalues)
        log_det = log_det_est * scaling_factor
        
        print(f"  Dimension: {dimension:.3f}")
        print(f"  Log determinant (estimated): {log_det:.6f}")
        
        # Store results
        result = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'mass': dirac.mass,
            'log_determinant': log_det,
            'eigenvalues': eigenvalues
        }
        
        return result
    
    def plot_dirac_spectrum(self, save_path=None):
        """
        Plot Dirac spectrum across energy scales.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'multiscale_dirac_spectrum' not in self.results:
            raise ValueError("No Dirac spectrum results available. Run compute_multiscale_dirac_spectrum first.")
            
        results = self.results['multiscale_dirac_spectrum']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Smallest eigenvalue vs energy scale
        smallest_eigs = [np.min(np.abs(spectrum)) for spectrum in results['spectra']]
        
        axs[0].loglog(results['energy_scales'], smallest_eigs, 'ro-', linewidth=2)
        axs[0].set_xlabel('Energy Scale (Planck units)')
        axs[0].set_ylabel('Smallest Eigenvalue')
        axs[0].set_title('Dirac Spectrum Gap vs Energy Scale')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Smallest eigenvalue vs spectral dimension
        axs[1].plot(results['dimensions'], smallest_eigs, 'bo-', linewidth=2)
        axs[1].set_xlabel('Spectral Dimension')
        axs[1].set_ylabel('Smallest Eigenvalue')
        axs[1].set_title('Dirac Spectrum Gap vs Dimension')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_chiral_condensate(self, save_path=None):
        """
        Plot chiral condensate across energy scales.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if 'multiscale_chiral_condensate' not in self.results:
            raise ValueError("No chiral condensate results available. Run compute_multiscale_chiral_condensate first.")
            
        results = self.results['multiscale_chiral_condensate']
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Chiral condensate vs energy scale
        axs[0].errorbar(
            results['energy_scales'],
            results['condensates'],
            yerr=results['errors'],
            fmt='ro-',
            linewidth=2,
            capsize=5
        )
        
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Energy Scale (Planck units)')
        axs[0].set_ylabel('Chiral Condensate |<ψ̄ψ>|')
        axs[0].set_title('Chiral Condensate vs Energy Scale')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Chiral condensate vs spectral dimension
        axs[1].errorbar(
            results['dimensions'],
            results['condensates'],
            yerr=results['errors'],
            fmt='bo-',
            linewidth=2,
            capsize=5
        )
        
        axs[1].set_xlabel('Spectral Dimension')
        axs[1].set_ylabel('Chiral Condensate |<ψ̄ψ>|')
        axs[1].set_title('Chiral Condensate vs Dimension')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def derive_fermion_dimension_scaling(self):
        """
        Derive how fermion observables scale with spectral dimension.
        
        Returns:
        --------
        dict
            Scaling relations
        """
        print("Deriving fermion dimension scaling laws...")
        
        # Need both spectrum and condensate results
        if 'multiscale_dirac_spectrum' not in self.results or 'multiscale_chiral_condensate' not in self.results:
            raise ValueError("Need both spectrum and condensate results. Run computations first.")
        
        spec_results = self.results['multiscale_dirac_spectrum']
        cond_results = self.results['multiscale_chiral_condensate']
        
        # Extract data
        dimensions = spec_results['dimensions']
        smallest_eigs = [np.min(np.abs(spectrum)) for spectrum in spec_results['spectra']]
        condensates = cond_results['condensates']
        
        # Define fit function for smallest eigenvalue
        # λ_min(d) ~ (d-2)^α
        def spec_scaling_func(d, alpha, gamma):
            return gamma * np.abs(d - 2.0)**alpha
        
        # Define fit function for condensate
        # <ψ̄ψ>(d) ~ (d-2)^β
        def cond_scaling_func(d, beta, delta):
            return delta * np.abs(d - 2.0)**beta
        
        # Fit parameters using curve_fit
        from scipy.optimize import curve_fit
        
        try:
            # Fit spectrum scaling
            popt_spec, _ = curve_fit(
                spec_scaling_func,
                dimensions,
                smallest_eigs,
                p0=[1.0, 0.1],
                bounds=([0, 0], [10, 10])
            )
            
            alpha, gamma = popt_spec
            spec_formula = f"λ_min(d) = {gamma:.4f} · |d - 2|^{alpha:.4f}"
            
            # Fit condensate scaling
            popt_cond, _ = curve_fit(
                cond_scaling_func,
                dimensions,
                condensates,
                p0=[1.0, 0.1],
                bounds=([0, 0], [10, 10])
            )
            
            beta, delta = popt_cond
            cond_formula = f"<ψ̄ψ>(d) = {delta:.4f} · |d - 2|^{beta:.4f}"
            
            print(f"  Spectrum scaling: {spec_formula}")
            print(f"  Condensate scaling: {cond_formula}")
            
            # Store scaling laws
            scaling_laws = {
                'spectrum': {
                    'formula': spec_formula,
                    'parameters': {
                        'alpha': alpha,
                        'gamma': gamma
                    }
                },
                'condensate': {
                    'formula': cond_formula,
                    'parameters': {
                        'beta': beta,
                        'delta': delta
                    }
                }
            }
            
            # Store in results
            self.results['fermion_scaling_laws'] = scaling_laws
            return scaling_laws
            
        except:
            print("  Failed to fit scaling functions. May need more data points.")
            return None


if __name__ == "__main__":
    # Test the fermion treatment
    
    # Create a fermion treatment instance
    fermion_treatment = FermionTreatment(
        dim_uv=2.0,
        dim_ir=4.0,
        transition_scale=1.0,
        lattice_size=8,  # Small for testing
        mass=0.1,
        gauge_group="SU3"  # For quark-like fermions
    )
    
    # Compute Dirac spectrum across energy scales
    spec_results = fermion_treatment.compute_multiscale_dirac_spectrum(
        num_scales=3,
        num_eigenvalues=10
    )
    
    # Compute chiral condensate across energy scales
    cond_results = fermion_treatment.compute_multiscale_chiral_condensate(
        num_scales=3,
        num_samples=5
    )
    
    # Derive fermion dimension scaling laws
    scaling_laws = fermion_treatment.derive_fermion_dimension_scaling()
    
    # Plot results
    fermion_treatment.plot_dirac_spectrum(save_path="dirac_spectrum.png")
    fermion_treatment.plot_chiral_condensate(save_path="chiral_condensate.png")
    
    print("\nFermion treatment test complete.") 