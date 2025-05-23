"""
Holographic QFT Emergence from Entanglement

This module implements concrete models showing how quantum field theories
emerge from entanglement structures in the holographic framework of quantum gravity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, linalg as splinalg
import networkx as nx

# Fix imports for local testing
try:
    from quantum_gravity_framework.holographic_duality import ExtendedHolographicDuality
    from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
except ImportError:
    from holographic_duality import ExtendedHolographicDuality
    from quantum_spacetime import QuantumSpacetimeAxioms


class HolographicQFTEmergence:
    """
    Implements models of QFT emergence from holographic entanglement structures.
    """
    
    def __init__(self, boundary_dim=3, bulk_dim=4, network_size=10, bond_dim=4):
        """
        Initialize the holographic QFT emergence model.
        
        Parameters:
        -----------
        boundary_dim : int
            Dimension of the boundary theory
        bulk_dim : int
            Dimension of the bulk geometry
        network_size : int
            Size of the network
        bond_dim : int
            Bond dimension for tensor network
        """
        self.boundary_dim = boundary_dim
        self.bulk_dim = bulk_dim
        self.network_size = network_size
        self.bond_dim = bond_dim
        
        # Initialize holographic duality components
        self.holographic = ExtendedHolographicDuality(
            boundary_dim=boundary_dim,
            bulk_dim=bulk_dim
        )
        
        # Quantum spacetime model
        self.spacetime = QuantumSpacetimeAxioms(dim=bulk_dim)
        
        # Initialize entanglement network
        self.entanglement_network = self._build_entanglement_network()
        
        # Store QFT data
        self.qft_operators = {}
        self.qft_correlators = {}
        self.emergent_properties = {}
    
    def _build_entanglement_network(self):
        """
        Build a network representing entanglement structure.
        
        Returns:
        --------
        networkx.Graph
            The entanglement network
        """
        # Create a graph to represent entanglement structure
        G = nx.Graph()
        
        # First create boundary nodes (arranged in a ring for visualization)
        boundary_nodes = []
        for i in range(self.network_size):
            angle = 2 * np.pi * i / self.network_size
            pos = (np.cos(angle), np.sin(angle), 0)
            node_id = f"boundary_{i}"
            G.add_node(node_id, pos=pos, type="boundary")
            boundary_nodes.append(node_id)
        
        # Create bulk nodes
        bulk_nodes = []
        for i in range(self.network_size // 2):
            # Position bulk nodes inside the boundary ring
            r = 0.7 * np.random.random()
            angle = 2 * np.pi * np.random.random()
            pos = (r * np.cos(angle), r * np.sin(angle), np.random.random() * 0.5)
            node_id = f"bulk_{i}"
            G.add_node(node_id, pos=pos, type="bulk")
            bulk_nodes.append(node_id)
        
        # Connect boundary nodes (nearest neighbors)
        for i in range(self.network_size):
            G.add_edge(
                boundary_nodes[i], 
                boundary_nodes[(i+1) % self.network_size],
                weight=1.0,
                type="boundary-boundary"
            )
        
        # Connect bulk nodes to boundary (using holographic minimal surfaces idea)
        # Each bulk node connects to a subset of boundary nodes
        for i, bulk_node in enumerate(bulk_nodes):
            # Number of boundary connections varies
            n_connections = np.random.randint(3, max(4, self.network_size // 3))
            
            # Choose random boundary nodes
            connected_boundaries = np.random.choice(
                boundary_nodes, size=n_connections, replace=False
            )
            
            # Create edges
            for boundary_node in connected_boundaries:
                # Edge weight based on distance (simplified)
                pos_bulk = G.nodes[bulk_node]["pos"]
                pos_boundary = G.nodes[boundary_node]["pos"]
                dist = np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos_bulk, pos_boundary)))
                weight = 1.0 / (1.0 + dist)
                
                G.add_edge(
                    bulk_node, boundary_node,
                    weight=weight,
                    type="bulk-boundary"
                )
        
        # Connect bulk nodes (representing spacetime entanglement)
        for i in range(len(bulk_nodes)):
            for j in range(i+1, len(bulk_nodes)):
                # Add edge with some probability (based on distance)
                pos_i = G.nodes[bulk_nodes[i]]["pos"]
                pos_j = G.nodes[bulk_nodes[j]]["pos"]
                dist = np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos_i, pos_j)))
                
                # Probability decreases with distance
                prob = 1.0 / (1.0 + 5.0 * dist)
                
                if np.random.random() < prob:
                    G.add_edge(
                        bulk_nodes[i], bulk_nodes[j],
                        weight=prob,
                        type="bulk-bulk"
                    )
        
        return G
    
    def compute_entanglement_spectrum(self):
        """
        Compute the entanglement spectrum of the network.
        
        This reveals the eigenvalues of the entanglement Hamiltonian,
        which is related to the emergent QFT structure.
        
        Returns:
        --------
        dict
            Entanglement spectrum results
        """
        print("Computing entanglement spectrum...")
        
        # Build adjacency matrix from graph
        nodes = list(self.entanglement_network.nodes())
        n_nodes = len(nodes)
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Create adjacency matrix
        adj_data = []
        adj_row = []
        adj_col = []
        
        for u, v, data in self.entanglement_network.edges(data=True):
            i, j = node_indices[u], node_indices[v]
            weight = data.get('weight', 1.0)
            
            adj_data.append(weight)
            adj_row.append(i)
            adj_col.append(j)
            
            # Make symmetric
            adj_data.append(weight)
            adj_row.append(j)
            adj_col.append(i)
        
        adjacency = csr_matrix((adj_data, (adj_row, adj_col)), shape=(n_nodes, n_nodes))
        
        # Compute Laplacian matrix
        degree = np.array(adjacency.sum(axis=1)).flatten()
        degree_matrix = csr_matrix((degree, (range(n_nodes), range(n_nodes))), shape=(n_nodes, n_nodes))
        laplacian = degree_matrix - adjacency
        
        # Compute eigenvalues and eigenvectors
        try:
            # Compute a subset of eigenvalues (the smallest ones)
            k = min(n_nodes - 2, 20)  # Number of eigenvalues to compute
            eigenvalues, eigenvectors = splinalg.eigsh(laplacian, k=k, which='SM')
            
            # Sort by eigenvalue
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except:
            # Fallback to dense computation for small matrices
            if n_nodes < 100:
                dense_lap = laplacian.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(dense_lap)
                
                # Sort by eigenvalue
                idx = eigenvalues.argsort()
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
            else:
                # For larger matrices, just compute a few eigenvalues
                eigenvalues = splinalg.eigsh(laplacian, k=5, which='SM', return_eigenvectors=False)
                eigenvalues.sort()
                eigenvectors = None
        
        # Store results
        results = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'node_indices': node_indices,
            'nodes': nodes
        }
        
        # Extract boundary spectrum (eigenvalues corresponding to boundary nodes)
        boundary_indices = [node_indices[n] for n in nodes if n.startswith('boundary_')]
        
        if eigenvectors is not None:
            boundary_projections = np.zeros(len(eigenvalues))
            for i, eigvec in enumerate(eigenvectors.T):
                # Calculate how much this eigenvector projects onto boundary subspace
                boundary_norm = np.sum(eigvec[boundary_indices]**2)
                boundary_projections[i] = boundary_norm
            
            results['boundary_projections'] = boundary_projections
        
        return results
    
    def derive_qft_operators(self):
        """
        Derive QFT operators from the entanglement structure.
        
        Returns:
        --------
        dict
            Dictionary of QFT operators
        """
        print("Deriving QFT operators from entanglement structure...")
        
        # First compute entanglement spectrum
        spectrum = self.compute_entanglement_spectrum()
        eigenvalues = spectrum['eigenvalues']
        eigenvectors = spectrum['eigenvectors']
        nodes = spectrum['nodes']
        node_indices = spectrum['node_indices']
        
        # Identify boundary nodes
        boundary_nodes = [n for n in nodes if n.startswith('boundary_')]
        boundary_indices = [node_indices[n] for n in boundary_nodes]
        
        # Create field operators on the boundary
        # Scalar field, for simplicity
        phi_op = np.zeros((len(nodes), len(nodes)))
        
        # Construct field operator from low energy eigenmodes
        # We use the first few non-zero modes (skip the zero mode)
        n_modes = min(5, len(eigenvalues) - 1)
        for i in range(1, n_modes + 1):
            mode = eigenvectors[:, i]
            # Weight by eigenvalue (energy)
            weight = 1.0 / np.sqrt(max(eigenvalues[i], 1e-10))
            phi_op += weight * np.outer(mode, mode)
            
        # Restrict to boundary
        phi_boundary = phi_op[np.ix_(boundary_indices, boundary_indices)]
        
        # Derive momentum operator (from generator of translations)
        # Simple finite difference approximation
        pi_boundary = np.zeros_like(phi_boundary)
        
        for i in range(len(boundary_nodes)):
            next_i = (i + 1) % len(boundary_nodes)
            pi_boundary[i, next_i] = 1.0
            pi_boundary[next_i, i] = -1.0
        
        pi_boundary *= 0.5j  # Make it hermitian
        
        # Construct Hamiltonian for the field theory
        # Simple scalar field Hamiltonian: H = π²/2 + (∇φ)²/2 + m²φ²/2
        mass_squared = 0.1
        
        # Kinetic term (π²/2)
        kinetic = 0.5 * np.dot(pi_boundary, pi_boundary)
        
        # Gradient term ((∇φ)²/2)
        # Simple finite difference for Laplacian
        laplacian = np.zeros_like(phi_boundary)
        for i in range(len(boundary_nodes)):
            prev_i = (i - 1) % len(boundary_nodes)
            next_i = (i + 1) % len(boundary_nodes)
            
            laplacian[i, i] = -2.0
            laplacian[i, prev_i] = 1.0
            laplacian[i, next_i] = 1.0
        
        gradient = 0.5 * np.dot(phi_boundary, np.dot(laplacian, phi_boundary))
        
        # Mass term (m²φ²/2)
        mass = 0.5 * mass_squared * np.dot(phi_boundary, phi_boundary)
        
        # Full Hamiltonian
        hamiltonian = kinetic + gradient + mass
        
        # Store operators
        self.qft_operators = {
            'phi': phi_boundary,
            'pi': pi_boundary,
            'hamiltonian': hamiltonian,
            'full_phi': phi_op,
            'eigenvalues': eigenvalues,
            'boundary_nodes': boundary_nodes
        }
        
        return self.qft_operators
    
    def compute_qft_correlators(self):
        """
        Compute QFT correlation functions from the entanglement structure.
        
        Returns:
        --------
        dict
            Dictionary of correlation functions
        """
        print("Computing QFT correlation functions...")
        
        # Make sure we have QFT operators
        if not self.qft_operators:
            self.derive_qft_operators()
        
        # Extract relevant operators
        phi = self.qft_operators['phi']
        eigenvalues = self.qft_operators['eigenvalues']
        
        # Calculate two-point function <φ(x)φ(y)>
        # In the ground state, this is given by the inverse of the Laplacian
        
        # Extract Laplacian eigenvalues and eigenmodes
        # Skip the zero mode (i=0)
        n_modes = min(20, len(eigenvalues) - 1)
        
        # Construct two-point function from eigenmodes
        two_point = np.zeros_like(phi)
        for i in range(1, n_modes + 1):
            if eigenvalues[i] > 1e-10:  # Avoid division by zero
                two_point += (1.0 / eigenvalues[i]) * np.outer(phi[i], phi[i])
        
        # Calculate propagator (Green's function)
        # This is essentially the two-point function
        propagator = two_point.copy()
        
        # Calculate four-point function (disconnected part)
        # For a free theory: <φ(x)φ(y)φ(z)φ(w)> = <φ(x)φ(y)><φ(z)φ(w)> + <φ(x)φ(z)><φ(y)φ(w)> + <φ(x)φ(w)><φ(y)φ(z)>
        
        # Just compute it for a subset of points for efficiency
        selected_points = min(5, phi.shape[0])
        four_point = np.zeros((selected_points, selected_points, selected_points, selected_points))
        
        for x in range(selected_points):
            for y in range(selected_points):
                for z in range(selected_points):
                    for w in range(selected_points):
                        four_point[x, y, z, w] = (
                            two_point[x, y] * two_point[z, w] +
                            two_point[x, z] * two_point[y, w] +
                            two_point[x, w] * two_point[y, z]
                        )
        
        # Store correlators
        self.qft_correlators = {
            'two_point': two_point,
            'propagator': propagator,
            'four_point': four_point
        }
        
        return self.qft_correlators
    
    def analyze_emergent_properties(self):
        """
        Analyze emergent properties of the QFT from the holographic model.
        
        Returns:
        --------
        dict
            Emergent properties
        """
        print("Analyzing emergent QFT properties...")
        
        # Make sure we have correlators
        if not self.qft_correlators:
            self.compute_qft_correlators()
        
        # Extract correlators
        two_point = self.qft_correlators['two_point']
        
        # Calculate correlation length
        # We'll look at how the two-point function decays with distance
        boundary_nodes = self.qft_operators['boundary_nodes']
        n_boundary = len(boundary_nodes)
        
        # Compute average correlations at different distances
        distances = []
        avg_correlations = []
        
        for d in range(1, n_boundary // 2 + 1):
            corrs = []
            for i in range(n_boundary):
                j = (i + d) % n_boundary
                corrs.append(np.abs(two_point[i, j]))
            
            distances.append(d)
            avg_correlations.append(np.mean(corrs))
        
        # Estimate correlation length by fitting to exponential decay
        # C(r) ~ exp(-r/ξ)
        if len(distances) > 2:
            try:
                log_corr = np.log(avg_correlations)
                fit = np.polyfit(distances, log_corr, 1)
                correlation_length = -1.0 / fit[0] if fit[0] < 0 else float('inf')
            except:
                correlation_length = np.nan
        else:
            correlation_length = np.nan
        
        # Estimate spectral dimension from eigenvalue distribution
        # For a d-dimensional theory, eigenvalue density ρ(λ) ~ λ^(d/2-1)
        eigenvalues = np.array(self.qft_operators['eigenvalues'])
        if len(eigenvalues) > 5:
            try:
                # Use eigenvalues 2-6 (skip the smallest ones)
                log_eigenvalues = np.log(eigenvalues[2:7])
                log_indices = np.log(np.arange(2, 7))
                
                # Fit to a power law
                fit = np.polyfit(log_indices, log_eigenvalues, 1)
                spectral_dimension = 2.0 * (fit[0] + 1)
            except:
                spectral_dimension = np.nan
        else:
            spectral_dimension = np.nan
        
        # Check if the theory has conformal properties
        # In a CFT, the two-point function decays as a power law: C(r) ~ 1/r^(2Δ)
        # where Δ is the scaling dimension
        
        # Try power law fit
        if len(distances) > 2:
            try:
                log_distances = np.log(distances)
                log_corr = np.log(avg_correlations)
                fit = np.polyfit(log_distances, log_corr, 1)
                power_law_exponent = -fit[0]
                
                # If it fits well to a power law, it might be conformal
                residuals = log_corr - (fit[0] * log_distances + fit[1])
                residual_variance = np.var(residuals)
                
                # Define a threshold for what we consider "conformal-like"
                is_conformal = residual_variance < 0.1 and power_law_exponent > 0
                
                if is_conformal:
                    # Estimate scaling dimension
                    scaling_dimension = power_law_exponent / 2.0
                else:
                    scaling_dimension = np.nan
            except:
                is_conformal = False
                scaling_dimension = np.nan
                power_law_exponent = np.nan
        else:
            is_conformal = False
            scaling_dimension = np.nan
            power_law_exponent = np.nan
        
        # Store emergent properties
        self.emergent_properties = {
            'correlation_length': correlation_length,
            'spectral_dimension': spectral_dimension,
            'is_conformal': is_conformal,
            'scaling_dimension': scaling_dimension,
            'power_law_exponent': power_law_exponent,
            'distances': distances,
            'avg_correlations': avg_correlations
        }
        
        # Print summary
        print("\nEmergent QFT Properties:")
        print(f"  Correlation length: {correlation_length:.3f}")
        print(f"  Spectral dimension: {spectral_dimension:.3f}")
        print(f"  Conformal theory: {is_conformal}")
        if is_conformal:
            print(f"  Scaling dimension: {scaling_dimension:.3f}")
        
        return self.emergent_properties
    
    def visualize_entanglement_network(self, save_path=None):
        """
        Visualize the entanglement network.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        print("Visualizing entanglement network...")
        
        # Create a figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get node positions
        pos = nx.get_node_attributes(self.entanglement_network, 'pos')
        
        # Draw nodes with different colors for boundary and bulk
        boundary_nodes = [n for n in self.entanglement_network.nodes() if n.startswith('boundary_')]
        bulk_nodes = [n for n in self.entanglement_network.nodes() if n.startswith('bulk_')]
        
        # Draw boundary nodes
        xs = [pos[n][0] for n in boundary_nodes]
        ys = [pos[n][1] for n in boundary_nodes]
        zs = [pos[n][2] for n in boundary_nodes]
        ax.scatter(xs, ys, zs, c='blue', s=100, alpha=0.7, label='Boundary')
        
        # Draw bulk nodes
        xs = [pos[n][0] for n in bulk_nodes]
        ys = [pos[n][1] for n in bulk_nodes]
        zs = [pos[n][2] for n in bulk_nodes]
        ax.scatter(xs, ys, zs, c='red', s=100, alpha=0.7, label='Bulk')
        
        # Draw edges
        for u, v, data in self.entanglement_network.edges(data=True):
            edge_type = data.get('type', '')
            
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]
            
            if edge_type == 'boundary-boundary':
                color = 'blue'
                alpha = 0.5
            elif edge_type == 'bulk-boundary':
                color = 'purple'
                alpha = 0.3
            else:  # bulk-bulk
                color = 'red'
                alpha = 0.2
                
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=data.get('weight', 1.0))
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Holographic Entanglement Network')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            np.ptp([pos[n][0] for n in pos]),
            np.ptp([pos[n][1] for n in pos]),
            np.ptp([pos[n][2] for n in pos])
        ]).max() / 2.0
        
        mid_x = np.mean([pos[n][0] for n in pos])
        mid_y = np.mean([pos[n][1] for n in pos])
        mid_z = np.mean([pos[n][2] for n in pos])
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_functions(self, save_path=None):
        """
        Plot correlation functions of the emergent QFT.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        print("Plotting correlation functions...")
        
        # Make sure we have emergent properties
        if not self.emergent_properties:
            self.analyze_emergent_properties()
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        distances = self.emergent_properties['distances']
        correlations = self.emergent_properties['avg_correlations']
        
        # Plot correlation vs distance
        ax.semilogy(distances, correlations, 'bo-', linewidth=2, label='Two-Point Function')
        
        # Add fit lines if we have them
        if not np.isnan(self.emergent_properties['correlation_length']):
            # Exponential fit (matches correlation length)
            xi = self.emergent_properties['correlation_length']
            x_fit = np.linspace(min(distances), max(distances), 100)
            y_exp = correlations[0] * np.exp(-x_fit / xi)
            ax.semilogy(x_fit, y_exp, 'r--', linewidth=2, 
                      label=f'Exp. Fit: ξ = {xi:.2f}')
        
        if not np.isnan(self.emergent_properties['power_law_exponent']):
            # Power law fit (for conformal theories)
            alpha = self.emergent_properties['power_law_exponent']
            y_pow = correlations[0] * (x_fit / distances[0])**(-alpha)
            ax.semilogy(x_fit, y_pow, 'g--', linewidth=2, 
                      label=f'Power Fit: α = {alpha:.2f}')
        
        # Set axis labels and title
        ax.set_xlabel('Distance')
        ax.set_ylabel('Correlation Function')
        ax.set_title('Emergent QFT Correlation Functions')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test the holographic QFT emergence model
    
    # Create a model
    holographic_qft = HolographicQFTEmergence(
        boundary_dim=2,
        bulk_dim=3,
        network_size=20
    )
    
    # Visualize the entanglement network
    holographic_qft.visualize_entanglement_network(save_path="entanglement_network.png")
    
    # Derive QFT operators
    operators = holographic_qft.derive_qft_operators()
    
    # Compute correlators
    correlators = holographic_qft.compute_qft_correlators()
    
    # Analyze emergent properties
    properties = holographic_qft.analyze_emergent_properties()
    
    # Plot correlation functions
    holographic_qft.plot_correlation_functions(save_path="correlation_functions.png")
    
    print("\nHolographic QFT emergence model test complete.") 