"""
Numerical Simulations for Quantum Gravity

This module provides numerical methods and algorithms for simulating
quantum gravity effects in various contexts, including spacetime
discretization, path integral evaluation, and tensor network approaches.
"""

import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms
from quantum_gravity_framework.qft_integration import QFTIntegration


class DiscretizedSpacetime:
    """
    Implements a discretized spacetime lattice for numerical simulations.
    """
    
    def __init__(self, dim=4, size=10, lattice_spacing=1.0, boundary='periodic'):
        """
        Initialize a discretized spacetime.
        
        Parameters:
        -----------
        dim : int
            Number of spacetime dimensions
        size : int
            Number of lattice points in each dimension
        lattice_spacing : float
            Physical spacing between lattice points (in Planck units)
        boundary : str
            Boundary conditions ('periodic', 'open', or 'reflecting')
        """
        self.dim = dim
        self.size = size
        self.lattice_spacing = lattice_spacing
        self.boundary = boundary
        
        # Total number of points
        self.total_points = size**dim
        
        # Create lattice graph
        self._create_lattice_graph()
        
        # Initialize metric field (flat by default)
        self.metric_field = np.zeros((self.total_points, dim, dim))
        for i in range(self.total_points):
            # Start with Minkowski metric
            self.metric_field[i] = np.diag([-1.0] + [1.0] * (dim - 1))
            
        # Initialize scalar field (zero by default)
        self.scalar_field = np.zeros(self.total_points)
        
        # Curvature tensor (to be computed)
        self.riemann_tensor = None
        self.ricci_scalar = None
        
    def _create_lattice_graph(self):
        """
        Create a graph representing the lattice.
        """
        # Create a graph
        self.lattice_graph = nx.grid_graph(dim=[self.size] * self.dim)
        
        # Add periodic boundary conditions if needed
        if self.boundary == 'periodic':
            for dim_idx in range(self.dim):
                for indices in self._get_boundary_points(dim_idx):
                    # Connect opposite boundaries
                    node1 = tuple(indices)
                    node2 = list(indices)
                    node2[dim_idx] = 0 if indices[dim_idx] == self.size - 1 else self.size - 1
                    node2 = tuple(node2)
                    self.lattice_graph.add_edge(node1, node2)
                    
        # Add positions to nodes for visualization
        pos = {}
        for node in self.lattice_graph.nodes():
            pos[node] = np.array(node) * self.lattice_spacing
        nx.set_node_attributes(self.lattice_graph, pos, 'position')
    
    def _get_boundary_points(self, dimension):
        """Get all lattice points on the boundary of a given dimension."""
        points = []
        for idx in np.ndindex(*[self.size] * self.dim):
            if idx[dimension] == 0 or idx[dimension] == self.size - 1:
                points.append(idx)
        return points
    
    def _lattice_index(self, coords):
        """Convert n-dimensional coordinates to a flat index."""
        index = 0
        for i, coord in enumerate(coords):
            index += coord * (self.size ** i)
        return index
    
    def _coord_from_index(self, index):
        """Convert flat index to n-dimensional coordinates."""
        coords = []
        for i in range(self.dim):
            coords.append(index % self.size)
            index //= self.size
        return tuple(coords)
    
    def set_metric(self, metric_function):
        """
        Set the metric field using a function.
        
        Parameters:
        -----------
        metric_function : callable
            Function that takes coordinates and returns a metric tensor
        """
        for i in range(self.total_points):
            coords = self._coord_from_index(i)
            # Scale coordinates by lattice spacing
            physical_coords = np.array(coords) * self.lattice_spacing
            self.metric_field[i] = metric_function(physical_coords)
    
    def set_scalar_field(self, scalar_function):
        """
        Set the scalar field using a function.
        
        Parameters:
        -----------
        scalar_function : callable
            Function that takes coordinates and returns a scalar value
        """
        for i in range(self.total_points):
            coords = self._coord_from_index(i)
            # Scale coordinates by lattice spacing
            physical_coords = np.array(coords) * self.lattice_spacing
            self.scalar_field[i] = scalar_function(physical_coords)
    
    def compute_laplacian(self):
        """
        Compute the discrete Laplacian operator on the lattice.
        
        Returns:
        --------
        scipy.sparse.csr_matrix
            Sparse matrix representing the Laplacian
        """
        # Create empty sparse matrix
        lap = sp.lil_matrix((self.total_points, self.total_points))
        
        # Loop over all nodes
        for node in self.lattice_graph.nodes():
            i = self._lattice_index(node)
            # Diagonal element: -degree
            lap[i, i] = -len(list(self.lattice_graph.neighbors(node)))
            # Off-diagonal elements: 1 for each neighbor
            for neighbor in self.lattice_graph.neighbors(node):
                j = self._lattice_index(neighbor)
                lap[i, j] = 1
                
        return lap.tocsr()
    
    def compute_discrete_curvature(self):
        """
        Compute discrete approximation of curvature.
        
        Returns:
        --------
        float
            Average scalar curvature
        """
        # Simplified approach using graph properties
        # (A more accurate implementation would use discrete differential geometry)
        
        # Compute the Laplacian
        laplacian = self.compute_laplacian()
        
        # Use the eigenvalues of the Laplacian to estimate curvature
        # For a d-dimensional manifold, the leading behavior of the Laplacian spectrum
        # contains information about scalar curvature
        try:
            # Compute a few eigenvalues (k should be larger for better approximation)
            k = min(100, self.total_points - 2)
            eigenvalues = eigsh(laplacian, k=k, which='SM', return_eigenvectors=False)
            
            # Estimate scalar curvature from spectral properties
            # This is a simplified approximation
            avg_curvature = np.mean(eigenvalues) - self.dim
            self.ricci_scalar = avg_curvature
            
            return avg_curvature
        except:
            # Fallback if eigenvalue computation fails
            return 0.0
    
    def visualize_2d_slice(self, slice_dims=(0, 1), slice_coords=None, field='scalar',
                          cmap='viridis', save_path=None):
        """
        Visualize a 2D slice of the lattice.
        
        Parameters:
        -----------
        slice_dims : tuple
            The two dimensions to show in the slice
        slice_coords : list
            Coordinates for the other dimensions
        field : str
            Field to visualize ('scalar', 'metric_determinant', 'curvature')
        cmap : str
            Colormap to use
        save_path : str, optional
            Path to save the visualization
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.dim < 2:
            raise ValueError("Need at least 2 dimensions for 2D visualization")
            
        # Default slice coordinates
        if slice_coords is None:
            slice_coords = [self.size // 2] * (self.dim - 2)
            
        # Create coordinate arrays
        dim1, dim2 = slice_dims
        coords = [0] * self.dim
        
        # Create 2D array for the slice
        slice_data = np.zeros((self.size, self.size))
        
        # Counter for non-slice dimensions
        non_slice_idx = 0
        
        # Fill the slice data
        for i in range(self.size):
            coords[dim1] = i
            for j in range(self.size):
                coords[dim2] = j
                
                # Set coordinates for other dimensions
                for d in range(self.dim):
                    if d != dim1 and d != dim2:
                        coords[d] = slice_coords[non_slice_idx]
                        non_slice_idx = (non_slice_idx + 1) % len(slice_coords)
                
                # Get the field value at this point
                idx = self._lattice_index(tuple(coords))
                
                if field == 'scalar':
                    slice_data[i, j] = self.scalar_field[idx]
                elif field == 'metric_determinant':
                    slice_data[i, j] = np.linalg.det(self.metric_field[idx])
                elif field == 'curvature':
                    # This requires curvature to be computed first
                    if self.ricci_scalar is None:
                        self.compute_discrete_curvature()
                    slice_data[i, j] = self.ricci_scalar[idx] if hasattr(self.ricci_scalar, '__getitem__') else 0
                
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(slice_data, origin='lower', extent=[0, self.size, 0, self.size],
                     cmap=cmap, interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=f'{field.replace("_", " ").title()}')
        
        # Labels
        ax.set_xlabel(f'Dimension {dim2}')
        ax.set_ylabel(f'Dimension {dim1}')
        ax.set_title(f'2D Slice of {field.replace("_", " ").title()} Field')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class PathIntegralMonteCarlo:
    """
    Implements Monte Carlo methods for evaluating path integrals in quantum gravity.
    """
    
    def __init__(self, lattice, action_type='einstein_hilbert', planck_scale=1e19):
        """
        Initialize the path integral Monte Carlo simulator.
        
        Parameters:
        -----------
        lattice : DiscretizedSpacetime
            The discretized spacetime lattice
        action_type : str
            Type of action ('einstein_hilbert', 'higher_derivative', 'causal_sets')
        planck_scale : float
            Planck scale in GeV
        """
        self.lattice = lattice
        self.action_type = action_type
        self.planck_scale = planck_scale
        
        # Coupling constants
        self.g_newton = 1.0 / planck_scale**2
        
        # For higher derivative terms
        self.alpha = 0.1  # Coefficient for R²
        self.beta = 0.05  # Coefficient for R_μν R^μν
        
        # MC simulation parameters
        self.num_thermalization = 1000
        self.num_samples = 5000
        self.step_size = 0.01
        
        # Observable history
        self.history = {
            'action': [],
            'volume': [],
            'curvature': []
        }
        
    def _compute_action(self):
        """
        Compute the action for the current field configuration.
        
        Returns:
        --------
        float
            Action value
        """
        if self.action_type == 'einstein_hilbert':
            # Einstein-Hilbert action: ∫ R √(-g) d⁴x
            # Discretized version: Σ_i R_i √(-g_i) ΔV
            
            # Compute curvature if not already computed
            if self.lattice.ricci_scalar is None:
                self.lattice.compute_discrete_curvature()
                
            # Sum over all lattice points
            action = 0.0
            delta_V = self.lattice.lattice_spacing**self.lattice.dim
            
            for i in range(self.lattice.total_points):
                # Get scalar curvature and metric determinant
                R = self.lattice.ricci_scalar
                if hasattr(R, '__getitem__'):
                    R_i = R[i]
                else:
                    R_i = R  # If it's a scalar value
                    
                g_det = np.abs(np.linalg.det(self.lattice.metric_field[i]))
                g_sqrt = np.sqrt(g_det)
                
                # Add to action
                action += R_i * g_sqrt * delta_V
                
            return action * self.g_newton
            
        elif self.action_type == 'higher_derivative':
            # Higher derivative terms (simplified)
            # S = ∫ [R + α R² + β R_μν R^μν] √(-g) d⁴x
            
            # First compute the Einstein-Hilbert part
            eh_action = self._compute_action()
            
            # Simplified higher derivative terms
            # In a full implementation, we would compute R² and R_μν R^μν
            # Here we just use R² as an approximation
            if self.lattice.ricci_scalar is None:
                self.lattice.compute_discrete_curvature()
                
            R = self.lattice.ricci_scalar
            if hasattr(R, '__getitem__'):
                R_squared = np.sum(R**2) 
            else:
                R_squared = R**2 * self.lattice.total_points
                
            # Add higher derivative terms
            hd_action = eh_action + self.alpha * R_squared
            
            return hd_action
            
        else:
            # Default to Einstein-Hilbert
            return self._compute_action()
    
    def _metro_hastings_step(self):
        """
        Perform one Metropolis-Hastings update step.
        
        Returns:
        --------
        bool
            Whether the step was accepted
        """
        # Choose a random lattice point
        point_idx = np.random.randint(0, self.lattice.total_points)
        
        # Store the current field values
        old_metric = self.lattice.metric_field[point_idx].copy()
        old_scalar = self.lattice.scalar_field[point_idx]
        
        # Compute current action
        old_action = self._compute_action()
        
        # Propose a change to the fields
        # For metric: add small perturbation while preserving symmetry
        delta_metric = np.random.normal(0, self.step_size, (self.lattice.dim, self.lattice.dim))
        delta_metric = (delta_metric + delta_metric.T) / 2  # Make symmetric
        
        new_metric = old_metric + delta_metric
        
        # For scalar field: add small random change
        delta_scalar = np.random.normal(0, self.step_size)
        new_scalar = old_scalar + delta_scalar
        
        # Apply the proposed changes
        self.lattice.metric_field[point_idx] = new_metric
        self.lattice.scalar_field[point_idx] = new_scalar
        
        # Reset cached curvature
        self.lattice.ricci_scalar = None
        
        # Compute new action
        new_action = self._compute_action()
        
        # Metropolis-Hastings acceptance
        delta_S = new_action - old_action
        if delta_S < 0 or np.random.random() < np.exp(-delta_S):
            # Accept
            return True
        else:
            # Reject: restore old values
            self.lattice.metric_field[point_idx] = old_metric
            self.lattice.scalar_field[point_idx] = old_scalar
            self.lattice.ricci_scalar = None
            return False
    
    def run_simulation(self, verbose=True):
        """
        Run the Monte Carlo simulation.
        
        Parameters:
        -----------
        verbose : bool
            Whether to show progress information
            
        Returns:
        --------
        dict
            Simulation results
        """
        # Record initial values
        current_action = self._compute_action()
        self.history['action'].append(current_action)
        
        # Thermalization steps
        if verbose:
            print(f"Performing {self.num_thermalization} thermalization steps...")
            
        thermalization_range = range(self.num_thermalization)
        if verbose:
            thermalization_range = tqdm(thermalization_range)
            
        for _ in thermalization_range:
            self._metro_hastings_step()
        
        # Production steps
        if verbose:
            print(f"Performing {self.num_samples} production steps...")
            
        samples_range = range(self.num_samples)
        if verbose:
            samples_range = tqdm(samples_range)
            
        accepted = 0
        for _ in samples_range:
            # Perform MC step
            accepted += self._metro_hastings_step()
            
            # Compute observables
            action = self._compute_action()
            
            # Compute average volume
            volume = 0.0
            for i in range(self.lattice.total_points):
                g_det = np.abs(np.linalg.det(self.lattice.metric_field[i]))
                volume += np.sqrt(g_det) * self.lattice.lattice_spacing**self.lattice.dim
                
            # Get scalar curvature
            if self.lattice.ricci_scalar is None:
                curvature = self.lattice.compute_discrete_curvature()
            else:
                curvature = self.lattice.ricci_scalar
                if hasattr(curvature, '__len__'):
                    curvature = np.mean(curvature)
            
            # Record history
            self.history['action'].append(action)
            self.history['volume'].append(volume)
            self.history['curvature'].append(curvature)
        
        # Calculate results
        acceptance_rate = accepted / self.num_samples
        
        # Remove thermalization steps from history for analysis
        for key in self.history:
            self.history[key] = self.history[key][-self.num_samples:]
        
        # Compute averages and errors
        results = {
            'acceptance_rate': acceptance_rate,
            'avg_action': np.mean(self.history['action']),
            'err_action': np.std(self.history['action']) / np.sqrt(self.num_samples),
            'avg_volume': np.mean(self.history['volume']),
            'err_volume': np.std(self.history['volume']) / np.sqrt(self.num_samples),
            'avg_curvature': np.mean(self.history['curvature']),
            'err_curvature': np.std(self.history['curvature']) / np.sqrt(self.num_samples)
        }
        
        if verbose:
            print(f"\nResults:")
            print(f"  Acceptance rate: {acceptance_rate:.4f}")
            print(f"  Average action: {results['avg_action']:.6f} ± {results['err_action']:.6f}")
            print(f"  Average volume: {results['avg_volume']:.6f} ± {results['err_volume']:.6f}")
            print(f"  Average curvature: {results['avg_curvature']:.6f} ± {results['err_curvature']:.6f}")
        
        return results
    
    def plot_history(self, save_path=None):
        """
        Plot the history of observables.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot action
        axs[0].plot(self.history['action'], 'b-')
        axs[0].set_ylabel('Action')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot volume
        axs[1].plot(self.history['volume'], 'r-')
        axs[1].set_ylabel('Volume')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot curvature
        axs[2].plot(self.history['curvature'], 'g-')
        axs[2].set_ylabel('Curvature')
        axs[2].set_xlabel('MC Steps')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class TensorNetworkStates:
    """
    Implements tensor network states for quantum gravity simulations.
    """
    
    def __init__(self, dim=4, bond_dim=2, network_size=4):
        """
        Initialize a tensor network state representation.
        
        Parameters:
        -----------
        dim : int
            Spacetime dimension
        bond_dim : int
            Bond dimension for the tensor network
        network_size : int
            Size of the tensor network
        """
        self.dim = dim
        self.bond_dim = bond_dim
        self.network_size = network_size
        
        # Number of nodes in the network
        self.num_nodes = network_size**dim
        
        # Initialize tensors (random)
        self.tensors = {}
        for i in range(self.num_nodes):
            # Each tensor has 2*dim indices (one for each direction)
            tensor_shape = (bond_dim,) * (2 * dim)
            self.tensors[i] = np.random.normal(0, 0.1, tensor_shape)
            
        # Define network connectivity based on geometry
        self._define_network_connectivity()
    
    def _define_network_connectivity(self):
        """
        Define the connectivity of the tensor network.
        """
        self.connections = {}
        
        # Loop over all nodes
        for i in range(self.num_nodes):
            # Convert to coordinates
            coords = self._index_to_coords(i)
            
            # For each dimension
            for d in range(self.dim):
                # Forward connection
                fwd_coords = list(coords)
                fwd_coords[d] = (coords[d] + 1) % self.network_size
                fwd_idx = self._coords_to_index(tuple(fwd_coords))
                
                # Store connections
                if i not in self.connections:
                    self.connections[i] = {}
                if fwd_idx not in self.connections:
                    self.connections[fwd_idx] = {}
                    
                # Connect d-th output of node i to d-th input of forward node
                self.connections[i][d] = (fwd_idx, d + self.dim)
                # The reverse connection
                self.connections[fwd_idx][d + self.dim] = (i, d)
    
    def _index_to_coords(self, index):
        """Convert flat index to coordinates."""
        coords = []
        for i in range(self.dim):
            coords.append(index % self.network_size)
            index //= self.network_size
        return tuple(coords)
    
    def _coords_to_index(self, coords):
        """Convert coordinates to flat index."""
        index = 0
        for i, coord in enumerate(coords):
            index += coord * (self.network_size ** i)
        return index
    
    def contract_network(self, contraction_order=None):
        """
        Approximate contraction of the tensor network.
        
        Parameters:
        -----------
        contraction_order : list, optional
            Order of tensor contractions
            
        Returns:
        --------
        float
            Estimated network contraction value
        """
        # This is a simplified approximation
        # Full tensor network contraction is computationally intensive
        
        # If no contraction order specified, use a simple order
        if contraction_order is None:
            contraction_order = list(range(self.num_nodes))
        
        # For demonstration only - this is not an actual tensor network contraction
        # We just return an estimate based on tensor norms
        result = 1.0
        for i in contraction_order:
            tensor_norm = np.linalg.norm(self.tensors[i])
            result *= tensor_norm
            
        return result
    
    def optimize_tensors(self, target_function, learning_rate=0.01, steps=100, verbose=True):
        """
        Optimize tensors to minimize/maximize a target function.
        
        Parameters:
        -----------
        target_function : callable
            Function that takes the tensor network and returns a value to minimize
        learning_rate : float
            Learning rate for optimization
        steps : int
            Number of optimization steps
        verbose : bool
            Whether to show progress information
            
        Returns:
        --------
        list
            History of target function values
        """
        history = []
        
        step_range = range(steps)
        if verbose:
            step_range = tqdm(step_range)
            
        for step in step_range:
            # Evaluate target function
            current_value = target_function(self)
            history.append(current_value)
            
            # Compute gradients (finite difference approximation)
            gradients = {}
            
            for node_idx in self.tensors:
                grad = np.zeros_like(self.tensors[node_idx])
                
                # For each element in the tensor
                it = np.nditer(self.tensors[node_idx], flags=['multi_index'])
                for _ in it:
                    multi_idx = it.multi_index
                    
                    # Save original value
                    orig_val = self.tensors[node_idx][multi_idx]
                    
                    # Add small perturbation
                    self.tensors[node_idx][multi_idx] += 0.01
                    perturbed_value = target_function(self)
                    
                    # Compute gradient
                    grad[multi_idx] = (perturbed_value - current_value) / 0.01
                    
                    # Restore original value
                    self.tensors[node_idx][multi_idx] = orig_val
                
                gradients[node_idx] = grad
            
            # Update tensors
            for node_idx in self.tensors:
                self.tensors[node_idx] -= learning_rate * gradients[node_idx]
                
        return history
    
    def visualize_2d_slice(self, slice_dims=(0, 1), slice_coords=None, save_path=None):
        """
        Visualize a 2D slice of the tensor network.
        
        Parameters:
        -----------
        slice_dims : tuple
            The two dimensions to show in the slice
        slice_coords : list
            Coordinates for the other dimensions
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.dim < 2:
            raise ValueError("Need at least 2 dimensions for 2D visualization")
            
        # Default slice coordinates
        if slice_coords is None:
            slice_coords = [self.network_size // 2] * (self.dim - 2)
            
        # Create a graph for visualization
        G = nx.Graph()
        
        # Create coordinate arrays
        dim1, dim2 = slice_dims
        all_coords = [0] * self.dim
        
        # Add nodes for the 2D slice
        pos = {}
        node_colors = []
        
        for i in range(self.network_size):
            all_coords[dim1] = i
            for j in range(self.network_size):
                all_coords[dim2] = j
                
                # Set coordinates for other dimensions
                for d in range(self.dim):
                    if d != dim1 and d != dim2:
                        idx = slice_dims.index(d) if d in slice_dims else d - 2
                        all_coords[d] = slice_coords[idx] if idx < len(slice_coords) else 0
                
                # Get node index
                node_idx = self._coords_to_index(tuple(all_coords))
                
                # Add node
                G.add_node(node_idx)
                
                # Set position
                pos[node_idx] = (i, j)
                
                # Set color based on tensor norm
                tensor_norm = np.linalg.norm(self.tensors[node_idx])
                node_colors.append(tensor_norm)
        
        # Add edges based on connections
        for node in G.nodes():
            for d in range(self.dim):
                if d in [dim1, dim2]:  # Only add edges in the slice dimensions
                    if d in self.connections.get(node, {}):
                        target, _ = self.connections[node][d]
                        if target in G.nodes():
                            G.add_edge(node, target)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the graph
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=100,
               node_color=node_colors, cmap='viridis', 
               edge_color='gray', width=1.0, alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(node_colors)
        plt.colorbar(sm, ax=ax, label='Tensor Norm')
        
        ax.set_title('2D Slice of Tensor Network')
        ax.set_xlabel(f'Dimension {dim1}')
        ax.set_ylabel(f'Dimension {dim2}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # Example usage
    print("Testing DiscretizedSpacetime...")
    spacetime = DiscretizedSpacetime(dim=4, size=10)
    
    # Set a scalar field
    def scalar_function(coords):
        r = np.linalg.norm(coords)
        return np.exp(-r**2)
    
    spacetime.set_scalar_field(scalar_function)
    
    # Compute curvature
    print(f"Average curvature: {spacetime.compute_discrete_curvature()}")
    
    # Set up a Monte Carlo simulation
    print("\nSetting up Path Integral Monte Carlo...")
    mc_sim = PathIntegralMonteCarlo(spacetime, action_type='einstein_hilbert')
    
    # Run a short simulation for testing
    mc_sim.num_thermalization = 10
    mc_sim.num_samples = 20
    results = mc_sim.run_simulation()
    
    print("\nTesting TensorNetworkStates...")
    tensor_net = TensorNetworkStates(dim=2, bond_dim=2, network_size=4)
    
    # Sample target function
    def simple_target(tn):
        return tn.contract_network()
    
    print("Optimizing tensor network...")
    history = tensor_net.optimize_tensors(simple_target, steps=10, verbose=True)
    print(f"Initial value: {history[0]}, Final value: {history[-1]}") 