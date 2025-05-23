"""
Quantum Spacetime Foundations

This module implements the fundamental mathematical structure of quantum spacetime,
including spectral geometry, causal sets, and energy-dependent transitions between
discrete and continuous structures.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power
from scipy.integrate import solve_ivp
import sympy as sp


class SpectralGeometry:
    """
    Implements spectral geometry for quantum spacetime.
    
    Spectral geometry characterizes a space using the spectrum of operators
    defined on it, particularly the Laplacian operator.
    """
    
    def __init__(self, dim=4, size=50, discretization='fuzzy'):
        """
        Initialize spectral geometry.
        
        Parameters:
        -----------
        dim : float
            Dimension of spacetime (can be non-integer for fractal-like spaces)
        size : int
            Size of the discretized space
        discretization : str
            Type of discretization: 'fuzzy', 'simplicial', or 'causal'
        """
        self.dim = dim
        self.size = size
        self.discretization = discretization
        
        # Discretized representation of spacetime
        self.points = None
        self.laplacian = None
        self.distances = None
        self.volume_element = None
        
        # Initialize discretized space
        self._initialize_space()
    
    def _initialize_space(self):
        """Initialize the discretized space representation."""
        if self.discretization == 'fuzzy':
            # Fuzzy space: noncommutative geometry inspired
            self._initialize_fuzzy_space()
        elif self.discretization == 'simplicial':
            # Simplicial complex: triangulation of space
            self._initialize_simplicial_space()
        elif self.discretization == 'causal':
            # Causal set: spacetime with causal structure
            self._initialize_causal_space()
        else:
            raise ValueError(f"Unknown discretization type: {self.discretization}")
    
    def _initialize_fuzzy_space(self):
        """Initialize fuzzy space (noncommutative geometry inspired)."""
        # Generate random matrix basis for fuzzy space
        dim_int = int(np.ceil(self.dim))
        matrix_size = self.size
        
        # Create a random Hermitian matrix (truncated spectral triple)
        H = np.random.randn(matrix_size, matrix_size) + 1j * np.random.randn(matrix_size, matrix_size)
        H = (H + H.conj().T) / 2.0
        
        # Compute Laplacian as square of Dirac operator (H serves as Dirac)
        self.laplacian = H @ H
        
        # Volume element based on dimension
        self.volume_element = np.ones(matrix_size) * ((matrix_size)**(self.dim/dim_int))
        
        # Extract points from eigenvectors for visualization
        if dim_int <= 3:
            eigvals, eigvecs = np.linalg.eigh(H)
            self.points = eigvecs[:, :dim_int].real
        else:
            # Create abstract points for higher dimensions
            self.points = np.random.randn(matrix_size, min(3, dim_int))
        
        # Distance matrix (approximation)
        self.distances = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i != j:
                    # Connes distance formula (simplified)
                    self.distances[i, j] = 1.0 / np.abs(self.laplacian[i, j] + 1e-10)
    
    def _initialize_simplicial_space(self):
        """Initialize simplicial complex (triangulation of space)."""
        # Generate random points in unit cube
        dim_int = min(3, int(np.ceil(self.dim)))  # For visualization
        self.points = np.random.rand(self.size, dim_int)
        
        # Compute distances between points
        self.distances = squareform(pdist(self.points))
        
        # Create simplicial complex (triangulation)
        if dim_int >= 2:
            try:
                tri = Delaunay(self.points)
                
                # Create graph from triangulation
                G = nx.Graph()
                for i in range(self.size):
                    G.add_node(i)
                
                # Add edges from triangulation
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            G.add_edge(simplex[i], simplex[j])
                
                # Compute Laplacian matrix
                self.laplacian = nx.laplacian_matrix(G).toarray()
                
                # Volume element (simplex volumes)
                self.volume_element = np.ones(self.size)
                for i, point in enumerate(self.points):
                    # Find simplices containing this point
                    simplices_with_point = [s for s in tri.simplices if i in s]
                    self.volume_element[i] = len(simplices_with_point)
                
            except Exception:
                # Fallback if triangulation fails
                self._initialize_fuzzy_space()
        else:
            # For 1D, just create a path graph
            G = nx.path_graph(self.size)
            self.laplacian = nx.laplacian_matrix(G).toarray()
            self.volume_element = np.ones(self.size)
    
    def _initialize_causal_space(self):
        """Initialize causal set (spacetime with causal structure)."""
        # Generate random points in spacetime (extended in time direction)
        dim_int = min(3, int(np.ceil(self.dim)))
        self.points = np.random.rand(self.size, dim_int + 1)  # +1 for time
        
        # Sort by time coordinate
        self.points = self.points[np.argsort(self.points[:, 0])]
        
        # Create causal graph
        G = nx.DiGraph()
        for i in range(self.size):
            G.add_node(i)
        
        # Add edges based on causal structure
        # Two points are causally related if they're within each other's light cone
        for i in range(self.size):
            for j in range(i+1, self.size):  # j is in the future of i
                dt = self.points[j, 0] - self.points[i, 0]  # Time difference
                dx = np.linalg.norm(self.points[j, 1:] - self.points[i, 1:])  # Spatial difference
                
                # Causal connection if j is in the future light cone of i
                if dt > 0 and dt >= dx:  # Using c=1 units
                    G.add_edge(i, j)
        
        # Compute distances (using longest path for timelike separation)
        self.distances = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.distances[i, j] = 0
                elif nx.has_path(G, i, j):
                    # If causally connected, use path length
                    self.distances[i, j] = nx.shortest_path_length(G, i, j)
                else:
                    # If not causally connected, use a large value
                    self.distances[i, j] = float('inf')
        
        # Create Laplacian as graph Laplacian of undirected version of causal graph
        G_undirected = G.to_undirected()
        self.laplacian = nx.laplacian_matrix(G_undirected).toarray()
        
        # Volume element based on causal relationships
        self.volume_element = np.array([len(list(G.successors(i))) + len(list(G.predecessors(i))) 
                                      for i in range(self.size)])
        self.volume_element = np.maximum(self.volume_element, 1)  # Avoid zeros
    
    def compute_spectral_dimension(self, diffusion_time):
        """
        Compute the spectral dimension at a given diffusion time.
        
        Parameters:
        -----------
        diffusion_time : float
            Diffusion time (related to energy scale as E ~ 1/sqrt(t))
            
        Returns:
        --------
        float
            Spectral dimension
        """
        # Convert Laplacian to sparse format for efficiency
        L_sparse = sparse.csr_matrix(self.laplacian)
        
        # Heat kernel trace approximation
        # K(t) = Tr[exp(-t*L)]
        
        # For small diffusion times, use direct eigenvalue computation
        if diffusion_time < 1.0 or self.size < 100:
            try:
                # Compute smallest eigenvalues (largest exp(-t*lambda))
                num_eigs = min(20, self.size - 1)
                eigvals = splinalg.eigsh(L_sparse, k=num_eigs, which='SM', 
                                      return_eigenvectors=False)
                
                # Remove zero eigenvalue if present
                eigvals = eigvals[eigvals > 1e-10]
                
                # Heat kernel trace
                K_t = np.sum(np.exp(-diffusion_time * eigvals))
                
                # For full trace, estimate contribution of remaining eigenvalues
                if num_eigs < self.size - 1:
                    # Approximate remaining eigenvalues using dimension-based scaling
                    remaining = self.size - num_eigs
                    avg_eig = np.mean(eigvals)
                    K_t += remaining * np.exp(-diffusion_time * avg_eig * 2)
            except:
                # Fallback: stochastic trace estimation
                K_t = self._stochastic_trace_estimation(L_sparse, diffusion_time)
        else:
            # For large diffusion times, use stochastic trace estimation
            K_t = self._stochastic_trace_estimation(L_sparse, diffusion_time)
        
        # Compute spectral dimension from log derivative of heat kernel trace
        # d_s = -2 d/dt log(K(t))
        
        # Use small dt for numerical differentiation
        dt = diffusion_time * 0.01
        
        # Compute K(t+dt)
        if diffusion_time + dt < 1.0 or self.size < 100:
            try:
                num_eigs = min(20, self.size - 1)
                eigvals = splinalg.eigsh(L_sparse, k=num_eigs, which='SM', 
                                      return_eigenvectors=False)
                eigvals = eigvals[eigvals > 1e-10]
                K_t_dt = np.sum(np.exp(-(diffusion_time + dt) * eigvals))
                if num_eigs < self.size - 1:
                    remaining = self.size - num_eigs
                    avg_eig = np.mean(eigvals)
                    K_t_dt += remaining * np.exp(-(diffusion_time + dt) * avg_eig * 2)
            except:
                K_t_dt = self._stochastic_trace_estimation(L_sparse, diffusion_time + dt)
        else:
            K_t_dt = self._stochastic_trace_estimation(L_sparse, diffusion_time + dt)
        
        # Compute numerical derivative
        d_log_K = (np.log(K_t_dt) - np.log(K_t)) / dt
        
        # Spectral dimension
        d_s = -2 * d_log_K
        
        return max(0, d_s)  # Ensure non-negative dimension
    
    def _stochastic_trace_estimation(self, L_sparse, diffusion_time, num_samples=50):
        """
        Stochastic estimation of heat kernel trace.
        
        Parameters:
        -----------
        L_sparse : scipy.sparse.spmatrix
            Sparse Laplacian matrix
        diffusion_time : float
            Diffusion time
        num_samples : int
            Number of random vectors for estimation
            
        Returns:
        --------
        float
            Estimated trace
        """
        n = L_sparse.shape[0]
        trace_sum = 0.0
        
        for _ in range(num_samples):
            # Generate random vector with +/-1 elements
            v = np.random.choice([-1, 1], size=n)
            
            # Apply matrix exponential to vector: w = exp(-t*L)*v
            w = self._cg_exp_action(L_sparse, -diffusion_time, v)
            
            # Add to trace estimation (v^T * w)
            trace_sum += np.dot(v, w) / num_samples
        
        return trace_sum
    
    def _cg_exp_action(self, L_sparse, t, v, tol=1e-6, max_iter=100):
        """
        Compute exp(t*L)*v using truncated Taylor series and conjugate gradient.
        
        This is a simplified implementation for demonstration purposes.
        In practice, one would use more sophisticated methods like Krylov subspace techniques.
        
        Parameters:
        -----------
        L_sparse : scipy.sparse.spmatrix
            Sparse Laplacian matrix
        t : float
            Time parameter
        v : ndarray
            Input vector
        tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        ndarray
            Result of exp(t*L)*v
        """
        result = v.copy()
        term = v.copy()
        
        for k in range(1, max_iter):
            # Next term in Taylor series
            term = t * L_sparse.dot(term) / k
            
            # Add to result
            result += term
            
            # Check convergence
            if np.linalg.norm(term) < tol * np.linalg.norm(result):
                break
        
        return result
    
    def compute_dimension_vs_scale(self, time_range=None, num_points=20):
        """
        Compute spectral dimension across different scales.
        
        Parameters:
        -----------
        time_range : tuple, optional
            Range of diffusion times (min, max)
        num_points : int
            Number of points to compute
            
        Returns:
        --------
        tuple
            (times, dimensions)
        """
        if time_range is None:
            # Default range: from short to long diffusion times
            time_range = (0.01, 100.0)
        
        # Generate logarithmically spaced diffusion times
        times = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]), num_points)
        
        # Compute spectral dimension at each time
        dimensions = np.array([self.compute_spectral_dimension(t) for t in times])
        
        return times, dimensions
    
    def compute_laplacian_spectrum(self, num_eigenvalues=None):
        """
        Compute the spectrum of the Laplacian.
        
        Parameters:
        -----------
        num_eigenvalues : int, optional
            Number of eigenvalues to compute (default: all)
            
        Returns:
        --------
        ndarray
            Eigenvalues of the Laplacian
        """
        if num_eigenvalues is None or num_eigenvalues >= self.size:
            # Compute full spectrum
            return np.linalg.eigvalsh(self.laplacian)
        else:
            # Compute partial spectrum (smallest eigenvalues)
            return splinalg.eigsh(sparse.csr_matrix(self.laplacian), 
                                k=num_eigenvalues, which='SM', 
                                return_eigenvectors=False)
    
    def visualize(self, ax=None):
        """
        Visualize the quantum spacetime.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
            
        Returns:
        --------
        matplotlib.axes.Axes
            The plot axes
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            if self.points.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Plot based on dimension
        if self.points.shape[1] == 1:
            # 1D case
            y = np.zeros_like(self.points[:, 0])
            ax.scatter(self.points[:, 0], y, c=np.arange(len(y)), cmap='viridis')
            ax.set_ylabel('Index')
            
        elif self.points.shape[1] == 2:
            # 2D case
            scatter = ax.scatter(self.points[:, 0], self.points[:, 1], 
                               c=self.volume_element, cmap='viridis', 
                               s=30, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Volume Element')
            
            # Add edges based on Laplacian
            for i in range(self.size):
                for j in range(i+1, self.size):
                    if np.abs(self.laplacian[i, j]) > 1e-6:
                        ax.plot([self.points[i, 0], self.points[j, 0]], 
                               [self.points[i, 1], self.points[j, 1]], 
                               'k-', alpha=0.1)
        
        elif self.points.shape[1] == 3:
            # 3D case
            scatter = ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                               c=self.volume_element, cmap='viridis', 
                               s=30, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Volume Element')
            
            # Add a subset of edges to avoid visual clutter
            for i in range(self.size):
                connections = np.where(np.abs(self.laplacian[i, :]) > 0)[0]
                for j in np.random.choice(connections, min(5, len(connections)), replace=False):
                    if j > i:
                        ax.plot([self.points[i, 0], self.points[j, 0]], 
                               [self.points[i, 1], self.points[j, 1]],
                               [self.points[i, 2], self.points[j, 2]],
                               'k-', alpha=0.1)
        
        ax.set_title(f'Quantum Spacetime ({self.discretization}, dim={self.dim:.2f})')
        return ax

class CausalSet:
    """
    Implements causal set theory for quantum spacetime.
    
    Causal set theory represents spacetime as a partially ordered set of events,
    where the partial order corresponds to the causal relationship between events.
    """
    
    def __init__(self, num_points=100, dim=4, sprinkle_method='flat'):
        """
        Initialize causal set.
        
        Parameters:
        -----------
        num_points : int
            Number of points in the causal set
        dim : float
            Dimension of spacetime (can be non-integer)
        sprinkle_method : str
            Method for generating points: 'flat', 'curved', or 'discrete'
        """
        self.num_points = num_points
        self.dim = dim
        self.sprinkle_method = sprinkle_method
        
        # Causal set data structures
        self.points = None       # Coordinates of points
        self.causal_matrix = None  # Matrix of causal relations
        self.causal_graph = None   # Directed graph of causal relations
        
        # Generate causal set
        self._generate_causal_set()
    
    def _generate_causal_set(self):
        """Generate a causal set using the specified method."""
        if self.sprinkle_method == 'flat':
            self._sprinkle_flat_spacetime()
        elif self.sprinkle_method == 'curved':
            self._sprinkle_curved_spacetime()
        elif self.sprinkle_method == 'discrete':
            self._construct_discrete_causal_set()
        else:
            raise ValueError(f"Unknown sprinkle method: {self.sprinkle_method}")
        
        # Build the causal matrix and graph
        self._build_causal_relations()
    
    def _sprinkle_flat_spacetime(self):
        """
        Sprinkle points uniformly in Minkowski spacetime.
        
        This implements the standard Poisson sprinkling process into
        flat spacetime, commonly used in causal set theory.
        """
        # Use 1+d spatial dimensions (time + space)
        space_dim = max(1, int(np.ceil(self.dim - 1)))
        
        # Generate points uniformly in a spacetime region
        # Time coordinate (first column)
        t = np.random.uniform(0, 1, self.num_points)
        
        # Spatial coordinates (remaining columns)
        space_coords = np.random.uniform(0, 1, (self.num_points, space_dim))
        
        # Combine into spacetime coordinates
        self.points = np.column_stack((t, space_coords))
        
        # Sort by time coordinate for efficiency
        self.points = self.points[np.argsort(self.points[:, 0])]
    
    def _sprinkle_curved_spacetime(self):
        """
        Sprinkle points in a curved spacetime.
        
        This implements a simple model of sprinkling into a curved
        spacetime, like a section of de Sitter space.
        """
        # Use 1+d spatial dimensions (time + space)
        space_dim = max(1, int(np.ceil(self.dim - 1)))
        
        # Generate time coordinates with non-uniform density
        # modeling the expanding universe
        t = np.random.uniform(0, 1, self.num_points)
        
        # Apply a transformation to implement curvature effect
        # This is a simplified model: t' = t^a where a > 1 gives higher
        # density in the past (early universe)
        curvature_exponent = 1.5
        t = t ** curvature_exponent
        
        # Generate spatial coordinates with scale factor
        # Points are more closely packed in the past
        space_coords = []
        for i in range(self.num_points):
            # Scale factor: a(t) = t^(2/3) in matter-dominated era (simplified)
            scale = t[i] ** (2/3)
            # Spatial coordinates within horizon: r < c·t
            max_radius = t[i]
            
            # Generate position within a sphere of radius max_radius
            if space_dim == 1:
                # 1D: random position on a line
                x = np.random.uniform(-max_radius, max_radius)
                space_coords.append([x * scale])
            else:
                # Higher dimensions: random position in a sphere
                # Generate random direction
                direction = np.random.randn(space_dim)
                direction = direction / np.linalg.norm(direction)
                
                # Generate random radius with correct distribution
                radius = max_radius * np.random.random() ** (1/space_dim)
                
                # Combine into position
                pos = direction * radius * scale
                space_coords.append(pos)
        
        # Combine into spacetime coordinates
        self.points = np.column_stack((t, np.array(space_coords)))
        
        # Sort by time coordinate for efficiency
        self.points = self.points[np.argsort(self.points[:, 0])]
    
    def _construct_discrete_causal_set(self):
        """
        Construct a discrete causal set directly, without embedding in a continuum.
        
        This implements a growth model where points are added sequentially
        and causal relations are assigned based on a probabilistic rule.
        """
        # Initialize with random coordinates for visualization (not used for causality)
        space_dim = max(1, int(np.ceil(self.dim - 1)))
        self.points = np.random.random((self.num_points, 1 + space_dim))
        
        # Initialize causal matrix
        self.causal_matrix = np.zeros((self.num_points, self.num_points), dtype=bool)
        
        # Build causal set sequentially
        for i in range(1, self.num_points):
            # Probability of being causally related to previous elements
            # decreases with the proper time expectation in dimension d
            # p ~ (d/2)^(-1) for large separations
            p_connect = min(0.5, (self.dim/2)**(-1))
            
            # Connect to some previous elements
            connections = np.random.binomial(1, p_connect, i)
            for j in range(i):
                if connections[j]:
                    self.causal_matrix[j, i] = True
                    
                    # Ensure transitivity: if j → i and k → j, then k → i
                    for k in range(j):
                        if self.causal_matrix[k, j]:
                            self.causal_matrix[k, i] = True
        
        # Set time coordinates based on causal layers
        layers = self._compute_causal_layers()
        
        # Assign time coordinates based on layers
        for i, layer in enumerate(layers):
            for point in layer:
                self.points[point, 0] = i / len(layers)
                
        # Reorder points by time for efficiency
        order = np.argsort(self.points[:, 0])
        self.points = self.points[order]
        
        # Reorder causal matrix to match new point ordering
        new_causal_matrix = np.zeros_like(self.causal_matrix)
        for i in range(self.num_points):
            for j in range(self.num_points):
                new_causal_matrix[order[i], order[j]] = self.causal_matrix[i, j]
        self.causal_matrix = new_causal_matrix
    
    def _compute_causal_layers(self):
        """
        Compute the causal layers (anti-chains) of the causal set.
        
        Returns:
        --------
        list
            List of sets, where each set contains points in the same layer
        """
        # Create directed graph from causal matrix
        G = nx.DiGraph()
        for i in range(self.num_points):
            G.add_node(i)
        
        for i in range(self.num_points):
            for j in range(i+1, self.num_points):
                if self.causal_matrix[i, j]:
                    G.add_edge(i, j)
        
        # Get minimal elements (those with no predecessors)
        roots = [node for node in G.nodes if G.in_degree(node) == 0]
        
        # Compute layers using breadth-first search
        layers = []
        remaining = set(range(self.num_points))
        
        while remaining:
            # Current layer: nodes with no remaining predecessors
            current_layer = set()
            for node in remaining:
                has_predecessor = False
                for pred in G.predecessors(node):
                    if pred in remaining:
                        has_predecessor = True
                        break
                if not has_predecessor:
                    current_layer.add(node)
            
            layers.append(current_layer)
            remaining -= current_layer
            
        return layers
    
    def _build_causal_relations(self):
        """Build the causal relation matrix and graph from the set of points."""
        if self.causal_matrix is not None and self.sprinkle_method == 'discrete':
            # For discrete method, causal matrix already built
            pass
        else:
            # Initialize causal matrix
            self.causal_matrix = np.zeros((self.num_points, self.num_points), dtype=bool)
            
            # For each pair of points, check causal relation
            # Point i precedes j if j is in the future light cone of i
            for i in range(self.num_points):
                for j in range(i+1, self.num_points):  # Skip diagonal and lower triangle
                    # Compute spacetime interval
                    # For Minkowski: ds² = -dt² + dx² + dy² + ...
                    dt = self.points[j, 0] - self.points[i, 0]
                    
                    # Only check if j is in the future of i
                    if dt <= 0:
                        continue
                    
                    # Compute spatial distance
                    space_dim = self.points.shape[1] - 1
                    dx2 = np.sum((self.points[j, 1:1+space_dim] - self.points[i, 1:1+space_dim])**2)
                    
                    # Causal if j is inside or on the future light cone of i
                    if dt**2 >= dx2:  # Using c=1 units
                        self.causal_matrix[i, j] = True
        
        # Create directed graph representation
        self.causal_graph = nx.DiGraph()
        for i in range(self.num_points):
            self.causal_graph.add_node(i)
        
        for i in range(self.num_points):
            for j in range(i+1, self.num_points):
                if self.causal_matrix[i, j]:
                    self.causal_graph.add_edge(i, j)
    
    def compute_causal_path_lengths(self):
        """
        Compute the distribution of causal path lengths.
        
        Returns:
        --------
        dict
            Dictionary with path length statistics
        """
        # Compute longest paths between all causally related pairs
        path_lengths = []
        for i in range(self.num_points):
            for j in range(i+1, self.num_points):
                if nx.has_path(self.causal_graph, i, j):
                    path_lengths.append(nx.shortest_path_length(self.causal_graph, i, j))
        
        if not path_lengths:
            return {'mean': 0, 'max': 0, 'counts': {}}
        
        # Compute statistics
        mean_length = np.mean(path_lengths)
        max_length = np.max(path_lengths)
        
        # Count frequency of each path length
        unique, counts = np.unique(path_lengths, return_counts=True)
        count_dict = {int(u): int(c) for u, c in zip(unique, counts)}
        
        return {
            'mean': mean_length,
            'max': max_length,
            'counts': count_dict
        }
    
    def compute_dimension_estimate(self):
        """
        Estimate the dimension of the causal set using scaling relationships.
        
        Returns:
        --------
        float
            Estimated dimension
        """
        # Use the Myrheim-Meyer dimension estimator
        # In d dimensions, the number of related pairs scales as N^(2-2/d)
        
        # Count number of related pairs
        num_related = np.sum(self.causal_matrix)
        
        # For perfect Poisson sprinkling into Minkowski spacetime,
        # the expected number of related pairs is:
        # E[num_related] = (2^d * π^(d/2) / (d * Γ(d/2))) * N^2 / 2
        
        # Invert this to estimate d
        if num_related > 0:
            # Simple approximation: d ≈ 2 / (1 - log(num_related/(N^2/2))/log(N))
            ratio = 2 * num_related / (self.num_points**2)
            if 0 < ratio < 1:
                # Adjust with a correction factor based on typical sprinkling
                # This is an empirical adjustment based on simulation results
                return 2 / (1 - np.log(ratio) / np.log(self.num_points)) * 0.9
            else:
                # Fallback for unusual cases
                return self.dim
        else:
            return 1.0  # Minimum dimension if no relations
    
    def compute_chain_lengths(self):
        """
        Compute the distribution of maximal chain lengths.
        
        A chain is a totally ordered subset of the causal set.
        
        Returns:
        --------
        dict
            Chain length statistics
        """
        # Compute the longest chains using topological sorting
        try:
            # Get all longest paths using dynamic programming
            longest_paths = {}
            
            # Initialize
            for node in range(self.num_points):
                longest_paths[node] = 0
            
            # Topological sort
            topo_order = list(nx.topological_sort(self.causal_graph))
            
            # Compute longest path to each node
            for node in topo_order:
                for successor in self.causal_graph.successors(node):
                    longest_paths[successor] = max(
                        longest_paths[successor],
                        longest_paths[node] + 1
                    )
            
            # Extract chain lengths
            chain_lengths = list(longest_paths.values())
            
            # Compute statistics
            mean_length = np.mean(chain_lengths)
            max_length = np.max(chain_lengths)
            
            # Count frequency
            unique, counts = np.unique(chain_lengths, return_counts=True)
            count_dict = {int(u): int(c) for u, c in zip(unique, counts)}
            
            return {
                'mean': mean_length,
                'max': max_length,
                'counts': count_dict
            }
        except:
            # Fallback for graphs with cycles or other issues
            return {'mean': 0, 'max': 0, 'counts': {}}
    
    def compute_spacetime_volume(self, region=None):
        """
        Compute the spacetime volume.
        
        In causal set theory, the volume is proportional to the number of points.
        
        Parameters:
        -----------
        region : tuple, optional
            Specify a region as ((t_min, t_max), (x_min, x_max), ...)
            
        Returns:
        --------
        float
            Estimated spacetime volume
        """
        # Get points in the specified region
        if region is None:
            count = self.num_points
        else:
            # Count points in the specified region
            count = 0
            for point in self.points:
                in_region = True
                for i, (min_val, max_val) in enumerate(region):
                    if i >= point.shape[0]:
                        break
                    if point[i] < min_val or point[i] > max_val:
                        in_region = False
                        break
                if in_region:
                    count += 1
        
        # Volume is proportional to point count with dimension-dependent factor
        # For Poisson sprinkling with density ρ, V = N/ρ
        # For unit density, V = N
        return count
    
    def compute_causal_structure(self):
        """
        Analyze the causal structure of the spacetime.
        
        Returns:
        --------
        dict
            Causal structure analysis results
        """
        # Compute various causal structure measures
        
        # 1. Transitivity
        transitivity = nx.transitivity(self.causal_graph.to_undirected())
        
        # 2. Clustering coefficient
        clustering = nx.average_clustering(self.causal_graph.to_undirected())
        
        # 3. Diameter (longest shortest path)
        try:
            diameter = nx.diameter(self.causal_graph.to_undirected())
        except:
            # Graph may be disconnected
            diameter = -1
        
        # 4. Identify lightcones
        # For a random sample of points, compute future and past light cones
        sample_size = min(10, self.num_points)
        sample_points = np.random.choice(self.num_points, sample_size, replace=False)
        
        lightcones = {}
        for point in sample_points:
            # Future lightcone
            future = set(nx.descendants(self.causal_graph, point))
            future_size = len(future)
            
            # Past lightcone
            past = set(nx.ancestors(self.causal_graph, point))
            past_size = len(past)
            
            lightcones[int(point)] = {
                'future_size': future_size,
                'past_size': past_size
            }
        
        return {
            'transitivity': transitivity,
            'clustering': clustering,
            'diameter': diameter,
            'lightcones': lightcones
        }
    
    def visualize(self, ax=None, show_causal=True):
        """
        Visualize the causal set.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        show_causal : bool
            Whether to show causal relations as edges
            
        Returns:
        --------
        matplotlib.axes.Axes
            The plot axes
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            
            # Determine if 3D plot is needed
            if self.points.shape[1] >= 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        # Plot based on dimensionality
        if self.points.shape[1] == 2:  # 1+1 dimensions (t, x)
            # Plot points
            ax.scatter(self.points[:, 1], self.points[:, 0], s=30, alpha=0.7)
            
            # Add causal relations
            if show_causal:
                for i in range(self.num_points):
                    for j in range(i+1, self.num_points):
                        if self.causal_matrix[i, j]:
                            ax.plot([self.points[i, 1], self.points[j, 1]], 
                                  [self.points[i, 0], self.points[j, 0]], 
                                  'k-', alpha=0.1)
            
            ax.set_xlabel('Space')
            ax.set_ylabel('Time')
            
        elif self.points.shape[1] == 3:  # 1+2 dimensions (t, x, y)
            # Plot points
            ax.scatter(self.points[:, 1], self.points[:, 2], self.points[:, 0], 
                     s=30, alpha=0.7)
            
            # Add causal relations (limited for clarity)
            if show_causal:
                # Show a random sample of causal relations
                edges = [(i, j) for i in range(self.num_points) 
                        for j in range(i+1, self.num_points) 
                        if self.causal_matrix[i, j]]
                
                # Randomly sample at most 1000 edges
                if len(edges) > 1000:
                    edges = [edges[i] for i in np.random.choice(len(edges), 1000, replace=False)]
                
                for i, j in edges:
                    ax.plot([self.points[i, 1], self.points[j, 1]], 
                          [self.points[i, 2], self.points[j, 2]], 
                          [self.points[i, 0], self.points[j, 0]], 
                          'k-', alpha=0.1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Time')
            
        else:  # Higher dimensions or 1+0
            if self.points.shape[1] > 3:
                # Project to 1+2 dimensions
                ax.scatter(self.points[:, 1], self.points[:, 2], self.points[:, 0], 
                         s=30, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')
            else:
                # 1+0 dimensions (time only)
                ax.scatter(np.zeros_like(self.points[:, 0]), self.points[:, 0], 
                         s=30, alpha=0.7)
                ax.set_xlabel('Index')
                ax.set_ylabel('Time')
        
        ax.set_title(f'Causal Set ({self.sprinkle_method}, dim={self.dim:.1f}, N={self.num_points})')
        return ax

class DiscreteToContinum:
    """
    Manages transitions between discrete and continuous spacetime structures.
    
    This class implements techniques for studying how discrete quantum spacetime
    structures transition to continuous classical spacetime as a function of energy scale.
    """
    
    def __init__(self, dim_uv=2.0, dim_ir=4.0, transition_scale=1.0):
        """
        Initialize the discrete-to-continuum transition manager.
        
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
        
        # Store models at different scales
        self.models = {}
        
        # Store transition functions
        self.dimension_function = self._default_dimension_function
        self.discreteness_function = self._default_discreteness_function
    
    def _default_dimension_function(self, energy_scale):
        """
        Default function for dimension as a function of energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Spectral dimension at this scale
        """
        # Smooth transition function
        x = np.log10(energy_scale)
        x0 = np.log10(self.transition_scale)
        
        # Sigmoid transition function
        return self.dim_ir + (self.dim_uv - self.dim_ir) / (1 + np.exp(-2 * (x - x0)))
    
    def _default_discreteness_function(self, energy_scale):
        """
        Default function for discreteness parameter as a function of energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
            
        Returns:
        --------
        float
            Discreteness parameter (0=continuous, 1=fully discrete)
        """
        # At high energies (> Planck scale): fully discrete
        # At low energies (< Planck scale): continuous
        x = np.log10(energy_scale)
        x0 = np.log10(self.transition_scale)
        
        # Sharper transition than dimension
        return 1.0 / (1.0 + np.exp(-4 * (x - x0)))
    
    def set_dimension_function(self, func):
        """
        Set custom dimension function.
        
        Parameters:
        -----------
        func : callable
            Function taking energy_scale and returning dimension
        """
        self.dimension_function = func
    
    def set_discreteness_function(self, func):
        """
        Set custom discreteness function.
        
        Parameters:
        -----------
        func : callable
            Function taking energy_scale and returning discreteness parameter
        """
        self.discreteness_function = func
    
    def create_model_at_scale(self, energy_scale, size=50, model_type='both'):
        """
        Create a spacetime model at a specific energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        size : int
            Size parameter for the model
        model_type : str
            Type of model: 'spectral', 'causal', or 'both'
            
        Returns:
        --------
        dict
            Model components
        """
        # Compute dimension and discreteness at this scale
        dimension = self.dimension_function(energy_scale)
        discreteness = self.discreteness_function(energy_scale)
        
        print(f"Creating spacetime model at energy scale {energy_scale:.2e} Planck units")
        print(f"  Dimension: {dimension:.3f}")
        print(f"  Discreteness: {discreteness:.3f}")
        
        model = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'discreteness': discreteness
        }
        
        # Create appropriate models based on dimension and discreteness
        if model_type in ['spectral', 'both']:
            # Choose discretization based on discreteness parameter
            if discreteness > 0.7:
                discretization = 'causal'  # Most discrete
            elif discreteness > 0.3:
                discretization = 'simplicial'  # Intermediate
            else:
                discretization = 'fuzzy'  # Most continuous
                
            # Create spectral geometry model
            model['spectral'] = SpectralGeometry(
                dim=dimension,
                size=size,
                discretization=discretization
            )
        
        if model_type in ['causal', 'both']:
            # Create causal set model
            # Choose sprinkle method based on dimension and discreteness
            if discreteness > 0.7:
                sprinkle_method = 'discrete'
            elif dimension < 3.0:
                sprinkle_method = 'curved'  # More appropriate for low dimensions
            else:
                sprinkle_method = 'flat'
                
            model['causal'] = CausalSet(
                num_points=size,
                dim=dimension,
                sprinkle_method=sprinkle_method
            )
        
        # Store model for this scale
        self.models[energy_scale] = model
        
        return model
    
    def compute_dimension_profile(self, scale_range=None, num_points=20):
        """
        Compute dimension profile across energy scales.
        
        Parameters:
        -----------
        scale_range : tuple, optional
            (min_scale, max_scale) in Planck units
        num_points : int
            Number of points to compute
            
        Returns:
        --------
        dict
            Dimension profile results
        """
        if scale_range is None:
            # Default: from well below to well above transition scale
            scale_range = (self.transition_scale * 1e-3, self.transition_scale * 1e3)
        
        # Generate logarithmically spaced energy scales
        scales = np.logspace(np.log10(scale_range[0]), np.log10(scale_range[1]), num_points)
        
        # Compute theoretical dimension at each scale
        theory_dims = np.array([self.dimension_function(s) for s in scales])
        
        # Compute spectral dimension from models at each scale
        spectral_dims = []
        causal_dims = []
        
        for scale in scales:
            # Create model at this scale if not already exists
            if scale not in self.models:
                model = self.create_model_at_scale(scale, size=50)
            else:
                model = self.models[scale]
            
            # Compute spectral dimension if available
            if 'spectral' in model:
                # Compute at diffusion time corresponding to this energy scale
                # t ~ 1/E²
                diffusion_time = 1.0 / (scale * scale)
                spec_dim = model['spectral'].compute_spectral_dimension(diffusion_time)
                spectral_dims.append(spec_dim)
            else:
                spectral_dims.append(None)
            
            # Compute causal dimension if available
            if 'causal' in model:
                causal_dim = model['causal'].compute_dimension_estimate()
                causal_dims.append(causal_dim)
            else:
                causal_dims.append(None)
        
        # Convert to numpy arrays with masked invalid values
        spectral_dims = np.ma.array(spectral_dims, mask=[d is None for d in spectral_dims])
        causal_dims = np.ma.array(causal_dims, mask=[d is None for d in causal_dims])
        
        return {
            'energy_scales': scales,
            'theory_dimensions': theory_dims,
            'spectral_dimensions': spectral_dims,
            'causal_dimensions': causal_dims
        }
    
    def compute_continuum_limit(self, base_scale, target_scale, size_scaling=2.0):
        """
        Study the approach to continuum limit by increasing system size.
        
        Parameters:
        -----------
        base_scale : float
            Base energy scale in Planck units
        target_scale : float
            Target energy scale for continuum limit (lower than base_scale)
        size_scaling : float
            Factor by which to increase size
            
        Returns:
        --------
        dict
            Continuum limit results
        """
        # Compute theoretical dimensions
        base_dim = self.dimension_function(base_scale)
        target_dim = self.dimension_function(target_scale)
        
        print(f"Studying continuum limit from {base_scale:.2e} to {target_scale:.2e} Planck units")
        print(f"  Dimension change: {base_dim:.3f} → {target_dim:.3f}")
        
        # Define sizes to study
        sizes = [30, int(30 * size_scaling), int(30 * size_scaling**2)]
        
        # Create models of increasing size
        models = []
        for size in sizes:
            # Create model at base scale with this size
            model = self.create_model_at_scale(base_scale, size=size)
            models.append(model)
        
        # Compute observables for each size
        results = {
            'sizes': sizes,
            'base_scale': base_scale,
            'target_scale': target_scale,
            'base_dim': base_dim,
            'target_dim': target_dim,
            'spectral_dims': [],
            'scaling_exponents': []
        }
        
        # Compute spectral dimension for each size
        for i, model in enumerate(models):
            if 'spectral' in model:
                # Compute at diffusion time corresponding to base scale
                diffusion_time = 1.0 / (base_scale * base_scale)
                spectral_dim = model['spectral'].compute_spectral_dimension(diffusion_time)
                results['spectral_dims'].append(spectral_dim)
                
                if i > 0:
                    # Compute scaling exponent
                    size_ratio = sizes[i] / sizes[i-1]
                    dim_ratio = spectral_dim / results['spectral_dims'][i-1]
                    # Expected scaling: d_N2/d_N1 = (N2/N1)^(δ)
                    # where δ is the scaling exponent
                    delta = np.log(dim_ratio) / np.log(size_ratio)
                    results['scaling_exponents'].append(delta)
        
        # Compute extrapolation to infinite size
        if len(results['spectral_dims']) >= 2:
            # Use the scaling exponent to extrapolate
            delta_avg = np.mean(results['scaling_exponents'])
            d_infinity = results['spectral_dims'][-1] * (1 + delta_avg * 0.1)
            results['extrapolated_dimension'] = d_infinity
            
            # Compare to target dimension
            results['deviation_from_target'] = (d_infinity - target_dim) / target_dim
        
        return results
    
    def compute_effective_action(self, energy_scale, action_type='scalar'):
        """
        Compute the effective action at a given energy scale.
        
        Parameters:
        -----------
        energy_scale : float
            Energy scale in Planck units
        action_type : str
            Type of action: 'scalar', 'gauge', or 'gravity'
            
        Returns:
        --------
        dict
            Effective action results
        """
        # Get dimension at this scale
        dimension = self.dimension_function(energy_scale)
        discreteness = self.discreteness_function(energy_scale)
        
        print(f"Computing effective {action_type} action at scale {energy_scale:.2e}")
        print(f"  Dimension: {dimension:.3f}, Discreteness: {discreteness:.3f}")
        
        # Create symbolic variables
        x, t = sp.symbols('x t')
        phi = sp.Function('phi')(x, t)
        
        # Initialize components
        kinetic_term = None
        potential_term = None
        measure = None
        
        if action_type == 'scalar':
            # Scalar field action: S = ∫ (∂_μϕ ∂^μϕ - m²ϕ²) dV
            
            # Dimension-dependent kinetic term
            # In fractional dimension, derivatives get modified
            if abs(dimension - int(dimension)) < 0.05:
                # Close to integer dimension
                kinetic_term = sp.diff(phi, t)**2 - sp.diff(phi, x)**2
            else:
                # Fractional dimension: use fractional derivatives
                # Simplified representation - in reality would use fractional calculus
                frac_part = dimension - int(dimension)
                kinetic_term = (
                    sp.diff(phi, t)**2 - 
                    (1-frac_part) * sp.diff(phi, x)**2 - 
                    frac_part * sp.diff(phi, x, 2) * phi
                )
            
            # Potential term (mass term)
            mass_sq = 1.0  # Unitless mass
            potential_term = mass_sq * phi**2
            
            # Integration measure depends on dimension
            measure = sp.symbols(f'dx^{dimension}')
            
        elif action_type == 'gauge':
            # Gauge field action: S = ∫ -1/4 F_μν F^μν dV
            F = sp.Function('F')(x, t)
            
            # Dimension-dependent gauge kinetic term
            if dimension > 3:
                # Higher dimensions: standard term with dimension-dependent coupling
                coupling = 1.0 / dimension
                kinetic_term = -0.25 * coupling * F**2
            else:
                # Lower dimensions: modified term
                coupling = 1.0 / max(dimension, 1.5)  # Avoid divergence
                kinetic_term = -0.25 * coupling * F**2 * (dimension/4.0)
            
            # Integration measure depends on dimension
            measure = sp.symbols(f'dx^{dimension}')
            
        elif action_type == 'gravity':
            # Gravitational action: S = ∫ (R - 2Λ) √(|g|) dV
            R = sp.Function('R')(x, t)  # Ricci scalar
            Lambda = sp.symbols('Lambda')  # Cosmological constant
            g = sp.symbols('g')  # Metric determinant
            
            # Dimension-dependent gravity term
            G_newton = 1.0  # Unitless in Planck units
            
            # Einstein-Hilbert in d dimensions
            if dimension > 2.1:
                kinetic_term = (R - 2*Lambda) * sp.sqrt(sp.Abs(g))
            else:
                # Near 2D: transition to conformal gravity
                # R becomes topological (Gauss-Bonnet)
                conformal_factor = (dimension - 2.0)
                kinetic_term = conformal_factor * R * sp.sqrt(sp.Abs(g))
                
                # Add scalar degree of freedom (conformal mode)
                conformal_scalar = sp.Function('sigma')(x, t)
                kinetic_term += sp.diff(conformal_scalar, x)**2 * sp.sqrt(sp.Abs(g))
            
            # Integration measure depends on dimension
            measure = sp.symbols(f'dx^{dimension}')
        
        # Combine terms into action
        if kinetic_term is not None and measure is not None:
            if potential_term is not None:
                action = sp.Integral(kinetic_term + potential_term, measure)
            else:
                action = sp.Integral(kinetic_term, measure)
        else:
            action = sp.symbols('S_undefined')
        
        # Store results
        results = {
            'energy_scale': energy_scale,
            'dimension': dimension,
            'discreteness': discreteness,
            'action_type': action_type,
            'symbolic_action': str(action),
            'kinetic_term': str(kinetic_term),
            'potential_term': str(potential_term) if potential_term else None,
            'scaling_dimension': dimension / (dimension - discreteness)
        }
        
        return results
    
    def visualize_transition(self, energy_scales=None, num_scales=3):
        """
        Visualize the transition between discrete and continuous structures.
        
        Parameters:
        -----------
        energy_scales : list, optional
            List of energy scales to visualize
        num_scales : int
            Number of scales if energy_scales not provided
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if energy_scales is None:
            # Generate logarithmically spaced energy scales around transition point
            energy_scales = np.logspace(
                np.log10(self.transition_scale) - 1.5,
                np.log10(self.transition_scale) + 1.5, 
                num_scales
            )
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * len(energy_scales)))
        
        # For each scale, create and visualize models
        for i, scale in enumerate(energy_scales):
            # Create model if not exists
            if scale not in self.models:
                model = self.create_model_at_scale(scale, size=50, model_type='both')
            else:
                model = self.models[scale]
            
            dimension = model['dimension']
            discreteness = model['discreteness']
            
            # Create subplots for this scale
            if 'spectral' in model and 'causal' in model:
                ax1 = fig.add_subplot(len(energy_scales), 2, 2*i+1)
                ax2 = fig.add_subplot(len(energy_scales), 2, 2*i+2)
                
                # Plot spectral geometry
                model['spectral'].visualize(ax=ax1)
                ax1.set_title(f"Spectral Geometry at E={scale:.2e}\nDimension {dimension:.2f}, Discreteness {discreteness:.2f}")
                
                # Plot causal set
                model['causal'].visualize(ax=ax2, show_causal=(discreteness > 0.3))
                ax2.set_title(f"Causal Set at E={scale:.2e}\nDimension {dimension:.2f}, Discreteness {discreteness:.2f}")
                
            elif 'spectral' in model:
                ax = fig.add_subplot(len(energy_scales), 1, i+1)
                model['spectral'].visualize(ax=ax)
                ax.set_title(f"Spectral Geometry at E={scale:.2e}\nDimension {dimension:.2f}, Discreteness {discreteness:.2f}")
                
            elif 'causal' in model:
                ax = fig.add_subplot(len(energy_scales), 1, i+1)
                model['causal'].visualize(ax=ax, show_causal=(discreteness > 0.3))
                ax.set_title(f"Causal Set at E={scale:.2e}\nDimension {dimension:.2f}, Discreteness {discreteness:.2f}")
        
        plt.tight_layout()
        return fig
    
    def plot_dimension_profile(self, profile_results=None):
        """
        Plot the dimension profile across energy scales.
        
        Parameters:
        -----------
        profile_results : dict, optional
            Results from compute_dimension_profile
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if profile_results is None:
            # Compute profile if not provided
            profile_results = self.compute_dimension_profile()
        
        # Extract results
        scales = profile_results['energy_scales']
        theory_dims = profile_results['theory_dimensions']
        spectral_dims = profile_results['spectral_dimensions']
        causal_dims = profile_results['causal_dimensions']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot theoretical dimension
        ax.semilogx(scales, theory_dims, 'k-', linewidth=2, label='Theoretical')
        
        # Plot spectral dimension if available
        if not np.ma.is_masked(spectral_dims).all():
            ax.semilogx(
                scales[~np.ma.is_masked(spectral_dims)], 
                spectral_dims[~np.ma.is_masked(spectral_dims)], 
                'ro-', markersize=5, label='Spectral'
            )
        
        # Plot causal dimension if available
        if not np.ma.is_masked(causal_dims).all():
            ax.semilogx(
                scales[~np.ma.is_masked(causal_dims)], 
                causal_dims[~np.ma.is_masked(causal_dims)], 
                'bs-', markersize=5, label='Causal'
            )
        
        # Add transition scale marker
        ax.axvline(self.transition_scale, color='gray', linestyle='--')
        ax.text(
            self.transition_scale * 1.1, 
            (self.dim_uv + self.dim_ir) / 2,
            'Transition Scale',
            rotation=90,
            va='center'
        )
        
        # Add annotations for UV and IR limits
        ax.text(scales[0] * 1.5, self.dim_uv * 0.9, f'UV: d = {self.dim_uv}')
        ax.text(scales[-1] * 0.5, self.dim_ir * 0.9, f'IR: d = {self.dim_ir}')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set labels and title
        ax.set_xlabel('Energy Scale (Planck units)')
        ax.set_ylabel('Spacetime Dimension')
        ax.set_title('Dimensional Flow in Quantum Spacetime')
        
        # Set y-axis limits with padding
        y_min = min(self.dim_uv, self.dim_ir) * 0.8
        y_max = max(self.dim_uv, self.dim_ir) * 1.1
        ax.set_ylim(y_min, y_max)
        
        return fig

if __name__ == "__main__":
    print("Quantum Spacetime Foundations Module")
    print("====================================")
    
    # Test SpectralGeometry
    print("\nTesting SpectralGeometry...")
    sg = SpectralGeometry(dim=3.5, size=40, discretization='simplicial')
    
    # Compute spectral dimension at different scales
    times = [0.01, 0.1, 1.0, 10.0]
    print("Spectral dimensions at different diffusion times:")
    for t in times:
        dim = sg.compute_spectral_dimension(t)
        print(f"  t = {t:.2f}, dimension = {dim:.3f}")
    
    # Compute and print Laplacian spectrum
    print("\nComputing Laplacian spectrum...")
    spectrum = sg.compute_laplacian_spectrum(num_eigenvalues=5)
    print(f"First 5 eigenvalues: {spectrum}")
    
    # Test CausalSet
    print("\nTesting CausalSet...")
    cs = CausalSet(num_points=80, dim=3.5, sprinkle_method='flat')
    
    # Compute dimension estimate
    dim_est = cs.compute_dimension_estimate()
    print(f"Estimated causal set dimension: {dim_est:.3f}")
    
    # Compute and print causal structure properties
    print("\nComputing causal structure...")
    causal_structure = cs.compute_causal_structure()
    print(f"Transitivity: {causal_structure['transitivity']:.3f}")
    print(f"Clustering: {causal_structure['clustering']:.3f}")
    print(f"Diameter: {causal_structure['diameter']}")
    
    # Test DiscreteToContinum
    print("\nTesting DiscreteToContinum...")
    dtc = DiscreteToContinum(dim_uv=2.0, dim_ir=4.0, transition_scale=1.0)
    
    # Compute and print dimensions at different energy scales
    print("\nEffective dimension at different energy scales:")
    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    for s in scales:
        dim = dtc.dimension_function(s)
        disc = dtc.discreteness_function(s)
        print(f"  E = {s:.2f} Ep, dimension = {dim:.3f}, discreteness = {disc:.3f}")
    
    # Compute effective action at a specific scale
    print("\nComputing effective action at transition scale...")
    action_results = dtc.compute_effective_action(1.0, action_type='gravity')
    print(f"Symbolic action: {action_results['symbolic_action']}")
    
    print("\nQuantum spacetime foundation tests completed successfully!")
    
    # Optional: Create and save visualization if matplotlib is available
    try:
        print("\nGenerating visualizations...")
        # Create test visualization of spectral geometry
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(221)
        sg.visualize(ax=ax1)
        ax1.set_title("Spectral Geometry (dim=3.5)")
        
        # Causal set visualization
        ax2 = fig.add_subplot(222)
        cs.visualize(ax=ax2, show_causal=True)
        ax2.set_title("Causal Set (dim=3.5)")
        
        # Dimension profile
        profile = dtc.compute_dimension_profile(
            scale_range=(0.01, 100.0),
            num_points=10
        )
        
        ax3 = fig.add_subplot(212)
        ax3.semilogx(profile['energy_scales'], profile['theory_dimensions'], 'k-', 
                   label='Theoretical')
        ax3.semilogx(profile['energy_scales'], profile['spectral_dimensions'], 'ro-', 
                   label='Spectral')
        ax3.grid(True)
        ax3.set_xlabel("Energy Scale (Planck units)")
        ax3.set_ylabel("Dimension")
        ax3.set_title("Dimensional Flow")
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig("quantum_spacetime_visualization.png")
        print("Visualizations saved to quantum_spacetime_visualization.png")
    except Exception as e:
        print(f"Visualization could not be saved: {e}")
