"""
Visualization Tools for Quantum Gravity

This module provides advanced visualization tools for the quantum gravity framework,
helping to communicate complex mathematical concepts through various visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pandas as pd


class QuantumSpacetimeVisualizer:
    """
    Visualization tools for quantum spacetime structures.
    """
    
    def __init__(self, style='default'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        style : str
            Visualization style ('default', 'dark', 'light', 'publication')
        """
        self.style = style
        self._set_style(style)
    
    def _set_style(self, style):
        """
        Set the matplotlib style.
        
        Parameters:
        -----------
        style : str
            Visualization style
        """
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'light':
            plt.style.use('default')
        elif style == 'publication':
            plt.style.use('seaborn-whitegrid')
        else:
            plt.style.use('default')
    
    def plot_causal_structure(self, graph, layout='spring', node_size=100, 
                             edge_width=1.0, colormap='viridis', save_path=None):
        """
        Plot a causal structure graph.
        
        Parameters:
        -----------
        graph : networkx.Graph
            Graph representing causal structure
        layout : str
            Graph layout algorithm ('spring', 'spectral', 'kamada_kawai', 'causal')
        node_size : int
            Size of nodes
        edge_width : float
            Width of edges
        colormap : str
            Colormap for node colors
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine layout
        if layout == 'spring':
            pos = nx.spring_layout(graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout == 'causal':
            # Custom causal layout (time flows upward)
            pos = {}
            for node in graph.nodes():
                in_degree = graph.in_degree(node)
                out_degree = graph.out_degree(node)
                pos[node] = (np.random.uniform(-1, 1), in_degree - out_degree)
        else:
            pos = nx.spring_layout(graph)
        
        # Determine node colors based on connectivity
        centrality = nx.betweenness_centrality(graph)
        node_colors = [centrality[node] for node in graph.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, 
                              node_color=node_colors, cmap=plt.cm.get_cmap(colormap), 
                              alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.6, 
                              edge_color='gray', arrows=True, ax=ax)
        
        # Draw labels if the graph is small enough
        if len(graph.nodes()) <= 50:
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax)
        
        # Set title and remove axis
        ax.set_title('Quantum Spacetime Causal Structure')
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap))
        sm.set_array(node_colors)
        plt.colorbar(sm, label='Centrality')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spectral_dimension(self, dimensions, diffusion_times, 
                               show_classical=True, save_path=None):
        """
        Plot spectral dimension vs diffusion time.
        
        Parameters:
        -----------
        dimensions : array
            Spectral dimensions at different diffusion times
        diffusion_times : array
            Diffusion time values
        show_classical : bool
            Whether to show classical dimension reference
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot dimension vs diffusion time
        ax.semilogx(diffusion_times, dimensions, 'b-', linewidth=2)
        
        # Add reference lines for classical dimensions if requested
        if show_classical:
            for d in range(2, 5):
                ax.axhline(y=d, color='r', linestyle='--', alpha=0.5, 
                         label=f"d = {d}" if d == 4 else None)
        
        # Labels and title
        ax.set_xlabel('Diffusion Time (Planck units)')
        ax.set_ylabel('Spectral Dimension')
        ax.set_title('Dimensional Flow in Quantum Spacetime')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Add legend if reference lines are shown
        if show_classical:
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def animate_dimensional_flow(self, dim_function, time_range=(1e-3, 1e3), 
                               n_frames=100, n_points=50, save_path=None):
        """
        Animate the dimensional flow in quantum spacetime.
        
        Parameters:
        -----------
        dim_function : callable
            Function that takes (x, y, t) and returns dimension
        time_range : tuple
            Range of time values (min, max)
        n_frames : int
            Number of animation frames
        n_points : int
            Number of points in each dimension
        save_path : str, optional
            Path to save the animation
            
        Returns:
        --------
        matplotlib.animation.Animation
            The animation object
        """
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        x = np.linspace(-5, 5, n_points)
        y = np.linspace(-5, 5, n_points)
        X, Y = np.meshgrid(x, y)
        
        # Create time values
        t_min, t_max = time_range
        times = np.logspace(np.log10(t_min), np.log10(t_max), n_frames)
        
        # Initial surface
        Z = dim_function(X, Y, times[0])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
        
        # Title text
        title = ax.set_title(f'Spacetime Dimension at t = {times[0]:.4g}')
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Dimension')
        ax.set_zlim(0, 5)
        
        # Animation update function
        def update(frame):
            ax.clear()
            Z = dim_function(X, Y, times[frame])
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
            title = ax.set_title(f'Spacetime Dimension at t = {times[frame]:.4g}')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Dimension')
            ax.set_zlim(0, 5)
            
            return surf, title
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=n_frames, 
                                    interval=100, blit=False)
        
        # Save if requested
        if save_path:
            ani.save(save_path, writer='pillow', fps=10, dpi=150)
        
        return ani
    
    def plot_entanglement_entropy(self, region_sizes, entropies, show_area_law=True,
                                volume_law=False, log_law=False, save_path=None):
        """
        Plot entanglement entropy vs region size.
        
        Parameters:
        -----------
        region_sizes : array
            Sizes of regions
        entropies : array
            Entanglement entropies
        show_area_law : bool
            Whether to show area law reference
        volume_law : bool
            Whether to show volume law reference
        log_law : bool
            Whether to show logarithmic law reference
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot entropy vs region size
        ax.plot(region_sizes, entropies, 'bo-', linewidth=2, label='QG Entropy')
        
        # Add reference curves
        x_range = np.linspace(min(region_sizes), max(region_sizes), 100)
        
        if show_area_law:
            # Area law: S ~ R^(d-1)
            # Normalize to match at the middle point
            mid_idx = len(region_sizes) // 2
            area_factor = entropies[mid_idx] / (region_sizes[mid_idx]**(2/3))
            ax.plot(x_range, area_factor * x_range**(2/3), 'r--', 
                  label='Area Law (S ~ R²/³)')
        
        if volume_law:
            # Volume law: S ~ R^d
            # Normalize to match at the middle point
            mid_idx = len(region_sizes) // 2
            vol_factor = entropies[mid_idx] / region_sizes[mid_idx]
            ax.plot(x_range, vol_factor * x_range, 'g--', 
                  label='Volume Law (S ~ R)')
        
        if log_law:
            # Log law: S ~ log(R)
            # Normalize to match at the middle point
            mid_idx = len(region_sizes) // 2
            log_factor = entropies[mid_idx] / np.log(region_sizes[mid_idx])
            ax.plot(x_range, log_factor * np.log(x_range), 'm--', 
                  label='Log Law (S ~ log R)')
        
        # Labels and title
        ax.set_xlabel('Region Size')
        ax.set_ylabel('Entanglement Entropy')
        ax.set_title('Entanglement Entropy Scaling in Quantum Spacetime')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class QuantumBlackHoleVisualizer:
    """
    Visualization tools for quantum black holes.
    """
    
    def __init__(self, style='default'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        style : str
            Visualization style ('default', 'dark', 'light', 'publication')
        """
        self.style = style
        self._set_style(style)
    
    def _set_style(self, style):
        """
        Set the matplotlib style.
        
        Parameters:
        -----------
        style : str
            Visualization style
        """
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'light':
            plt.style.use('default')
        elif style == 'publication':
            plt.style.use('seaborn-whitegrid')
        else:
            plt.style.use('default')
    
    def plot_hawking_radiation(self, frequencies, spectrum, 
                              show_thermal=True, temperature=None,
                              save_path=None):
        """
        Plot Hawking radiation spectrum.
        
        Parameters:
        -----------
        frequencies : array
            Frequency values
        spectrum : array
            Radiation spectrum
        show_thermal : bool
            Whether to show thermal reference
        temperature : float, optional
            Black hole temperature for thermal reference
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectrum
        ax.plot(frequencies, spectrum, 'b-', linewidth=2, label='QG Spectrum')
        
        # Add thermal reference if requested
        if show_thermal and temperature is not None:
            # Planck thermal spectrum
            thermal = frequencies**3 / (np.exp(frequencies / temperature) - 1)
            # Normalize to match at peak
            peak_idx = np.argmax(spectrum)
            norm_factor = spectrum[peak_idx] / thermal[peak_idx]
            ax.plot(frequencies, norm_factor * thermal, 'r--', 
                  label='Thermal Spectrum')
        
        # Labels and title
        ax.set_xlabel('Frequency (Planck units)')
        ax.set_ylabel('Emission Rate')
        ax.set_title('Hawking Radiation Spectrum with Quantum Gravity Corrections')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_evaporation_curve(self, times, masses, 
                              show_standard=True, save_path=None):
        """
        Plot black hole evaporation curve.
        
        Parameters:
        -----------
        times : array
            Time values
        masses : array
            Black hole masses
        show_standard : bool
            Whether to show standard evaporation curve
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mass vs time
        ax.plot(times, masses, 'b-', linewidth=2, label='QG Evaporation')
        
        # Add standard evaporation curve if requested
        if show_standard:
            # Standard evaporation: M(t) = M_0 * (1 - t/t_evap)^(1/3)
            # where t_evap = 5120 π G² M_0³ / (ħ c⁴)
            # Simplify: M(t) = M_0 * (1 - t/t_evap)^(1/3)
            
            # Estimate t_evap from the input data
            m0 = masses[0]
            t_evap = times[-1] * 1.2  # Slightly extend beyond the last point
            
            # Generate standard curve
            std_times = np.linspace(times[0], times[-1], 100)
            std_masses = m0 * (1 - std_times/t_evap)**(1/3)
            std_masses = np.maximum(std_masses, 0)  # Prevent negative masses
            
            ax.plot(std_times, std_masses, 'r--', 
                  label='Standard Evaporation')
        
        # Labels and title
        ax.set_xlabel('Time (Planck units)')
        ax.set_ylabel('Black Hole Mass (Planck units)')
        ax.set_title('Black Hole Evaporation with Quantum Gravity Corrections')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_interior_geometry(self, radii, curvatures,
                                   show_classical=True, save_path=None):
        """
        Visualize black hole interior geometry.
        
        Parameters:
        -----------
        radii : array
            Radial coordinates
        curvatures : array
            Curvature values
        show_classical : bool
            Whether to show classical reference
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Curvature vs radius
        ax1.semilogy(radii, curvatures, 'b-', linewidth=2, label='QG Curvature')
        
        # Add classical reference if requested
        if show_classical:
            # Classical curvature ~ 1/r³
            # Normalize to match at the largest radius
            r_max_idx = len(radii) - 1
            class_factor = curvatures[r_max_idx] * radii[r_max_idx]**3
            classical = class_factor / np.maximum(radii, 1e-10)**3
            
            ax1.semilogy(radii, classical, 'r--', 
                       label='Classical (1/r³)')
            
            # Also add vertical line at r=0 (classical singularity)
            ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Labels and grid
        ax1.set_xlabel('Radius (Planck units)')
        ax1.set_ylabel('Curvature')
        ax1.set_title('Black Hole Interior Curvature')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # Plot 2: Visualize geometry (simplified 2D representation)
        # Use radii and curvatures to create a warped visualization
        
        # Create a grid of points in polar coordinates
        theta = np.linspace(0, 2*np.pi, 100)
        r_grid = np.linspace(0, max(radii), 20)
        
        # Create coordinate arrays
        theta_grid, r_grid_2d = np.meshgrid(theta, r_grid)
        
        # Create warp factor based on curvature
        # Interpolate curvature to the r_grid values
        from scipy.interpolate import interp1d
        curv_interp = interp1d(radii, curvatures, 
                               bounds_error=False, fill_value='extrapolate')
        warp_factor = 1.0 / (1.0 + curv_interp(r_grid) / 10.0)
        
        # Apply warp factor to coordinates
        warped_r = np.outer(r_grid * warp_factor, np.ones(len(theta)))
        
        # Convert to Cartesian coordinates
        X = warped_r * np.cos(theta_grid)
        Y = warped_r * np.sin(theta_grid)
        
        # Create a colormap based on curvature
        curvature_grid = np.outer(curv_interp(r_grid), np.ones(len(theta)))
        
        # Plot grid lines
        for i in range(len(r_grid)):
            ax2.plot(X[i,:], Y[i,:], 'k-', alpha=0.3)
        
        for i in range(0, len(theta), 10):
            ax2.plot(X[:,i], Y[:,i], 'k-', alpha=0.3)
        
        # Color the surface using a scatter plot
        sc = ax2.scatter(X, Y, c=np.log10(curvature_grid), 
                       cmap='plasma', s=10, alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('Log Curvature')
        
        # Labels and settings
        ax2.set_xlabel('X (Planck units)')
        ax2.set_ylabel('Y (Planck units)')
        ax2.set_title('Quantum-Corrected Interior Geometry')
        ax2.set_aspect('equal')
        ax2.grid(False)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class HolographicVisualizer:
    """
    Visualization tools for holographic duality.
    """
    
    def __init__(self, style='default'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        style : str
            Visualization style ('default', 'dark', 'light', 'publication')
        """
        self.style = style
        self._set_style(style)
    
    def _set_style(self, style):
        """
        Set the matplotlib style.
        
        Parameters:
        -----------
        style : str
            Visualization style
        """
        if style == 'dark':
            plt.style.use('dark_background')
        elif style == 'light':
            plt.style.use('default')
        elif style == 'publication':
            plt.style.use('seaborn-whitegrid')
        else:
            plt.style.use('default')
    
    def visualize_bulk_boundary_mapping(self, bulk_points, boundary_points,
                                      minimal_surface=None, save_path=None):
        """
        Visualize the bulk-boundary mapping in holographic duality.
        
        Parameters:
        -----------
        bulk_points : array
            Coordinates of bulk points
        boundary_points : array
            Coordinates of boundary points
        minimal_surface : array, optional
            Coordinates of minimal surface points
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot bulk points
        ax.scatter(bulk_points[:,0], bulk_points[:,1], bulk_points[:,2],
                 c='blue', alpha=0.6, label='Bulk')
        
        # Plot boundary points
        ax.scatter(boundary_points[:,0], boundary_points[:,1], boundary_points[:,2],
                 c='red', s=50, alpha=0.8, label='Boundary')
        
        # Plot minimal surface if provided
        if minimal_surface is not None:
            ax.scatter(minimal_surface[:,0], minimal_surface[:,1], minimal_surface[:,2],
                     c='green', alpha=0.7, label='Minimal Surface')
            
            # Try to create a surface if there are enough points
            if len(minimal_surface) > 10:
                try:
                    from scipy.spatial import Delaunay
                    tri = Delaunay(minimal_surface[:,:2])
                    ax.plot_trisurf(minimal_surface[:,0], minimal_surface[:,1], minimal_surface[:,2],
                                  triangles=tri.simplices, alpha=0.3, color='green')
                except:
                    pass
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Holographic Bulk-Boundary Mapping')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.max([
            np.ptp(bulk_points[:,0]), 
            np.ptp(bulk_points[:,1]), 
            np.ptp(bulk_points[:,2])
        ])
        mid_x = np.mean([np.min(bulk_points[:,0]), np.max(bulk_points[:,0])])
        mid_y = np.mean([np.min(bulk_points[:,1]), np.max(bulk_points[:,1])])
        mid_z = np.mean([np.min(bulk_points[:,2]), np.max(bulk_points[:,2])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_entanglement_wedge(self, boundary_region, wedge_points,
                               minimal_surface=None, save_path=None):
        """
        Visualize the entanglement wedge in holographic duality.
        
        Parameters:
        -----------
        boundary_region : array
            Coordinates of boundary region points
        wedge_points : array
            Coordinates of entanglement wedge points
        minimal_surface : array, optional
            Coordinates of minimal surface points
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot boundary region
        ax.scatter(boundary_region[:,0], boundary_region[:,1], boundary_region[:,2],
                 c='red', s=50, alpha=0.8, label='Boundary Region')
        
        # Plot entanglement wedge
        ax.scatter(wedge_points[:,0], wedge_points[:,1], wedge_points[:,2],
                 c='blue', alpha=0.4, label='Entanglement Wedge')
        
        # Plot minimal surface if provided
        if minimal_surface is not None:
            ax.scatter(minimal_surface[:,0], minimal_surface[:,1], minimal_surface[:,2],
                     c='green', alpha=0.7, label='Minimal Surface')
            
            # Try to create a surface if there are enough points
            if len(minimal_surface) > 10:
                try:
                    from scipy.spatial import Delaunay
                    tri = Delaunay(minimal_surface[:,:2])
                    ax.plot_trisurf(minimal_surface[:,0], minimal_surface[:,1], minimal_surface[:,2],
                                  triangles=tri.simplices, alpha=0.3, color='green')
                except:
                    pass
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Holographic Entanglement Wedge')
        ax.legend()
        
        # Set equal aspect ratio
        all_points = np.vstack([boundary_region, wedge_points])
        if minimal_surface is not None:
            all_points = np.vstack([all_points, minimal_surface])
            
        max_range = np.max([
            np.ptp(all_points[:,0]), 
            np.ptp(all_points[:,1]), 
            np.ptp(all_points[:,2])
        ])
        mid_x = np.mean([np.min(all_points[:,0]), np.max(all_points[:,0])])
        mid_y = np.mean([np.min(all_points[:,1]), np.max(all_points[:,1])])
        mid_z = np.mean([np.min(all_points[:,2]), np.max(all_points[:,2])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 