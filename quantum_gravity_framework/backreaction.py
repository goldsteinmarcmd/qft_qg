"""
Backreaction of Quantum Fields on Spacetime Geometry

This module implements the mechanisms by which quantum fields 
affect spacetime geometry - one of the key aspects of quantum gravity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

from quantum_gravity_framework.category_theory import CategoryTheoryGeometry
from quantum_gravity_framework.quantum_spacetime import QuantumSpacetimeAxioms


class QuantumBackreaction:
    """
    Implementation of backreaction mechanisms in quantum gravity.
    """
    
    def __init__(self, dimension=4, qg_scale=1e19, planck_length=1.616e-35):
        """
        Initialize the backreaction framework.
        
        Args:
            dimension: Classical spacetime dimension
            qg_scale: Quantum gravity energy scale in GeV
            planck_length: Planck length in meters
        """
        self.dimension = dimension
        self.qg_scale = qg_scale
        self.planck_length = planck_length
        
        # Initialize QG components
        self.qst = QuantumSpacetimeAxioms(dim=dimension)
        self.category_geometry = CategoryTheoryGeometry(dim=dimension)
        
        # Backreaction coupling constants
        self.xi_scalar = 0.1  # Scalar field-spacetime coupling
        self.xi_gauge = 0.15  # Gauge field-spacetime coupling
        self.xi_fermi = 0.08  # Fermion field-spacetime coupling
        
        # Background space parameters
        self.initialize_background()
    
    def initialize_background(self):
        """
        Initialize the background spacetime structure.
        """
        # Create a simplicial complex representation of spacetime
        # This is a simplified representation using graph structure
        self.spacetime_graph = nx.Graph()
        
        # Number of vertices (determined by categorical structure)
        n_vertices = len(self.category_geometry.objects)
        
        # Create vertices
        for i in range(n_vertices):
            # Store vertex properties 
            self.spacetime_graph.add_node(i, dimension=4.0, curvature=0.0)
        
        # Create edges based on categorical morphisms
        for morph_id, morph in self.category_geometry.morphisms.items():
            source = morph['source']
            target = morph['target']
            
            # Add edge if source and target are valid node ids
            if (isinstance(source, (int, str)) and 
                isinstance(target, (int, str)) and
                source in self.spacetime_graph.nodes and
                target in self.spacetime_graph.nodes):
                
                # Edge weight represents distance/metric
                weight = 1.0
                if 'properties' in morph and 'distance' in morph['properties']:
                    weight = morph['properties']['distance']
                
                self.spacetime_graph.add_edge(source, target, weight=weight)
        
        # Initial metric values
        self.metric_determinant = np.ones(n_vertices)
        self.ricci_scalar = np.zeros(n_vertices)
    
    def compute_effective_dimension(self, energy_scale):
        """
        Compute effective spectral dimension at a given energy scale.
        
        Args:
            energy_scale: Energy scale in GeV
            
        Returns:
            Effective spectral dimension
        """
        # Convert energy scale to diffusion time (inverse relation)
        diffusion_time = 1.0 / (energy_scale * energy_scale)
        
        # Use quantum spacetime module to compute spectral dimension
        return self.qst.compute_spectral_dimension(diffusion_time)
    
    def apply_field_backreaction(self, field_config, field_type="scalar"):
        """
        Apply backreaction from quantum fields to spacetime.
        
        Args:
            field_config: Field configuration with energy distribution
            field_type: Type of field ("scalar", "gauge", "fermion")
            
        Returns:
            Dictionary with modified spacetime properties
        """
        # Select appropriate coupling constant
        if field_type == "scalar":
            coupling = self.xi_scalar
        elif field_type == "gauge":
            coupling = self.xi_gauge
        elif field_type == "fermion":
            coupling = self.xi_fermi
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
        
        # Extract field energy-momentum distribution
        energy_density = field_config.get('energy_density', 
                                         np.ones(len(self.spacetime_graph.nodes)) * 0.1)
        
        # Ensure dimensions match
        if len(energy_density) != len(self.spacetime_graph.nodes):
            # Resize or sample energy density to match graph
            if isinstance(energy_density, np.ndarray):
                energy_density = np.resize(energy_density, len(self.spacetime_graph.nodes))
            else:
                energy_density = np.ones(len(self.spacetime_graph.nodes)) * energy_density
        
        # Scale by QG scale - backreaction is suppressed at low energies
        scaled_coupling = coupling * (energy_density / self.qg_scale**2)
        
        # Update metric determinant (simplified volume element change)
        self.metric_determinant *= (1.0 + scaled_coupling)
        
        # Update Ricci scalar (simplified curvature response)
        self.ricci_scalar += scaled_coupling * energy_density
        
        # Apply changes to graph nodes
        for i, node in enumerate(self.spacetime_graph.nodes):
            # Update local dimension (simplification of quantum effect)
            current_dim = self.spacetime_graph.nodes[node]['dimension']
            
            # Dimension flows toward UV value at high energy density
            uv_dim = 2.0  # Typical UV dimension in QG approaches
            ir_dim = 4.0  # Classical dimension
            
            flow_factor = scaled_coupling[i]
            new_dim = current_dim - (current_dim - uv_dim) * flow_factor
            
            # Ensure dimension stays in reasonable range
            new_dim = max(uv_dim, min(new_dim, ir_dim))
            
            # Update node properties
            self.spacetime_graph.nodes[node]['dimension'] = new_dim
            self.spacetime_graph.nodes[node]['curvature'] = self.ricci_scalar[i]
        
        # Modify edge properties based on the backreaction
        # This represents deformation of the spacetime metric
        for u, v, data in self.spacetime_graph.edges(data=True):
            # Get node indices
            u_idx = list(self.spacetime_graph.nodes).index(u)
            v_idx = list(self.spacetime_graph.nodes).index(v)
            
            # Average backreaction effect between connected nodes
            avg_effect = 0.5 * (scaled_coupling[u_idx] + scaled_coupling[v_idx])
            
            # Update edge weight (metric deformation)
            current_weight = data['weight']
            
            # Non-commutative effect: distances can increase or decrease
            # depending on the field type and energy
            if field_type == "scalar":
                # Scalar fields tend to increase distances (inflation-like)
                new_weight = current_weight * (1.0 + avg_effect)
            elif field_type == "gauge":
                # Gauge fields can decrease distances at high energies
                new_weight = current_weight / (1.0 + avg_effect)
            else:
                # Fermions have mixed effects
                new_weight = current_weight * (1.0 + avg_effect * 
                                             np.sin(avg_effect * np.pi))
            
            # Update edge weight
            self.spacetime_graph.edges[u, v]['weight'] = new_weight
        
        # Return modified spacetime properties
        return {
            'metric_determinant': self.metric_determinant,
            'ricci_scalar': self.ricci_scalar,
            'average_dimension': np.mean([
                self.spacetime_graph.nodes[node]['dimension'] 
                for node in self.spacetime_graph.nodes
            ]),
            'average_curvature': np.mean(self.ricci_scalar),
            'graph_edges': self.spacetime_graph.number_of_edges(),
            'graph_nodes': self.spacetime_graph.number_of_nodes()
        }
    
    def solve_semiclassical_einstein(self, field_config, time_range=(0, 10), n_steps=100):
        """
        Solve semiclassical Einstein equations with quantum corrections.
        
        This is a simplified model using the trace anomaly and
        quantum corrections to the energy-momentum tensor.
        
        Args:
            field_config: Field configuration with energy distribution
            time_range: Integration time range
            n_steps: Number of integration steps
            
        Returns:
            Dictionary with solution data
        """
        # Extract field energy-momentum
        if isinstance(field_config, dict):
            energy_density = field_config.get('energy_density', 0.1)
            pressure = field_config.get('pressure', energy_density/3)  # Radiation equation of state
            field_type = field_config.get('field_type', 'scalar')
        else:
            # Default values
            energy_density = 0.1
            pressure = energy_density/3
            field_type = 'scalar'
        
        # Select appropriate coupling
        if field_type == "scalar":
            coupling = self.xi_scalar
        elif field_type == "gauge":
            coupling = self.xi_gauge
        elif field_type == "fermion":
            coupling = self.xi_fermi
        else:
            coupling = self.xi_scalar
        
        # Scale for QG effects
        qg_coupling = coupling / self.qg_scale**2
        
        # Semiclassical Einstein equation:
        # G_μν = 8πG [T_μν + qg_coupling * (quantum corrections)]
        
        # In a simplified model, we can use a Friedmann-like system with QG corrections
        # For the scale factor a(t) of a homogeneous, isotropic universe
        
        def einstein_equations(t, y):
            """
            System of differential equations for modified Friedmann model.
            
            y[0] = a(t) (scale factor)
            y[1] = da/dt (scale factor derivative)
            """
            a, a_dot = y
            
            # Classical Friedmann equation: (da/dt)^2 = (8πG/3) ρ a^2
            # Modified with quantum corrections
            
            # Hubble parameter
            H = a_dot / a
            
            # Quantum correction term (trace anomaly-like)
            # Higher derivatives of a represented by H^2 terms
            quantum_correction = qg_coupling * (H**2 + H**4 * a**2)
            
            # Second derivative of a from modified Friedmann
            a_dotdot = -4*np.pi*(energy_density + 3*pressure)*a/3 + quantum_correction*a
            
            return [a_dot, a_dotdot]
        
        # Initial conditions: a(0) = 1, da/dt(0) = H0
        initial_state = [1.0, 0.1]
        
        # Solve the system
        t_eval = np.linspace(time_range[0], time_range[1], n_steps)
        solution = solve_ivp(
            einstein_equations, 
            time_range, 
            initial_state, 
            t_eval=t_eval,
            method='RK45'
        )
        
        # Extract solution
        t = solution.t
        a = solution.y[0]
        a_dot = solution.y[1]
        
        # Compute derived quantities
        hubble = a_dot / a
        deceleration = -a * np.gradient(np.gradient(a, t), t) / a_dot**2
        
        # Effective dimension along the solution
        effective_dimension = []
        for i in range(len(t)):
            # Estimate energy scale from Hubble parameter
            energy_scale = hubble[i] * self.qg_scale
            dim = self.compute_effective_dimension(energy_scale)
            effective_dimension.append(dim)
        
        return {
            'time': t,
            'scale_factor': a,
            'hubble_parameter': hubble,
            'deceleration_parameter': deceleration,
            'effective_dimension': effective_dimension,
            'field_energy_density': energy_density,
            'field_pressure': pressure,
            'field_type': field_type,
            'qg_coupling': qg_coupling * self.qg_scale**2  # Original coupling
        }
    
    def visualize_backreaction(self, initial_config, final_config):
        """
        Visualize the backreaction effects on spacetime.
        
        Args:
            initial_config: Initial spacetime configuration
            final_config: Final spacetime configuration after backreaction
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot graph representation of spacetime
        ax = axs[0, 0]
        
        # Create spring layout for graph visualization
        pos = nx.spring_layout(self.spacetime_graph)
        
        # Color nodes by dimension
        dimensions = [self.spacetime_graph.nodes[node]['dimension'] 
                    for node in self.spacetime_graph.nodes]
        
        # Draw nodes with dimension-based coloring
        cmap = plt.cm.viridis
        nodes = nx.draw_networkx_nodes(
            self.spacetime_graph, pos, ax=ax,
            node_color=dimensions, 
            node_size=50,
            cmap=cmap, 
            vmin=2.0, vmax=4.0
        )
        
        # Add a colorbar
        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label('Spectral Dimension')
        
        # Draw edges with width proportional to weight
        edge_weights = [d['weight'] for u, v, d in self.spacetime_graph.edges(data=True)]
        
        # Normalize weights for visualization
        if edge_weights:
            max_weight = max(edge_weights)
            edge_weights = [w/max_weight*2 for w in edge_weights]
        
        nx.draw_networkx_edges(
            self.spacetime_graph, pos, ax=ax,
            width=edge_weights, 
            alpha=0.5
        )
        
        ax.set_title('Spacetime Graph with Backreaction')
        ax.set_axis_off()
        
        # Plot dimensional flow - backreaction relationship
        ax = axs[0, 1]
        
        # Generate energy scale range
        energy_scales = np.logspace(0, 19, 50)  # 1 GeV to 10^19 GeV
        dimensions = [self.compute_effective_dimension(E) for E in energy_scales]
        
        ax.semilogx(energy_scales, dimensions, 'b-', linewidth=2)
        ax.set_xlabel('Energy Scale (GeV)')
        ax.set_ylabel('Spectral Dimension')
        ax.set_title('Dimensional Flow with Energy Scale')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Highlight QG scale
        ax.axvline(x=self.qg_scale, color='red', linestyle='--')
        ax.text(self.qg_scale*1.1, 3.0, 'QG Scale', 
               rotation=90, color='red')
        
        # Plot Ricci scalar distribution
        ax = axs[1, 0]
        
        # Histogram of curvature values
        ax.hist(self.ricci_scalar, bins=20, alpha=0.7)
        ax.set_xlabel('Ricci Scalar')
        ax.set_ylabel('Frequency')
        ax.set_title('Curvature Distribution')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Plot metric determinant changes
        ax = axs[1, 1]
        
        # Compute percent change from initial to final
        if hasattr(initial_config, 'metric_determinant') and hasattr(final_config, 'metric_determinant'):
            initial_det = initial_config.metric_determinant
            final_det = final_config.metric_determinant
            pct_change = (final_det - initial_det) / initial_det * 100
            
            ax.hist(pct_change, bins=20, alpha=0.7)
            ax.set_xlabel('Percent Change in √g')
            ax.set_ylabel('Frequency')
            ax.set_title('Volume Element Changes')
        else:
            # Alternative plot if configurations don't have metric data
            sample_points = np.linspace(0, 1, 50)
            ax.plot(sample_points, self.metric_determinant, 'g-', linewidth=2)
            ax.set_xlabel('Vertex Index (normalized)')
            ax.set_ylabel('√g (Volume Element)')
            ax.set_title('Metric Determinant Distribution')
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add overall title
        plt.suptitle('Backreaction of Quantum Fields on Spacetime', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig


def compute_combined_backreaction(scalar_config=None, gauge_config=None, fermion_config=None,
                                qg_scale=1e16):
    """
    Compute the combined backreaction from multiple field types.
    
    Args:
        scalar_config: Scalar field configuration
        gauge_config: Gauge field configuration
        fermion_config: Fermion field configuration
        qg_scale: QG energy scale in GeV
        
    Returns:
        Dictionary with backreaction results
    """
    # Initialize backreaction module
    backreaction = QuantumBackreaction(qg_scale=qg_scale)
    
    # Store initial configuration
    initial_config = {
        'average_dimension': 4.0,
        'average_curvature': 0.0,
        'metric_determinant': backreaction.metric_determinant.copy()
    }
    
    # Apply backreaction from each field type if provided
    results = {}
    
    if scalar_config is not None:
        scalar_config['field_type'] = 'scalar'
        results['scalar'] = backreaction.apply_field_backreaction(
            scalar_config, field_type='scalar')
    
    if gauge_config is not None:
        gauge_config['field_type'] = 'gauge'
        results['gauge'] = backreaction.apply_field_backreaction(
            gauge_config, field_type='gauge')
    
    if fermion_config is not None:
        fermion_config['field_type'] = 'fermion'
        results['fermion'] = backreaction.apply_field_backreaction(
            fermion_config, field_type='fermion')
    
    # Store final configuration
    final_config = {
        'average_dimension': np.mean([
            backreaction.spacetime_graph.nodes[node]['dimension'] 
            for node in backreaction.spacetime_graph.nodes
        ]),
        'average_curvature': np.mean(backreaction.ricci_scalar),
        'metric_determinant': backreaction.metric_determinant.copy()
    }
    
    # Create visualization
    fig = backreaction.visualize_backreaction(initial_config, final_config)
    
    # Save figure
    fig.savefig('qg_backreaction.png', dpi=300, bbox_inches='tight')
    
    # Solve semiclassical Einstein equations
    # Combine field configurations
    combined_energy = 0.0
    combined_pressure = 0.0
    
    if scalar_config is not None and 'energy_density' in scalar_config:
        combined_energy += scalar_config['energy_density']
        combined_pressure += scalar_config.get('pressure', 
                                             scalar_config['energy_density']/3)
    
    if gauge_config is not None and 'energy_density' in gauge_config:
        combined_energy += gauge_config['energy_density']
        combined_pressure += gauge_config.get('pressure', 0)
    
    if fermion_config is not None and 'energy_density' in fermion_config:
        combined_energy += fermion_config['energy_density']
        combined_pressure += fermion_config.get('pressure', 
                                             fermion_config['energy_density']/3)
    
    combined_config = {
        'energy_density': combined_energy,
        'pressure': combined_pressure,
        'field_type': 'combined'
    }
    
    # Solve equations
    einstein_solution = backreaction.solve_semiclassical_einstein(combined_config)
    
    # Add combined results
    results['combined'] = final_config
    results['einstein_solution'] = einstein_solution
    results['initial_config'] = initial_config
    results['qg_scale'] = qg_scale
    
    return results


# Example usage
if __name__ == "__main__":
    # Sample field configurations
    scalar_config = {
        'energy_density': 0.1,
        'pressure': 0.03,
    }
    
    gauge_config = {
        'energy_density': 0.2,
        'pressure': 0.0,  # Gauge fields have zero trace classically
    }
    
    # Compute backreaction
    results = compute_combined_backreaction(
        scalar_config=scalar_config,
        gauge_config=gauge_config,
        qg_scale=1e16
    )
    
    # Print summary
    print("\nBackreaction Results:")
    print("--------------------")
    print(f"Initial dimension: {results['initial_config']['average_dimension']:.4f}")
    print(f"Final dimension: {results['combined']['average_dimension']:.4f}")
    print(f"Final average curvature: {results['combined']['average_curvature']:.6e}")
    print(f"QG scale: {results['qg_scale']:.1e} GeV") 