"""Feynman diagram calculation module for QFT.

This module implements tools for generating and calculating Feynman diagrams
for quantum field theory processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional

@dataclass
class Particle:
    """Class representing a particle in a Feynman diagram."""
    name: str
    mass: float
    charge: float = 0.0
    spin: float = 0.0
    
    def propagator(self, momentum_squared):
        """Calculate the propagator for this particle."""
        return 1 / (momentum_squared - self.mass**2 + 1e-10j)

@dataclass
class Vertex:
    """Class representing an interaction vertex in a Feynman diagram."""
    position: Tuple[float, float]
    particles: List[Particle]
    coupling: float
    
    def amplitude(self):
        """Calculate the amplitude contribution from this vertex."""
        return self.coupling

@dataclass
class Edge:
    """Class representing an edge (propagator) in a Feynman diagram."""
    start_vertex: int  # Index of starting vertex
    end_vertex: int    # Index of ending vertex
    particle: Particle
    momentum: np.ndarray
    
    def propagator_value(self):
        """Calculate the propagator value for this edge."""
        p_squared = np.sum(self.momentum**2)
        return self.particle.propagator(p_squared)

class FeynmanDiagram:
    """Class representing a complete Feynman diagram."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.incoming_particles: List[Particle] = []
        self.outgoing_particles: List[Particle] = []
    
    def add_vertex(self, vertex: Vertex) -> int:
        """Add a vertex to the diagram and return its index."""
        self.vertices.append(vertex)
        return len(self.vertices) - 1
    
    def add_edge(self, edge: Edge):
        """Add an edge (propagator) to the diagram."""
        self.edges.append(edge)
    
    def add_incoming_particle(self, particle: Particle):
        """Add an incoming particle to the diagram."""
        self.incoming_particles.append(particle)
    
    def add_outgoing_particle(self, particle: Particle):
        """Add an outgoing particle to the diagram."""
        self.outgoing_particles.append(particle)
    
    def calculate_amplitude(self) -> complex:
        """Calculate the amplitude for this Feynman diagram."""
        # Start with vertex factors
        amplitude = 1.0
        for vertex in self.vertices:
            amplitude *= vertex.amplitude()
        
        # Multiply by propagator factors
        for edge in self.edges:
            amplitude *= edge.propagator_value()
        
        return amplitude
    
    def visualize(self, ax=None, show=True):
        """Visualize the Feynman diagram."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot vertices
        for i, vertex in enumerate(self.vertices):
            ax.plot(vertex.position[0], vertex.position[1], 'o', markersize=10)
            ax.text(vertex.position[0], vertex.position[1]+0.1, f"v{i}")
        
        # Plot edges
        for edge in self.edges:
            start_pos = self.vertices[edge.start_vertex].position
            end_pos = self.vertices[edge.end_vertex].position
            
            # Draw a line for the propagator
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], '-', 
                   label=edge.particle.name)
            
            # Add momentum label at midpoint
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax.text(mid_x, mid_y, f"p={np.round(edge.momentum, 2)}")
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Feynman Diagram: {self.name}')
        ax.legend()
        
        if show:
            plt.show()
        
        return ax

def electron_positron_scattering() -> FeynmanDiagram:
    """Create a Feynman diagram for electron-positron scattering."""
    # Create particles
    electron = Particle(name="electron", mass=0.511, charge=-1, spin=0.5)
    positron = Particle(name="positron", mass=0.511, charge=1, spin=0.5)
    photon = Particle(name="photon", mass=0, charge=0, spin=1)
    
    # Create diagram
    diagram = FeynmanDiagram("Electron-Positron Scattering")
    
    # Add vertices
    v1 = Vertex(position=(0, 1), particles=[electron, photon], coupling=1)
    v2 = Vertex(position=(0, -1), particles=[positron, photon], coupling=1)
    
    v1_idx = diagram.add_vertex(v1)
    v2_idx = diagram.add_vertex(v2)
    
    # Add photon propagator
    momentum = np.array([1.0, 0, 0, 0])  # Example momentum
    photon_edge = Edge(start_vertex=v1_idx, end_vertex=v2_idx, 
                      particle=photon, momentum=momentum)
    diagram.add_edge(photon_edge)
    
    # Add incoming/outgoing particles
    diagram.add_incoming_particle(electron)
    diagram.add_incoming_particle(positron)
    diagram.add_outgoing_particle(electron)
    diagram.add_outgoing_particle(positron)
    
    return diagram

# Example usage
if __name__ == "__main__":
    diagram = electron_positron_scattering()
    amplitude = diagram.calculate_amplitude()
    print(f"Scattering amplitude: {amplitude}")
    diagram.visualize() 