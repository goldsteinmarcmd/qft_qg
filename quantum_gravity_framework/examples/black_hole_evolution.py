"""
Quantum Black Hole Formation and Evaporation

This example demonstrates a comprehensive simulation of quantum black hole
formation and evaporation, showing how dimensional flow affects the process.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for direct execution
sys.path.insert(0, os.path.abspath('..'))

# Import necessary modules
from quantum_gravity_framework import UnifiedFramework
from quantum_gravity_framework.black_hole_microstates import BlackHoleMicrostates
from quantum_gravity_framework.quantum_spacetime_foundations import SpectralGeometry


def simulate_black_hole_evolution(initial_mass=10.0, dim_uv=2.0, dim_ir=4.0, 
                                transition_scale=1.0, num_steps=100):
    """
    Simulate quantum black hole formation and complete evaporation.
    
    Parameters:
    -----------
    initial_mass : float
        Initial mass of the black hole in Planck units
    dim_uv : float
        UV (high energy) spectral dimension
    dim_ir : float
        IR (low energy) spectral dimension
    transition_scale : float
        Energy scale of dimensional transition in Planck units
    num_steps : int
        Number of time steps in the simulation
    
    Returns:
    --------
    dict
        Simulation results
    """
    print(f"Simulating quantum black hole evolution with initial mass = {initial_mass} M_p")
    
    # Initialize black hole microstates module
    bh = BlackHoleMicrostates(dim_uv=dim_uv, dim_ir=dim_ir, transition_scale=transition_scale)
    
    # Initialize spacetime geometry
    spacetime = SpectralGeometry(dim=dim_ir, size=50)
    
    # Define time steps for simulation
    # Use variable time steps (finer near formation and final evaporation)
    formation_steps = int(num_steps * 0.2)
    evaporation_steps = int(num_steps * 0.8)
    
    # Formation phase: matter collapse
    formation_times = np.linspace(-10, 0, formation_steps)
    
    # Evaporation phase: logarithmic time steps to capture final burst
    evap_times_raw = np.logspace(0, np.log10(bh.compute_evaporation_time(initial_mass) * 1.1), 
                                evaporation_steps)
    evaporation_times = np.sort(evap_times_raw)
    
    # Combine time steps
    times = np.concatenate([formation_times, evaporation_times])
    
    # Arrays to store simulation results
    masses = np.zeros_like(times)
    entropies = np.zeros_like(times)
    temperatures = np.zeros_like(times)
    radii = np.zeros_like(times)
    dimensions = np.zeros_like(times)
    information_content = np.zeros_like(times)
    hawking_rates = np.zeros_like(times)
    
    # Initial conditions for formation phase
    for i, t in enumerate(formation_times):
        # Simplified model: exponential approach to initial_mass
        progress = np.exp(t) / np.exp(0)
        if progress < 0.01:
            progress = 0.01  # Minimum starting mass
            
        # During formation, mass increases
        masses[i] = initial_mass * progress
        
        # Compute other properties from this mass
        dimensions[i] = dim_ir  # Initially in IR regime
        radii[i] = bh.compute_radius(masses[i], dimensions[i])
        entropies[i] = bh.compute_entropy(masses[i], use_dimension_flow=False)
        temperatures[i] = bh.compute_temperature(masses[i], use_dimension_flow=False)
        information_content[i] = 0.0  # Initially no information has escaped
        hawking_rates[i] = 0.0  # No evaporation during formation
    
    # Evaporation phase simulation
    current_mass = initial_mass
    
    for i, t in enumerate(evaporation_times):
        idx = i + formation_steps  # Index in the full arrays
        
        # Compute energy scale associated with black hole
        # Inverse relationship between mass and energy scale
        energy_scale = 1.0 / (2.0 * current_mass)
        
        # Get dimension at this energy scale
        dimensions[idx] = bh.compute_effective_dimension(energy_scale)
        
        # Compute evaporation rate with dimension corrections
        hawking_rate = bh.compute_evaporation_rate(
            current_mass, 
            dimensions[idx]
        )
        
        # Store the rate
        hawking_rates[idx] = hawking_rate
        
        # Update mass using variable time step
        if i > 0:
            dt = t - evaporation_times[i-1]
            mass_loss = hawking_rate * dt
            current_mass -= mass_loss
            
            # Prevent negative mass
            if current_mass < 0:
                current_mass = 0
        
        # Store current mass
        masses[idx] = current_mass
        
        # Compute other properties from this mass
        radii[idx] = bh.compute_radius(current_mass, dimensions[idx])
        entropies[idx] = bh.compute_entropy(current_mass, use_dimension_flow=True)
        temperatures[idx] = bh.compute_temperature(current_mass, use_dimension_flow=True)
        
        # Information content: starts emerging in late phases of evaporation
        # Simplified model: information emerges as mass approaches the Planck scale
        if current_mass < 2.0:
            # Information recovery accelerates as mass approaches zero
            info_fraction = 1.0 - (current_mass / 2.0)**2
            information_content[idx] = entropies[0] * info_fraction
        else:
            information_content[idx] = 0.0
    
    # Compile results
    results = {
        'times': times,
        'masses': masses,
        'entropies': entropies,
        'temperatures': temperatures,
        'radii': radii,
        'dimensions': dimensions,
        'hawking_rates': hawking_rates,
        'information_content': information_content,
        'parameters': {
            'initial_mass': initial_mass,
            'dim_uv': dim_uv,
            'dim_ir': dim_ir,
            'transition_scale': transition_scale
        }
    }
    
    return results


def visualize_black_hole_evolution(results, save_plots=True, animation=False):
    """
    Visualize black hole evolution results.
    
    Parameters:
    -----------
    results : dict
        Simulation results from simulate_black_hole_evolution
    save_plots : bool
        Whether to save plots to files
    animation : bool
        Whether to create animation of the evolution
        
    Returns:
    --------
    tuple
        (figures, animation_object)
    """
    # Extract data from results
    times = results['times']
    masses = results['masses']
    entropies = results['entropies']
    temperatures = results['temperatures']
    radii = results['radii']
    dimensions = results['dimensions']
    hawking_rates = results['hawking_rates']
    information_content = results['information_content']
    
    # Find formation/evaporation transition index
    transition_idx = np.argmin(np.abs(times))
    
    # Create figures
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot mass vs time
    axes1[0, 0].plot(times, masses, 'b-', linewidth=2)
    axes1[0, 0].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes1[0, 0].set_xlabel('Time (Planck units)')
    axes1[0, 0].set_ylabel('Mass (Planck masses)')
    axes1[0, 0].set_title('Black Hole Mass Evolution')
    axes1[0, 0].grid(True, alpha=0.3)
    
    # Plot entropy vs time
    axes1[0, 1].plot(times, entropies, 'g-', linewidth=2)
    axes1[0, 1].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes1[0, 1].set_xlabel('Time (Planck units)')
    axes1[0, 1].set_ylabel('Entropy (Planck units)')
    axes1[0, 1].set_title('Black Hole Entropy')
    axes1[0, 1].grid(True, alpha=0.3)
    
    # Plot temperature vs time
    axes1[1, 0].plot(times, temperatures, 'r-', linewidth=2)
    axes1[1, 0].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes1[1, 0].set_xlabel('Time (Planck units)')
    axes1[1, 0].set_ylabel('Temperature (Planck units)')
    axes1[1, 0].set_title('Black Hole Temperature')
    axes1[1, 0].grid(True, alpha=0.3)
    
    # Plot effective dimension vs time
    axes1[1, 1].plot(times, dimensions, 'k-', linewidth=2)
    axes1[1, 1].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes1[1, 1].set_xlabel('Time (Planck units)')
    axes1[1, 1].set_ylabel('Effective Dimension')
    axes1[1, 1].set_title('Effective Spacetime Dimension')
    axes1[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        fig1.savefig("black_hole_evolution_1.png", dpi=300, bbox_inches='tight')
    
    # Create second figure for information and evaporation rate
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Hawking radiation rate
    axes2[0].semilogy(times, hawking_rates, 'm-', linewidth=2)
    axes2[0].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes2[0].set_xlabel('Time (Planck units)')
    axes2[0].set_ylabel('Evaporation Rate (Planck units)')
    axes2[0].set_title('Hawking Radiation Rate')
    axes2[0].grid(True, alpha=0.3)
    
    # Plot information content vs time
    axes2[1].plot(times, information_content, 'c-', linewidth=2)
    # Also plot entropy for comparison
    axes2[1].plot(times, entropies, 'g--', linewidth=1.5, alpha=0.7, label='Entropy')
    axes2[1].axvline(x=times[transition_idx], color='r', linestyle='--', alpha=0.7)
    axes2[1].set_xlabel('Time (Planck units)')
    axes2[1].set_ylabel('Information (bits)')
    axes2[1].set_title('Information Content and Recovery')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        fig2.savefig("black_hole_evolution_2.png", dpi=300, bbox_inches='tight')
    
    # Create animation if requested
    anim = None
    if animation:
        # Create figure for animation
        fig_anim = plt.figure(figsize=(10, 8))
        ax_anim = fig_anim.add_subplot(111, projection='3d')
        
        # Define update function for animation
        def update(frame):
            ax_anim.clear()
            
            # Current properties
            r = radii[frame]
            m = masses[frame]
            t = times[frame]
            dim = dimensions[frame]
            
            # Create sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot black hole
            ax_anim.plot_surface(x, y, z, color='black', alpha=0.7)
            
            # Plot evaporation
            if frame >= transition_idx:
                # Add Hawking radiation as points
                num_particles = int(hawking_rates[frame] * 1000)
                if num_particles > 0:
                    rad_r = np.random.uniform(r, r*2, num_particles)
                    rad_theta = np.random.uniform(0, 2*np.pi, num_particles)
                    rad_phi = np.random.uniform(0, np.pi, num_particles)
                    
                    rad_x = rad_r * np.sin(rad_phi) * np.cos(rad_theta)
                    rad_y = rad_r * np.sin(rad_phi) * np.sin(rad_theta)
                    rad_z = rad_r * np.cos(rad_phi)
                    
                    ax_anim.scatter(rad_x, rad_y, rad_z, c='yellow', s=1, alpha=0.5)
            
            # Set plot properties
            max_radius = max(radii) * 1.5
            ax_anim.set_xlim(-max_radius, max_radius)
            ax_anim.set_ylim(-max_radius, max_radius)
            ax_anim.set_zlim(-max_radius, max_radius)
            
            ax_anim.set_title(f"Black Hole Evolution\nTime: {t:.2f}, Mass: {m:.2f}, Dimension: {dim:.2f}")
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig_anim, update, frames=len(times), interval=50, blit=True)
        
        if save_plots:
            anim.save("black_hole_evolution.mp4", writer='ffmpeg', dpi=200)
    
    # Return figures and animation
    return ((fig1, fig2), anim)


def main():
    """Main execution function."""
    # Define parameters
    initial_mass = 10.0
    dim_uv = 2.0
    dim_ir = 4.0
    transition_scale = 1.0
    
    # Run simulation
    results = simulate_black_hole_evolution(
        initial_mass=initial_mass,
        dim_uv=dim_uv,
        dim_ir=dim_ir,
        transition_scale=transition_scale,
        num_steps=200
    )
    
    # Visualize results
    figs, anim = visualize_black_hole_evolution(
        results, 
        save_plots=True,
        animation=False  # Set to True to create animation (requires ffmpeg)
    )
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main() 