"""Tensor Network module for QFT.

This module implements tensor network methods for quantum field theory,
including matrix product states (MPS), tensor renormalization group (TRG),
and related approaches.
"""

import numpy as np
import scipy.linalg as la
from typing import List, Dict, Tuple, Callable, Optional, Union
import matplotlib.pyplot as plt

class MatrixProductState:
    """Matrix Product State (MPS) representation of a quantum state.
    
    Represents a quantum state as a product of tensors with the structure:
    |ψ⟩ = ∑ A[1]^i1 A[2]^i2 ... A[N]^iN |i1 i2 ... iN⟩
    """
    
    def __init__(self, physical_dims, bond_dim=2, random_init=True):
        """Initialize an MPS.
        
        Args:
            physical_dims: List of physical dimensions for each site
            bond_dim: Maximum bond dimension
            random_init: Whether to initialize with random tensors
        """
        self.N = len(physical_dims)  # Number of sites
        self.physical_dims = physical_dims
        self.bond_dim = bond_dim
        
        # Create tensor list with proper dimensions
        self.tensors = []
        
        if random_init:
            # First tensor: (1, d, D)
            self.tensors.append(np.random.random((1, physical_dims[0], min(bond_dim, physical_dims[0]))))
            
            # Middle tensors: (D, d, D)
            for i in range(1, self.N-1):
                d_in = min(bond_dim, np.prod(physical_dims[:i]))
                d_out = min(bond_dim, np.prod(physical_dims[:i+1]))
                self.tensors.append(np.random.random((d_in, physical_dims[i], d_out)))
            
            # Last tensor: (D, d, 1)
            if self.N > 1:
                d_in = min(bond_dim, np.prod(physical_dims[:-1]))
                self.tensors.append(np.random.random((d_in, physical_dims[-1], 1)))
        else:
            # Create empty tensors with proper dimensions
            self.tensors = [np.zeros((1, physical_dims[0], min(bond_dim, physical_dims[0])))]
            for i in range(1, self.N-1):
                d_in = min(bond_dim, np.prod(physical_dims[:i]))
                d_out = min(bond_dim, np.prod(physical_dims[:i+1]))
                self.tensors.append(np.zeros((d_in, physical_dims[i], d_out)))
            if self.N > 1:
                d_in = min(bond_dim, np.prod(physical_dims[:-1]))
                self.tensors.append(np.zeros((d_in, physical_dims[-1], 1)))
        
        # Normalize
        self.normalize()
    
    def get_state_vector(self):
        """Convert MPS to full state vector.
        
        Returns:
            Full state vector (for small systems)
        """
        # Start with the first tensor
        state = self.tensors[0][0, :, :]
        
        # Contract with each subsequent tensor
        for i in range(1, self.N):
            # Reshape to merge physical index with bond index
            state = np.reshape(state, (np.prod(state.shape[:-1]), state.shape[-1]))
            # Contract with next tensor along bond dimension
            next_tensor = self.tensors[i]
            state = np.matmul(state, next_tensor.reshape(next_tensor.shape[0], -1))
            # Reshape to separate physical indices
            state = np.reshape(state, (*self.physical_dims[:i], next_tensor.shape[1], next_tensor.shape[2]))
        
        # Final state shape should be (d1, d2, ..., dN, 1)
        state = state.reshape(self.physical_dims)
        
        return state
    
    def normalize(self):
        """Normalize the MPS (canonical form)."""
        # Right-canonical form
        for i in range(self.N-1, 0, -1):
            tensor = self.tensors[i]
            shape = tensor.shape
            tensor = tensor.reshape(shape[0], -1)
            
            # SVD
            u, s, vh = la.svd(tensor, full_matrices=False)
            
            # Update current tensor
            self.tensors[i] = vh.reshape(-1, shape[1], shape[2])
            
            # Update previous tensor
            self.tensors[i-1] = np.tensordot(self.tensors[i-1], 
                                           (u * s).reshape(shape[0], -1),
                                           axes=(-1, 0))
        
        # Normalize first tensor
        norm = np.sqrt(np.sum(np.abs(self.tensors[0])**2))
        if norm > 1e-10:
            self.tensors[0] /= norm
    
    def apply_operator(self, operator, site):
        """Apply a local operator to the MPS.
        
        Args:
            operator: Operator matrix (d x d)
            site: Site to apply the operator
            
        Returns:
            New MPS with operator applied
        """
        # Copy the current MPS
        result = MatrixProductState(self.physical_dims, self.bond_dim, random_init=False)
        for i in range(self.N):
            result.tensors[i] = self.tensors[i].copy()
            
        # Apply operator at the specified site
        tensor = result.tensors[site]
        tensor = np.tensordot(operator, tensor, axes=(1, 1))
        # Transpose to get correct order (a, i, b) -> (a, j, b)
        tensor = np.transpose(tensor, (1, 0, 2))
        result.tensors[site] = tensor
        
        return result
    
    def expectation_value(self, operator, site):
        """Calculate expectation value of a local operator.
        
        Args:
            operator: Operator matrix (d x d)
            site: Site to calculate expectation value
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        # Apply operator to MPS
        op_mps = self.apply_operator(operator, site)
        
        # Calculate overlap
        return self.overlap(op_mps)
    
    def overlap(self, other_mps):
        """Calculate overlap between two MPS.
        
        Args:
            other_mps: Another MPS
            
        Returns:
            Overlap ⟨ψ|ψ'⟩
        """
        # Check if dimensions match
        if self.N != other_mps.N:
            raise ValueError("MPS have different lengths")
        
        # Contract from left to right
        # Start with first tensor contraction
        left = np.tensordot(np.conj(self.tensors[0]), other_mps.tensors[0], 
                          axes=([0, 1], [0, 1]))
        
        # Contract with remaining tensors
        for i in range(1, self.N):
            left = np.tensordot(left, np.conj(self.tensors[i]), 
                              axes=(0, 0))
            left = np.tensordot(left, other_mps.tensors[i], 
                              axes=([0, 1], [0, 1]))
        
        # Final contraction should give a scalar
        return left[0, 0]

class TensorRenormalizationGroup:
    """Tensor Renormalization Group (TRG) for 2D classical systems.
    
    Implements the TRG algorithm for approximating partition functions
    and observables in 2D classical lattice models.
    """
    
    def __init__(self, initial_tensor, bond_dim=4):
        """Initialize TRG with a local tensor.
        
        Args:
            initial_tensor: Local tensor representing Boltzmann weights
            bond_dim: Maximum bond dimension for truncation
        """
        self.tensor = initial_tensor
        self.bond_dim = bond_dim
        
        # Check tensor shape - should be 4D for 2D square lattice
        if len(initial_tensor.shape) != 4:
            raise ValueError("Initial tensor should have 4 indices for 2D square lattice")
    
    def coarse_grain_step(self):
        """Perform a single coarse-graining step.
        
        Returns:
            New coarse-grained tensor
        """
        # Extract tensor shape
        D = self.tensor.shape[0]
        
        # Reshape tensor for SVD
        # Contract and reshape to group indices
        T1 = np.reshape(self.tensor, (D*D, D*D))
        
        # SVD
        u, s, vh = la.svd(T1, full_matrices=False)
        
        # Truncate
        if len(s) > self.bond_dim:
            u = u[:, :self.bond_dim]
            s = s[:self.bond_dim]
            vh = vh[:self.bond_dim, :]
        
        # Form new tensors
        sqrt_s = np.sqrt(s)
        u_s = u * sqrt_s
        vh_s = vh * sqrt_s[:, np.newaxis]
        
        # Reshape into 3-index tensors
        S1 = np.reshape(u_s, (D, D, -1))
        S2 = np.reshape(vh_s, (-1, D, D))
        
        # Contract to form new 4-index tensor
        # (This is a simplified version - full TRG has more contractions)
        new_tensor = np.tensordot(S1, S2, axes=(2, 0))
        # Permute indices to standard order
        new_tensor = np.transpose(new_tensor, (0, 2, 1, 3))
        
        # Update the tensor
        self.tensor = new_tensor
        
        return new_tensor
    
    def compute_partition_function(self, iterations):
        """Approximate the partition function.
        
        Args:
            iterations: Number of coarse-graining steps
            
        Returns:
            Approximate partition function value
        """
        # Track tensor trace and system size
        tensor_trace = np.trace(np.reshape(self.tensor, (self.tensor.shape[0]*self.tensor.shape[1], 
                                                      self.tensor.shape[2]*self.tensor.shape[3])))
        system_size = 2**iterations  # Each iteration doubles the effective system size
        
        for _ in range(iterations):
            self.coarse_grain_step()
            
            # Update trace for partition function
            tensor_trace = np.trace(np.reshape(self.tensor, (self.tensor.shape[0]*self.tensor.shape[1], 
                                                         self.tensor.shape[2]*self.tensor.shape[3])))
            
        # Final result is approximately Z^(1/N)
        return tensor_trace**(system_size**2)

def ising_initial_tensor(beta):
    """Create initial tensor for 2D Ising model.
    
    Args:
        beta: Inverse temperature
    
    Returns:
        Local tensor for TRG
    """
    # Ising model Boltzmann weights
    w_plus = np.exp(beta)  # parallel spins
    w_minus = np.exp(-beta)  # anti-parallel spins
    
    # Local tensor
    tensor = np.zeros((2, 2, 2, 2))
    
    # Fill tensor elements
    # T_{ijkl} corresponds to spins i,j,k,l around a plaquette
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # Calculate Boltzmann weight based on neighboring spins
                    weight = 1.0
                    weight *= w_plus if i == j else w_minus
                    weight *= w_plus if j == k else w_minus
                    weight *= w_plus if k == l else w_minus
                    weight *= w_plus if l == i else w_minus
                    tensor[i, j, k, l] = np.sqrt(weight)
    
    return tensor

# Example usage
if __name__ == "__main__":
    # Example 1: Matrix Product State
    print("Example 1: Matrix Product State")
    
    # Create a simple MPS for a 6-site spin-1/2 chain
    physical_dims = [2] * 6
    mps = MatrixProductState(physical_dims, bond_dim=4)
    
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Calculate expectation values
    mag_z = sum(mps.expectation_value(sigma_z, i) for i in range(6)) / 6
    mag_x = sum(mps.expectation_value(sigma_x, i) for i in range(6)) / 6
    
    print(f"Magnetization Z: {mag_z.real:.6f}")
    print(f"Magnetization X: {mag_x.real:.6f}")
    
    # Example 2: Tensor Renormalization Group
    print("\nExample 2: Tensor Renormalization Group")
    
    # Calculate partition function for 2D Ising model
    # near critical temperature
    T_c = 2.0 / np.log(1 + np.sqrt(2))  # Critical temperature
    temperatures = [0.9 * T_c, T_c, 1.1 * T_c]
    
    print("Ising model partition function:")
    for T in temperatures:
        beta = 1.0 / T
        initial_tensor = ising_initial_tensor(beta)
        trg = TensorRenormalizationGroup(initial_tensor, bond_dim=10)
        
        # Approximate partition function with 4 iterations (16x16 lattice)
        Z = trg.compute_partition_function(4)
        
        print(f"  T/T_c = {T/T_c:.2f}: log(Z)/N ≈ {np.log(Z)/16**2:.6f}")
    
    print("\nNote: These are simplified implementations for educational purposes.") 