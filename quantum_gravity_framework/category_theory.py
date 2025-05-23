"""
Advanced Quantum Gravity Framework: Category Theory Implementation

This module implements the category theory approach to quantum geometry
using higher categories and groupoids to model quantum spacetime.
"""

import numpy as np
import networkx as nx
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Callable, Set, Optional


class MonoidalCategory:
    """
    Implements a monoidal category for categorical quantum field theory.
    
    A monoidal category has a tensor product operation that combines objects and morphisms.
    This is essential for modeling quantum field theory structures.
    """
    
    def __init__(self, name, objects, tensor_product=None, morphisms=None):
        """
        Initialize a monoidal category.
        
        Parameters:
        -----------
        name : str
            Name of the category
        objects : list
            List of object names in the category
        tensor_product : function, optional
            Function to compute tensor product of objects
        morphisms : dict, optional
            Dictionary of morphisms in the category
        """
        self.name = name
        self.objects = objects
        self.tensor_product = tensor_product or (lambda x, y: f"{x}⊗{y}")
        self.morphisms = morphisms or {}
        
        # Initialize identity morphisms for each object
        for obj in objects:
            id_name = f"id_{obj}"
            if id_name not in self.morphisms:
                self.morphisms[id_name] = {
                    'source': obj,
                    'target': obj,
                    'is_identity': True
                }
    
    def tensor(self, obj1, obj2):
        """
        Compute tensor product of two objects.
        
        Parameters:
        -----------
        obj1, obj2 : str
            Objects to combine with tensor product
            
        Returns:
        --------
        str
            Tensor product object
        """
        return self.tensor_product(obj1, obj2)
    
    def add_morphism(self, name, source, target, properties=None):
        """
        Add a morphism to the category.
        
        Parameters:
        -----------
        name : str
            Name of the morphism
        source : str
            Source object
        target : str
            Target object
        properties : dict, optional
            Additional properties of the morphism
            
        Returns:
        --------
        dict
            The created morphism
        """
        if source not in self.objects:
            self.objects.append(source)
        if target not in self.objects:
            self.objects.append(target)
            
        morphism = {
            'source': source,
            'target': target,
            'properties': properties or {}
        }
        
        self.morphisms[name] = morphism
        return morphism
    
    def compose(self, morphism1, morphism2):
        """
        Compose two morphisms.
        
        Parameters:
        -----------
        morphism1, morphism2 : str
            Names of morphisms to compose
            
        Returns:
        --------
        dict or None
            The composed morphism, or None if composition is not possible
        """
        if morphism1 not in self.morphisms or morphism2 not in self.morphisms:
            return None
            
        m1 = self.morphisms[morphism1]
        m2 = self.morphisms[morphism2]
        
        # Check if composition is possible
        if m1['source'] != m2['target']:
            return None
            
        # Create composed morphism
        composed_name = f"{morphism1}_then_{morphism2}"
        composed = {
            'source': m2['source'],
            'target': m1['target'],
            'components': [morphism2, morphism1],
            'properties': {}
        }
        
        # Combine properties as needed
        if 'properties' in m1 and 'properties' in m2:
            for key in set(m1['properties'].keys()) & set(m2['properties'].keys()):
                if isinstance(m1['properties'][key], (int, float)) and isinstance(m2['properties'][key], (int, float)):
                    composed['properties'][key] = m1['properties'][key] * m2['properties'][key]
        
        return composed


class Functor:
    """
    Implements a functor between categories.
    
    A functor is a mapping between categories that preserves structure.
    """
    
    def __init__(self, source, target, name, object_map, morphism_map=None):
        """
        Initialize a functor.
        
        Parameters:
        -----------
        source : MonoidalCategory
            Source category
        target : MonoidalCategory
            Target category
        name : str
            Name of the functor
        object_map : dict
            Mapping from source objects to target objects
        morphism_map : dict, optional
            Mapping from source morphisms to target morphisms
        """
        self.source = source
        self.target = target
        self.name = name
        self.object_map = object_map
        self.morphism_map = morphism_map or {}
        
        # Validate object map
        for src_obj in source.objects:
            if src_obj not in object_map:
                # Default identity mapping if not specified
                self.object_map[src_obj] = src_obj
        
        # Create default morphism mappings for identities
        for obj in source.objects:
            id_src = f"id_{obj}"
            if id_src in source.morphisms and id_src not in self.morphism_map:
                id_tgt = f"id_{self.object_map[obj]}"
                if id_tgt in target.morphisms:
                    self.morphism_map[id_src] = id_tgt
    
    def apply_to_object(self, obj):
        """
        Apply functor to an object.
        
        Parameters:
        -----------
        obj : str
            Source object
            
        Returns:
        --------
        str
            Target object
        """
        if obj in self.object_map:
            return self.object_map[obj]
        
        # Try to handle tensor products
        if '⊗' in obj:
            parts = obj.replace('(', '').replace(')', '').split('⊗')
            mapped_parts = [self.apply_to_object(part) for part in parts]
            return self.target.tensor(*mapped_parts)
            
        return obj  # Default identity mapping
    
    def apply_to_morphism(self, morphism):
        """
        Apply functor to a morphism.
        
        Parameters:
        -----------
        morphism : str
            Source morphism
            
        Returns:
        --------
        str
            Target morphism
        """
        if morphism in self.morphism_map:
            return self.morphism_map[morphism]
            
        # Default handling for unmapped morphisms
        if morphism in self.source.morphisms:
            src_morph = self.source.morphisms[morphism]
            src_obj = src_morph['source']
            tgt_obj = src_morph['target']
            
            # Map objects
            mapped_src = self.apply_to_object(src_obj)
            mapped_tgt = self.apply_to_object(tgt_obj)
            
            # Look for a morphism in target category with matching source/target
            for tgt_morph_name, tgt_morph in self.target.morphisms.items():
                if tgt_morph['source'] == mapped_src and tgt_morph['target'] == mapped_tgt:
                    return tgt_morph_name
                    
        return None  # No mapping found


class CategoryTheoryGeometry:
    """
    Implements the category theory approach to quantum geometry.
    
    Uses higher category structures to model quantum spacetime,
    incorporating concepts from 2-categories and groupoids.
    """
    
    def __init__(self, dim=4, n_points=50):
        self.dim = dim
        self.n_points = n_points
        
        # Initialize category structure
        self.objects = self._initialize_objects()
        self.morphisms = self._initialize_morphisms()
        self.two_morphisms = self._initialize_two_morphisms()
        
        # Structure for composition operations
        self.composition_table = self._create_composition_table()
        
        # 2-Hilbert space structure
        self.two_inner_products = self._initialize_two_inner_products()
        
        # New: Higher category structures
        self.three_morphisms = self._initialize_three_morphisms()
        
        # New: Topos structure
        self.subobject_classifier = self._initialize_subobject_classifier()
        self.presheaves = self._initialize_presheaves()
        
    def _initialize_objects(self):
        """Initialize objects (points) in the category."""
        # In a category theoretic approach, objects are often events or regions
        # For simplicity, we'll use a set of labeled points
        
        # Create points with additional structure
        objects = {}
        for i in range(self.n_points):
            # Generate random coordinates in space
            coords = np.random.uniform(-1, 1, self.dim)
            
            # Create an object
            objects[f"p{i}"] = {
                'coordinates': coords,
                'dimension': 0,  # 0-dimensional object (point)
                'properties': {
                    'local_volume': self.dim**(-0.5)  # Simple volume measure
                }
            }
            
        # Also create some higher-dimensional objects (regions)
        # by grouping points
        for dim in range(1, self.dim + 1):
            # Number of dim-dimensional objects to create
            n_dim_objects = max(5, self.n_points // (2**dim))
            
            for i in range(n_dim_objects):
                # Select a set of lower-dimensional objects to include
                included_objects = []
                
                # Select points randomly
                n_points = min(dim + 1, self.n_points)
                point_indices = np.random.choice(self.n_points, n_points, replace=False)
                for idx in point_indices:
                    included_objects.append(f"p{idx}")
                
                # Create the higher-dimensional object
                obj_id = f"d{dim}_o{i}"
                objects[obj_id] = {
                    'included_objects': included_objects,
                    'dimension': dim,
                    'properties': {
                        'local_volume': np.random.gamma(dim, 0.1)  # Simple volume measure
                    }
                }
                
        return objects
    
    def _initialize_morphisms(self):
        """Initialize morphisms (maps between objects) in the category."""
        # In the context of quantum geometry, morphisms are often
        # quantum operators, paths, or transformations
        
        morphisms = {}
        
        # Create morphisms between objects of same dimension
        # These represent transitions or connections between events
        dimensions = set(obj['dimension'] for obj in self.objects.values())
        
        for dim in dimensions:
            # Get objects of this dimension
            dim_objects = [oid for oid, obj in self.objects.items() if obj['dimension'] == dim]
            
            # Create morphisms between some pairs
            for i, obj1 in enumerate(dim_objects):
                for obj2 in dim_objects[i+1:]:
                    # Only create morphisms with some probability
                    # Higher dimensions have fewer connections
                    if np.random.random() < 0.5 / (1 + dim):
                        morph_id = f"{obj1}_to_{obj2}"
                        
                        # For points, can compute a metric distance
                        if dim == 0:
                            coords1 = self.objects[obj1]['coordinates']
                            coords2 = self.objects[obj2]['coordinates']
                            distance = np.sqrt(np.sum((coords1 - coords2)**2))
                            
                            morphisms[morph_id] = {
                                'source': obj1,
                                'target': obj2,
                                'properties': {
                                    'distance': distance,
                                    'weight': np.exp(-distance)
                                }
                            }
                        else:
                            # For higher-dimensional objects, use simpler morphism
                            morphisms[morph_id] = {
                                'source': obj1,
                                'target': obj2,
                                'properties': {
                                    'weight': np.random.random()
                                }
                            }
                        
                        # Also add the reverse morphism
                        rev_morph_id = f"{obj2}_to_{obj1}"
                        morphisms[rev_morph_id] = {
                            'source': obj2,
                            'target': obj1,
                            'properties': morphisms[morph_id]['properties'].copy()
                        }
        
        # Create dimensional morphisms (between objects of different dimensions)
        # These represent inclusion maps or projections
        for high_dim in range(1, self.dim + 1):
            high_objects = [oid for oid, obj in self.objects.items() if obj['dimension'] == high_dim]
            
            for low_dim in range(high_dim):
                low_objects = [oid for oid, obj in self.objects.items() if obj['dimension'] == low_dim]
                
                # Create inclusion morphisms
                for high_obj in high_objects:
                    included = self.objects[high_obj].get('included_objects', [])
                    
                    for low_obj in included:
                        if low_obj in self.objects and self.objects[low_obj]['dimension'] == low_dim:
                            incl_morph_id = f"incl_{low_obj}_in_{high_obj}"
                            morphisms[incl_morph_id] = {
                                'source': low_obj,
                                'target': high_obj,
                                'properties': {
                                    'type': 'inclusion',
                                    'weight': 1.0
                                }
                            }
                            
                            # Also add projection morphism
                            proj_morph_id = f"proj_{high_obj}_to_{low_obj}"
                            morphisms[proj_morph_id] = {
                                'source': high_obj,
                                'target': low_obj,
                                'properties': {
                                    'type': 'projection',
                                    'weight': 1.0 / max(1, len(included))
                                }
                            }
                
        return morphisms
    
    def _initialize_two_morphisms(self):
        """Initialize 2-morphisms (morphisms between morphisms)."""
        # 2-morphisms represent transformations between paths
        # In quantum gravity, these can model curvature or gauge transformations
        
        two_morphisms = {}
        
        # Find valid pairs of morphisms with same source and target
        # These can be linked by a 2-morphism
        morph_by_endpoints = {}
        
        for mid, morph in self.morphisms.items():
            source = morph['source']
            target = morph['target']
            key = (source, target)
            
            if key not in morph_by_endpoints:
                morph_by_endpoints[key] = []
            morph_by_endpoints[key].append(mid)
        
        # Create 2-morphisms between parallel morphisms
        for (source, target), parallel_morphs in morph_by_endpoints.items():
            if len(parallel_morphs) < 2:
                continue
                
            # Create 2-morphisms between some pairs
            for i, m1 in enumerate(parallel_morphs):
                for m2 in parallel_morphs[i+1:]:
                    # Only create some 2-morphisms
                    if np.random.random() < 0.3:
                        two_morph_id = f"2m_{m1}_to_{m2}"
                        
                        # Get source and target morphisms
                        morph1 = self.morphisms[m1]
                        morph2 = self.morphisms[m2]
                        
                        # Create a 2-morphism
                        two_morphisms[two_morph_id] = {
                            'source_morphism': m1,
                            'target_morphism': m2,
                            'properties': {
                                'phase': np.random.uniform(0, 2*np.pi),  # Quantum phase
                                'amplitude': np.random.random()  # Quantum amplitude
                            }
                        }
                        
                        # Add the reverse 2-morphism
                        rev_two_morph_id = f"2m_{m2}_to_{m1}"
                        # Inverse has conjugate phase
                        phase = two_morphisms[two_morph_id]['properties']['phase']
                        two_morphisms[rev_two_morph_id] = {
                            'source_morphism': m2,
                            'target_morphism': m1,
                            'properties': {
                                'phase': (2*np.pi - phase) % (2*np.pi),
                                'amplitude': two_morphisms[two_morph_id]['properties']['amplitude']
                            }
                        }
        
        # Create 2-morphisms for composition of morphisms
        # Represent commutativity of different paths
        paths = self._find_composite_paths(max_length=2)
        
        for p1, p2 in itertools.combinations(paths, 2):
            # Check if paths have same start and end
            if p1[0] == p2[0] and p1[-1] == p2[-1]:
                # Create a 2-morphism between the composite morphisms
                comp1_id = "_then_".join(p1)
                comp2_id = "_then_".join(p2)
                
                two_morph_id = f"2m_comp_{comp1_id}_to_{comp2_id}"
                
                # Create properties based on path characteristics
                # In a full QG theory, this would encode curvature
                properties = {
                    'phase': np.random.uniform(0, 2*np.pi),
                    'amplitude': np.random.random(),
                    'is_composite': True
                }
                
                two_morphisms[two_morph_id] = {
                    'source_morphism': comp1_id,
                    'target_morphism': comp2_id,
                    'properties': properties,
                    'is_composite': True
                }
        
        return two_morphisms
    
    def _find_composite_paths(self, max_length=2):
        """Find paths of composable morphisms up to a certain length."""
        # Convert morphisms to a directed graph for path finding
        graph = nx.DiGraph()
        
        # Add nodes (objects)
        for obj_id in self.objects:
            graph.add_node(obj_id)
            
        # Add edges (morphisms)
        for morph_id, morph in self.morphisms.items():
            source = morph['source']
            target = morph['target']
            graph.add_edge(source, target, id=morph_id)
            
        # Find all paths up to max_length
        paths = []
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    # Find paths from source to target
                    for path in nx.all_simple_paths(graph, source, target, cutoff=max_length):
                        if len(path) > 1:  # At least one morphism
                            # Convert node path to morphism path
                            morph_path = []
                            for i in range(len(path) - 1):
                                edge_data = graph.get_edge_data(path[i], path[i+1])
                                morph_path.append(edge_data['id'])
                                
                            paths.append(morph_path)
        
        return paths
    
    def _create_composition_table(self):
        """Create a table for morphism composition."""
        # This represents the categorical composition operation
        # For composable morphisms f: A → B and g: B → C, we get g∘f: A → C
        
        composition_table = {}
        
        # For each pair of morphisms, check if they're composable
        for m1_id, m1 in self.morphisms.items():
            for m2_id, m2 in self.morphisms.items():
                # Composable if target of m1 is source of m2
                if m1['target'] == m2['source']:
                    # Compute properties of the composite morphism
                    composite_properties = {}
                    
                    # Handle different property types
                    if 'distance' in m1['properties'] and 'distance' in m2['properties']:
                        # For metrics, add distances
                        composite_properties['distance'] = m1['properties']['distance'] + m2['properties']['distance']
                        
                    if 'weight' in m1['properties'] and 'weight' in m2['properties']:
                        # For weights, multiply
                        composite_properties['weight'] = m1['properties']['weight'] * m2['properties']['weight']
                    
                    # Record the composition
                    composition_table[(m1_id, m2_id)] = {
                        'source': m1['source'],
                        'target': m2['target'],
                        'properties': composite_properties,
                        'component_morphisms': [m1_id, m2_id]
                    }
        
        return composition_table
    
    def _initialize_two_inner_products(self):
        """Initialize 2-inner products for 2-Hilbert space structure."""
        # 2-Hilbert spaces extend Hilbert spaces using category theory
        # The inner product becomes a functor to the category of Hilbert spaces
        
        # Here we use a simplified model where 2-inner products are defined
        # between pairs of morphisms with the same source and target
        
        two_inner_products = {}
        
        # Find morphisms with same source and target
        morph_by_endpoints = {}
        for mid, morph in self.morphisms.items():
            source = morph['source']
            target = morph['target']
            key = (source, target)
            
            if key not in morph_by_endpoints:
                morph_by_endpoints[key] = []
            morph_by_endpoints[key].append(mid)
        
        # Create 2-inner products for pairs of parallel morphisms
        for (source, target), parallel_morphs in morph_by_endpoints.items():
            for m1, m2 in itertools.product(parallel_morphs, repeat=2):
                # Inner product key
                key = (m1, m2)
                
                # For self-inner product, use real scalar
                if m1 == m2:
                    ip_value = self.morphisms[m1]['properties'].get('weight', 1.0)
                    
                # For off-diagonal, use complex value with phase
                else:
                    # Check if there's a 2-morphism between them
                    two_morph_id = f"2m_{m1}_to_{m2}"
                    if two_morph_id in self.two_morphisms:
                        # Use amplitude and phase from 2-morphism
                        amplitude = self.two_morphisms[two_morph_id]['properties']['amplitude']
                        phase = self.two_morphisms[two_morph_id]['properties']['phase']
                        
                        # Complex inner product value
                        ip_value = amplitude * np.exp(1j * phase)
                    else:
                        # No direct 2-morphism, use a small random value
                        ip_value = 0.1 * np.random.random() * np.exp(1j * np.random.uniform(0, 2*np.pi))
                        
                two_inner_products[key] = ip_value
                
        return two_inner_products
    
    def compute_morphism_composition(self, morphism1_id, morphism2_id):
        """
        Compute the composition of two morphisms.
        
        Parameters:
        -----------
        morphism1_id, morphism2_id : str
            IDs of the morphisms to compose
            
        Returns:
        --------
        dict or None
            Composite morphism data, or None if not composable
        """
        key = (morphism1_id, morphism2_id)
        if key in self.composition_table:
            return self.composition_table[key]
        return None
    
    def compute_horizontal_composition(self, two_morph1_id, two_morph2_id):
        """
        Compute horizontal composition of 2-morphisms.
        
        This represents composing transformations of paths.
        
        Parameters:
        -----------
        two_morph1_id, two_morph2_id : str
            IDs of the 2-morphisms to compose horizontally
            
        Returns:
        --------
        dict or None
            Composite 2-morphism data, or None if not composable
        """
        # Check if 2-morphisms exist
        if two_morph1_id not in self.two_morphisms or two_morph2_id not in self.two_morphisms:
            return None
            
        two_morph1 = self.two_morphisms[two_morph1_id]
        two_morph2 = self.two_morphisms[two_morph2_id]
        
        # Get component morphisms
        m1_source = two_morph1['source_morphism']
        m1_target = two_morph1['target_morphism']
        m2_source = two_morph2['source_morphism']
        m2_target = two_morph2['target_morphism']
        
        # Check if composable: m1_target should be composable with m2_source
        # and m1_source should be composable with m2_target
        if (m1_target, m2_source) not in self.composition_table or \
           (m1_source, m2_target) not in self.composition_table:
            return None
            
        # Compute horizontal composition
        # The result is a 2-morphism between the two composite morphisms
        composite_source = f"{m1_source}_then_{m2_source}"
        composite_target = f"{m1_target}_then_{m2_target}"
        
        # Compose properties - for phases, add them
        phase1 = two_morph1['properties'].get('phase', 0)
        phase2 = two_morph2['properties'].get('phase', 0)
        composite_phase = (phase1 + phase2) % (2 * np.pi)
        
        # For amplitudes, multiply them
        amplitude1 = two_morph1['properties'].get('amplitude', 1.0)
        amplitude2 = two_morph2['properties'].get('amplitude', 1.0)
        composite_amplitude = amplitude1 * amplitude2
        
        # Create the composite 2-morphism
        composite_id = f"2m_h_comp_{two_morph1_id}_{two_morph2_id}"
        return {
            'id': composite_id,
            'source_morphism': composite_source,
            'target_morphism': composite_target,
            'properties': {
                'phase': composite_phase,
                'amplitude': composite_amplitude,
                'is_horizontal_composite': True,
                'component_2morphisms': [two_morph1_id, two_morph2_id]
            }
        }
    
    def compute_vertical_composition(self, two_morph1_id, two_morph2_id):
        """
        Compute vertical composition of 2-morphisms.
        
        This represents combining sequential transformations.
        
        Parameters:
        -----------
        two_morph1_id, two_morph2_id : str
            IDs of the 2-morphisms to compose vertically
            
        Returns:
        --------
        dict or None
            Composite 2-morphism data, or None if not composable
        """
        # Check if 2-morphisms exist
        if two_morph1_id not in self.two_morphisms or two_morph2_id not in self.two_morphisms:
            return None
            
        two_morph1 = self.two_morphisms[two_morph1_id]
        two_morph2 = self.two_morphisms[two_morph2_id]
        
        # Get component morphisms
        m1_source = two_morph1['source_morphism']
        m1_target = two_morph1['target_morphism']
        m2_source = two_morph2['source_morphism']
        m2_target = two_morph2['target_morphism']
        
        # Check if vertically composable: m1_target should equal m2_source
        if m1_target != m2_source:
            return None
            
        # Compute vertical composition
        # The result is a 2-morphism from m1_source to m2_target
        
        # Compose properties - for phases, add them
        phase1 = two_morph1['properties'].get('phase', 0)
        phase2 = two_morph2['properties'].get('phase', 0)
        composite_phase = (phase1 + phase2) % (2 * np.pi)
        
        # For amplitudes, multiply them
        amplitude1 = two_morph1['properties'].get('amplitude', 1.0)
        amplitude2 = two_morph2['properties'].get('amplitude', 1.0)
        composite_amplitude = amplitude1 * amplitude2
        
        # Create the composite 2-morphism
        composite_id = f"2m_v_comp_{two_morph1_id}_{two_morph2_id}"
        return {
            'id': composite_id,
            'source_morphism': m1_source,
            'target_morphism': m2_target,
            'properties': {
                'phase': composite_phase,
                'amplitude': composite_amplitude,
                'is_vertical_composite': True,
                'component_2morphisms': [two_morph1_id, two_morph2_id]
            }
        }
    
    def compute_two_inner_product(self, morphism1_id, morphism2_id):
        """
        Compute 2-inner product between two morphisms.
        
        Parameters:
        -----------
        morphism1_id, morphism2_id : str
            IDs of the morphisms to compute inner product for
            
        Returns:
        --------
        complex
            2-inner product value, or 0 if not defined
        """
        key = (morphism1_id, morphism2_id)
        return self.two_inner_products.get(key, 0.0)
    
    def compute_exchange_law(self, two_morph_horizontal, two_morph_vertical1, two_morph_horizontal2, two_morph_vertical2):
        """
        Check the exchange law for 2-categories.
        
        The exchange law states that two different ways of composing 2-morphisms
        in a square diagram should yield the same result.
        
        Parameters:
        -----------
        two_morph_* : str
            IDs of the 2-morphisms forming a square
            
        Returns:
        --------
        bool
            True if exchange law holds, False otherwise
        """
        # First way: horizontal then vertical
        h_comp = self.compute_horizontal_composition(two_morph_horizontal, two_morph_horizontal2)
        if h_comp is None:
            return False
            
        v_comp1 = self.compute_vertical_composition(two_morph_vertical1, two_morph_vertical2)
        if v_comp1 is None:
            return False
            
        # Second way: vertical then horizontal
        v_comp2 = self.compute_vertical_composition(two_morph_horizontal, two_morph_vertical1)
        if v_comp2 is None:
            return False
            
        v_comp3 = self.compute_vertical_composition(two_morph_vertical2, two_morph_horizontal2)
        if v_comp3 is None:
            return False
            
        h_comp2 = self.compute_horizontal_composition(v_comp2['id'], v_comp3['id'])
        if h_comp2 is None:
            return False
            
        # Compare results (should be equal up to a phase)
        phase1 = h_comp['properties']['phase']
        amplitude1 = h_comp['properties']['amplitude']
        
        phase2 = h_comp2['properties']['phase']
        amplitude2 = h_comp2['properties']['amplitude']
        
        # Check if approximately equal
        phase_diff = abs((phase1 - phase2 + np.pi) % (2*np.pi) - np.pi)
        amp_ratio = abs(amplitude1 / amplitude2 - 1) if amplitude2 != 0 else float('inf')
        
        # Consider equal if within small tolerance
        return phase_diff < 0.01 and amp_ratio < 0.01
    
    def compute_2morphism_curvature(self, obj1_id, obj2_id):
        """
        Compute curvature using 2-morphisms for paths between objects.
        
        In category theory, curvature can be expressed using the failure of
        2-morphisms to satisfy certain coherence conditions.
        
        Parameters:
        -----------
        obj1_id, obj2_id : str
            IDs of objects to compute curvature between
            
        Returns:
        --------
        float
            Curvature measure between objects
        """
        # Find all morphisms between the objects
        morphisms_12 = []
        for mid, morph in self.morphisms.items():
            if morph['source'] == obj1_id and morph['target'] == obj2_id:
                morphisms_12.append(mid)
                
        if len(morphisms_12) < 2:
            return 0.0  # Not enough morphisms to compute curvature
            
        # Compute curvature as a function of 2-morphism properties
        # In a full theory, this would involve holonomy around loops
        
        # Sum phases of 2-morphisms between these morphisms
        total_phase = 0.0
        count = 0
        
        for m1, m2 in itertools.combinations(morphisms_12, 2):
            two_morph_id = f"2m_{m1}_to_{m2}"
            if two_morph_id in self.two_morphisms:
                phase = self.two_morphisms[two_morph_id]['properties']['phase']
                total_phase += phase
                count += 1
                
        # Normalized curvature
        if count > 0:
            # Normalize to [-1, 1]
            curvature = np.cos(total_phase / count)
        else:
            curvature = 0.0
            
        return curvature
    
    def compute_quantum_groupoid_structure(self):
        """
        Compute the quantum groupoid structure of the geometry.
        
        Quantum groupoids generalize symmetry in quantum spacetime,
        providing a framework for quantum reference frames.
        
        Returns:
        --------
        dict
            Quantum groupoid structure data
        """
        # In full quantum geometry, this would compute Hopf algebra-like
        # structures that generalize groups for quantum symmetry
        
        # Identify morphisms that have inverses (part of groupoid)
        invertible_morphisms = {}
        for mid, morph in self.morphisms.items():
            source = morph['source']
            target = morph['target']
            
            # Check for potential inverse morphisms
            for mid2, morph2 in self.morphisms.items():
                if morph2['source'] == target and morph2['target'] == source:
                    # Check if composition is identity-like
                    comp1 = self.compute_morphism_composition(mid, mid2)
                    comp2 = self.compute_morphism_composition(mid2, mid)
                    
                    if comp1 and comp2:
                        # These morphisms are approximate inverses
                        invertible_morphisms[mid] = {
                            'inverse': mid2,
                            'source': source,
                            'target': target
                        }
                        break
        
        # Identify isomorphism classes (orbits of the groupoid)
        orbits = {}
        visited = set()
        
        for obj_id in self.objects:
            if obj_id in visited:
                continue
                
            # Find the orbit of this object
            orbit = {obj_id}
            to_visit = [obj_id]
            
            while to_visit:
                current = to_visit.pop(0)
                
                # Find all objects connected via invertible morphisms
                for mid, inv_data in invertible_morphisms.items():
                    source = inv_data['source']
                    target = inv_data['target']
                    
                    if source == current and target not in orbit:
                        orbit.add(target)
                        to_visit.append(target)
                        
                    elif target == current and source not in orbit:
                        orbit.add(source)
                        to_visit.append(source)
            
            # Add this orbit if it's not empty
            if orbit:
                orbit_id = f"orbit_{'_'.join(sorted(list(orbit)))}"
                orbits[orbit_id] = list(orbit)
                visited.update(orbit)
                
        # Compute isotropy groups (stabilizers)
        isotropy = {}
        
        for obj_id in self.objects:
            # Find morphisms from object to itself
            stabilizer = []
            
            for mid, morph in self.morphisms.items():
                if morph['source'] == obj_id and morph['target'] == obj_id:
                    stabilizer.append(mid)
                    
            # Check if these morphisms form a group-like structure
            if stabilizer:
                isotropy[obj_id] = {
                    'morphisms': stabilizer,
                    'is_group': self._check_group_structure(stabilizer)
                }
        
        return {
            'invertible_morphisms': invertible_morphisms,
            'orbits': orbits,
            'isotropy': isotropy,
            'is_groupoid': len(invertible_morphisms) > 0
        }
    
    def _check_group_structure(self, morphisms):
        """Check if a set of morphisms forms a group-like structure."""
        # For true group, need:
        # 1. Closure under composition
        # 2. Identity element
        # 3. Inverse elements
        # 4. Associativity (automatic in category)
        
        # Check closure
        for m1 in morphisms:
            for m2 in morphisms:
                comp = self.compute_morphism_composition(m1, m2)
                if comp is None or comp['component_morphisms'] not in morphisms:
                    return False
                    
        # Check for identity-like element
        has_identity = False
        for m in morphisms:
            is_identity = True
            
            # An identity-like element should satisfy m ∘ x ≈ x and x ∘ m ≈ x
            for x in morphisms:
                comp1 = self.compute_morphism_composition(m, x)
                comp2 = self.compute_morphism_composition(x, m)
                
                if comp1 is None or comp2 is None or \
                   comp1['component_morphisms'] != [m, x] or \
                   comp2['component_morphisms'] != [x, m]:
                    is_identity = False
                    break
                    
            if is_identity:
                has_identity = True
                break
                
        if not has_identity:
            return False
            
        # Check inverses
        for m in morphisms:
            has_inverse = False
            
            for inv in morphisms:
                comp1 = self.compute_morphism_composition(m, inv)
                comp2 = self.compute_morphism_composition(inv, m)
                
                # Check if compositions are identity-like
                if comp1 and comp2 and \
                   comp1.get('is_identity', False) and \
                   comp2.get('is_identity', False):
                    has_inverse = True
                    break
                    
            if not has_inverse:
                return False
                
        return True
    
    def _initialize_three_morphisms(self):
        """Initialize 3-morphisms (morphisms between 2-morphisms)."""
        # 3-morphisms represent transformations between surface transformations
        # In quantum gravity, can model transitions between spacetime configurations
        
        three_morphisms = {}
        
        # Find pairs of 2-morphisms that can be connected
        for tm1_id, tm1 in self.two_morphisms.items():
            for tm2_id, tm2 in self.two_morphisms.items():
                # Ensure they're different 2-morphisms
                if tm1_id == tm2_id:
                    continue
                    
                # Check if they share morphisms (for simplicity)
                source1 = tm1['source_morphism']
                target1 = tm1['target_morphism']
                source2 = tm2['source_morphism']
                target2 = tm2['target_morphism']
                
                if source1 == source2 or source1 == target2 or target1 == source2 or target1 == target2:
                    # These 2-morphisms are related and can be connected by a 3-morphism
                    if np.random.random() < 0.1:  # Only create some 3-morphisms
                        three_morph_id = f"3m_{tm1_id}_to_{tm2_id}"
                        
                        # Create a 3-morphism with quantum properties
                        three_morphisms[three_morph_id] = {
                            'source_2morphism': tm1_id,
                            'target_2morphism': tm2_id,
                            'properties': {
                                'amplitude': np.random.random(),
                                'phase': np.random.uniform(0, 2*np.pi),
                                'coherence': np.random.random()  # Quantum coherence parameter
                            }
                        }
        
        return three_morphisms
    
    def _initialize_subobject_classifier(self):
        """Initialize subobject classifier for topos structure."""
        # In topos theory, the subobject classifier plays role of truth values
        # For quantum logic, these are more complex than just True/False
        
        # Create a simple subobject classifier with quantum logic structure
        omega = {
            'values': ['true', 'false', 'superposition', 'entangled'],
            'operations': {
                'and': {},  # Will store quantum AND operation
                'or': {},   # Will store quantum OR operation
                'not': {}   # Will store quantum NOT operation
            }
        }
        
        # Define operations based on quantum logic
        for v1 in omega['values']:
            # NOT operation
            if v1 == 'true':
                omega['operations']['not'][v1] = 'false'
            elif v1 == 'false':
                omega['operations']['not'][v1] = 'true'
            elif v1 == 'superposition':
                omega['operations']['not'][v1] = 'superposition'
            else:  # entangled
                omega['operations']['not'][v1] = 'entangled'
                
            # AND and OR operations
            for v2 in omega['values']:
                # AND operation (quantum)
                key = (v1, v2)
                if v1 == 'false' or v2 == 'false':
                    omega['operations']['and'][key] = 'false'
                elif v1 == 'true' and v2 == 'true':
                    omega['operations']['and'][key] = 'true'
                elif 'superposition' in (v1, v2):
                    omega['operations']['and'][key] = 'superposition'
                else:
                    omega['operations']['and'][key] = 'entangled'
                
                # OR operation (quantum)
                if v1 == 'true' or v2 == 'true':
                    omega['operations']['or'][key] = 'true'
                elif v1 == 'false' and v2 == 'false':
                    omega['operations']['or'][key] = 'false'
                elif 'superposition' in (v1, v2):
                    omega['operations']['or'][key] = 'superposition'
                else:
                    omega['operations']['or'][key] = 'entangled'
        
        return omega
    
    def _initialize_presheaves(self):
        """Initialize presheaf structure for topos approach."""
        # Presheaves are contravariant functors from category C to Set
        # They can represent local data/observables in quantum geometry
        
        presheaves = {}
        
        # Create a simple "measurement" presheaf
        # This maps objects (space regions) to sets of possible observable values
        measurement_presheaf = {
            'name': 'quantum_observables',
            'values': {}
        }
        
        # Assign observable values to each object
        for obj_id, obj in self.objects.items():
            # For 0-dimensional objects (points), assign position observables
            if obj['dimension'] == 0:
                # Create a set of possible position measurement outcomes
                values = {
                    'position': obj['coordinates'],
                    'uncertainty': np.random.random() * self.dim**(-0.5)  # Position uncertainty
                }
            else:
                # For higher-dimensional objects, assign extended observables
                values = {
                    'volume': obj['properties'].get('local_volume', 1.0),
                    'curvature': np.random.normal(0, 0.1),
                    'energy_density': np.random.exponential(1.0)
                }
                
            measurement_presheaf['values'][obj_id] = values
            
        # Define how measurements restrict when moving to subobjects
        measurement_presheaf['restriction_maps'] = {}
        
        # For each morphism, define how the presheaf restricts
        for morph_id, morph in self.morphisms.items():
            source = morph['source']
            target = morph['target']
            
            if 'type' in morph['properties'] and morph['properties']['type'] == 'inclusion':
                # For inclusion morphisms, define restriction map from larger to smaller object
                # Here we just copy values, but could implement more complex restriction
                if source in measurement_presheaf['values'] and target in measurement_presheaf['values']:
                    measurement_presheaf['restriction_maps'][morph_id] = {
                        'source': target,  # The larger object (target of inclusion)
                        'target': source,  # The smaller object (source of inclusion)
                        'map': lambda x: x  # Identity map for simplicity
                    }
        
        presheaves['measurement'] = measurement_presheaf
        
        return presheaves
    
    def compute_n_morphism_composition(self, n, morphism1_id, morphism2_id):
        """
        Compute the composition of two n-morphisms.
        
        Parameters:
        -----------
        n : int
            Level of morphisms (1=morphisms, 2=2-morphisms, etc.)
        morphism1_id, morphism2_id : str
            IDs of the n-morphisms to compose
            
        Returns:
        --------
        dict or None
            Composite n-morphism data, or None if not composable
        """
        if n == 1:
            return self.compute_morphism_composition(morphism1_id, morphism2_id)
        elif n == 2:
            # Vertical composition of 2-morphisms
            return self.compute_vertical_composition(morphism1_id, morphism2_id)
        elif n == 3:
            # Composition of 3-morphisms
            # Check if 3-morphisms exist
            if (morphism1_id not in self.three_morphisms or 
                morphism2_id not in self.three_morphisms):
                return None
                
            morph1 = self.three_morphisms[morphism1_id]
            morph2 = self.three_morphisms[morphism2_id]
            
            # Check if composable (target of 1st = source of 2nd)
            if morph1['target_2morphism'] != morph2['source_2morphism']:
                return None
                
            # Create composite 3-morphism
            composite_id = f"3m_comp_{morphism1_id}_{morphism2_id}"
            
            # Combine properties
            phase1 = morph1['properties']['phase']
            phase2 = morph2['properties']['phase']
            composite_phase = (phase1 + phase2) % (2 * np.pi)
            
            amplitude1 = morph1['properties']['amplitude']
            amplitude2 = morph2['properties']['amplitude']
            composite_amplitude = amplitude1 * amplitude2
            
            # Combine coherence (simplified model)
            coherence1 = morph1['properties']['coherence']
            coherence2 = morph2['properties']['coherence']
            composite_coherence = coherence1 * coherence2
            
            return {
                'id': composite_id,
                'source_2morphism': morph1['source_2morphism'],
                'target_2morphism': morph2['target_2morphism'],
                'properties': {
                    'phase': composite_phase,
                    'amplitude': composite_amplitude,
                    'coherence': composite_coherence,
                    'is_composite': True
                }
            }
        else:
            # Higher n not yet implemented
            return None
            
    def evaluate_topos_logic(self, statement, context=None):
        """
        Evaluate a logical statement in the quantum topos logic.
        
        Parameters:
        -----------
        statement : dict
            Dictionary describing the logical statement
        context : str, optional
            Object ID providing context for evaluation
            
        Returns:
        --------
        str
            Truth value in the quantum logic subobject classifier
        """
        if 'type' not in statement:
            return 'false'  # Invalid statement
            
        omega = self.subobject_classifier
        
        if statement['type'] == 'atomic':
            # Atomic proposition about an object
            if 'object' not in statement or 'property' not in statement:
                return 'false'
                
            obj_id = statement['object']
            prop = statement['property']
            
            if obj_id not in self.objects:
                return 'false'
                
            # Evaluate based on object properties
            if prop == 'exists':
                return 'true'
            elif prop == 'is_point' and self.objects[obj_id]['dimension'] == 0:
                return 'true'
            elif prop == 'is_region' and self.objects[obj_id]['dimension'] > 0:
                return 'true'
            elif prop == 'is_quantum' and np.random.random() < 0.5:
                # Simulate quantum uncertainty
                return 'superposition'
            else:
                return 'false'
                
        elif statement['type'] == 'not':
            # Negation
            if 'statement' not in statement:
                return 'false'
                
            inner_value = self.evaluate_topos_logic(statement['statement'], context)
            return omega['operations']['not'][inner_value]
            
        elif statement['type'] == 'and':
            # Conjunction
            if 'statements' not in statement or len(statement['statements']) < 2:
                return 'false'
                
            # Evaluate first two statements
            value1 = self.evaluate_topos_logic(statement['statements'][0], context)
            value2 = self.evaluate_topos_logic(statement['statements'][1], context)
            result = omega['operations']['and'][(value1, value2)]
            
            # Process any additional statements
            for i in range(2, len(statement['statements'])):
                value_next = self.evaluate_topos_logic(statement['statements'][i], context)
                result = omega['operations']['and'][(result, value_next)]
                
            return result
            
        elif statement['type'] == 'or':
            # Disjunction
            if 'statements' not in statement or len(statement['statements']) < 2:
                return 'false'
                
            # Evaluate first two statements
            value1 = self.evaluate_topos_logic(statement['statements'][0], context)
            value2 = self.evaluate_topos_logic(statement['statements'][1], context)
            result = omega['operations']['or'][(value1, value2)]
            
            # Process any additional statements
            for i in range(2, len(statement['statements'])):
                value_next = self.evaluate_topos_logic(statement['statements'][i], context)
                result = omega['operations']['or'][(result, value_next)]
                
            return result
            
        elif statement['type'] == 'entangled':
            # Entanglement statement connecting two objects
            if 'objects' not in statement or len(statement['objects']) != 2:
                return 'false'
                
            obj1, obj2 = statement['objects']
            if obj1 not in self.objects or obj2 not in self.objects:
                return 'false'
                
            # Check if objects are entangled
            # In a real implementation, would compute this from quantum state
            # Here we simulate with basic probabilistic model
            if np.random.random() < 0.3:
                return 'entangled'
            else:
                return 'false'
                
        else:
            return 'false'  # Unknown statement type

    def sheaf_cohomology(self, presheaf_name):
        """
        Compute (simplified) sheaf cohomology for a presheaf.
        
        In topos theory, sheaf cohomology gives global information
        from local data, important for quantum geometry.
        
        Parameters:
        -----------
        presheaf_name : str
            Name of the presheaf to compute cohomology for
            
        Returns:
        --------
        dict
            Simplified cohomology data
        """
        if presheaf_name not in self.presheaves:
            return {'error': 'Presheaf not found'}
            
        presheaf = self.presheaves[presheaf_name]
        
        # For a real implementation, would compute Čech cohomology
        # Here we use a simplified model
        
        # Group objects by dimension
        dim_objects = {}
        for obj_id, obj in self.objects.items():
            dim = obj['dimension']
            if dim not in dim_objects:
                dim_objects[dim] = []
            dim_objects[dim].append(obj_id)
            
        # Simplified cohomology computation
        # H^0: Global sections (consistent data across all objects)
        # H^1: Obstructions to extending local sections
        
        h0_dim = 1  # Dimension of H^0 (simplified)
        h1_data = {}
        
        # Check consistency of data across objects
        for dim in sorted(dim_objects.keys()):
            if dim + 1 in dim_objects:
                # Look at objects and their "covers" at next dimension
                for high_obj_id in dim_objects[dim + 1]:
                    # Find lower dim objects included in this one
                    included = []
                    for low_obj_id in dim_objects[dim]:
                        # Check if there's an inclusion morphism
                        for morph_id, morph in self.morphisms.items():
                            if (morph['source'] == low_obj_id and 
                                morph['target'] == high_obj_id and
                                morph['properties'].get('type') == 'inclusion'):
                                included.append(low_obj_id)
                                break
                    
                    # Check for consistency in data assignment
                    if len(included) > 1:
                        consistent = True
                        for i in range(len(included)-1):
                            # Compare values in a simplified way
                            val1 = presheaf['values'].get(included[i], {})
                            val2 = presheaf['values'].get(included[i+1], {})
                            
                            # Check if values are roughly consistent
                            if val1 and val2:
                                # Just check if they have the same keys
                                if set(val1.keys()) != set(val2.keys()):
                                    consistent = False
                                    break
                        
                        if not consistent:
                            h1_data[high_obj_id] = "Inconsistent data on lower-dim objects"
        
        # H^1 dimension is roughly the number of inconsistencies
        h1_dim = len(h1_data)
        
        return {
            'H^0': {'dimension': h0_dim},
            'H^1': {'dimension': h1_dim, 'data': h1_data},
            'euler_characteristic': h0_dim - h1_dim
        }


if __name__ == "__main__":
    # Test the category theory geometry
    print("Testing Category Theory Geometry")
    ctg = CategoryTheoryGeometry(dim=3, n_points=20)
    
    # Print some basic information
    print(f"Number of objects: {len(ctg.objects)}")
    print(f"Number of morphisms: {len(ctg.morphisms)}")
    print(f"Number of 2-morphisms: {len(ctg.two_morphisms)}")
    
    # Test composition
    if ctg.morphisms:
        m_ids = list(ctg.morphisms.keys())
        if len(m_ids) >= 2:
            m1, m2 = m_ids[0], m_ids[1]
            comp = ctg.compute_morphism_composition(m1, m2)
            print(f"\nComposition of {m1} and {m2}:")
            print(f"Composable: {comp is not None}")
            
    # Test quantum groupoid structure
    qg_structure = ctg.compute_quantum_groupoid_structure()
    print(f"\nQuantum groupoid structure:")
    print(f"Number of invertible morphisms: {len(qg_structure['invertible_morphisms'])}")
    print(f"Number of orbits: {len(qg_structure['orbits'])}")
    print(f"Is groupoid: {qg_structure['is_groupoid']}") 