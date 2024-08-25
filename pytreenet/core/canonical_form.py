"""
This module is concerned with the canonical form of a TreeTensorNetwork.

The canonical form of a TreeTensorNetwork is a specific choice of the gauge
freedom causing all tensors apart from the orthogonality center to be a
equivalent to an isometry. This is achieved by performing a series of QR
decompositions on the tensors of the network.
"""
from __future__ import annotations
from typing import Tuple
from uuid import uuid1

from copy import copy,deepcopy

import numpy as np
from .leg_specification import LegSpecification
from .node import Node
from ..util.tensor_splitting import SplitMode
from ..util.tensor_splitting import (SVDParameters , tensor_qr_decomposition, truncated_tensor_svd, ContractionMode)
from ..util import compute_transfer_tensor
from ..core.graph_node import GraphNode

def canonical_form(ttn: TreeTensorNetwork,
                   orthogonality_center_id: str,
                   mode: SplitMode = SplitMode.REDUCED,
                   contr_mode: ContractionMode = ContractionMode.VCONTR):
    """
    Modifies a TreeTensorNetwork into canonical form.

    Args:
        ttn (TreeTensorNetwork): The TTN for which to be transformed into
            canonical form.
        orthogonality_center_id (str): The identifier of the tensor node which
            is the orthogonality center for the canonical form.
        mode: The mode to be used for the QR decomposition. For details refe
            to `tensor_util.tensor_qr_decomposition`.
    """
    distance_dict = ttn.distance_to_node(
        orthogonality_center_id)
    maximum_distance = max(distance_dict.values())
    # Perform QR-decomposition on all TensorNodes but the orthogonality center
    for distance in reversed(range(1, maximum_distance+1)):
        # Perform QR on nodes furthest away first.
        node_id_with_distance = [node_id for node_id in distance_dict.keys()
                                 if distance_dict[node_id] == distance]
        for node_id in node_id_with_distance:
            node = ttn.nodes[node_id]
            minimum_distance_neighbour_id = _find_smallest_distance_neighbour(node,
                                                                              distance_dict)
            if isinstance(mode,SplitMode):  
                split_qr_contract_r_to_neighbour(ttn,
                                             node_id,
                                             minimum_distance_neighbour_id,
                                             mode=mode)
            elif isinstance(mode,SVDParameters):
                split_svd_contract_sv_to_neighbour(ttn = ttn,
                                                  node_id = node_id,
                                                  neighbour_id = minimum_distance_neighbour_id,
                                                  SVDParameters = mode,
                                                  contr_mode = contr_mode)
    ttn.orthogonality_center_id = orthogonality_center_id

def adjust_ttn1_structure_to_ttn2(ttn1, ttn2):
    """
    Adjusts the structure of ttn1 to match the structure of ttn2.

    Args:
        ttn1 (TTN): The original Tensor Train Network.
        ttn2 (TTN): The target Tensor Train Network.

    Returns:
        ttn3 (TTN): The adjusted ttn1 with the structure of ttn2.
    """
    ttn3 = deepcopy(ttn2)
    orth_center = ttn1.orthogonality_center_id
    for node_id in ttn3.nodes:
        ttn1_neighbours = ttn1.nodes[node_id].neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(ttn1_neighbours)}
        ttn1_neighbours = ttn2.nodes[node_id].neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
        nneighbours = ttn2.nodes[node_id].nneighbours()
        ttn1_tensor = ttn1.tensors[node_id].transpose(permutation + (nneighbours,))
        ttn3.tensors[node_id] = ttn1_tensor
        ttn3.nodes[node_id].link_tensor(ttn1_tensor)
    ttn3.orthogonality_center_id = orth_center    
    return ttn3    

def adjust_ttno_structure_to_ttn(ttno, ttn):
    ttno2 = deepcopy(ttno)

    for node_id in ttno2.nodes:
        node_id = "Site(0,0)"
        ttno_neighbours = ttno.nodes[node_id].neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(ttno_neighbours)}
        ttn1_neighbours = ttn.nodes[node_id].neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
        nneighbours = ttn.nodes[node_id].nneighbours()
        ttno_tensor = ttno.tensors[node_id].transpose(permutation + (nneighbours,) + (nneighbours + 1,))
        ttno2.tensors[node_id] = ttno_tensor
        ttno2.nodes[node_id].link_tensor(ttno_tensor)
        ttn_neighbours = ttn.nodes[node_id].neighbouring_nodes()
        if ttno2.nodes[node_id].is_root():
            ttno2.nodes[node_id].children = ttn_neighbours
        else:
            ttno2.nodes[node_id].children = ttn_neighbours[1:] 
    return ttno2       

def _find_smallest_distance_neighbour(node: Node,
                                      distance_dict: dict[str, int]) -> str:
    """
    Finds the neighbour of a node with the smallest distance to the center node.
    
    Args:
        node (Node): The node for which to search the neighbours.
        distance_dict (dict[str,int]): A dictionary with the distance of every
            node to the orthogonality center.

    Returns:
        str: The identifier of the neighbour of the node with the smallest
            distance to the orthogonality center.
    """
    neighbour_ids = node.neighbouring_nodes()
    neighbour_distance_dict = {neighbour_id: distance_dict[neighbour_id]
                               for neighbour_id in neighbour_ids}
    minimum_distance_neighbour_id = min(neighbour_distance_dict,
                                        key=neighbour_distance_dict.get)
    return minimum_distance_neighbour_id

def split_qr_contract_r_to_neighbour(ttn: TreeTensorNetwork,
                                     node_id: str,
                                     neighbour_id: str,
                                     mode: SplitMode = SplitMode.REDUCED):
    """
    Takes a node an splits of the virtual leg to a neighbours via QR
     decomposition. The resulting R tensor is contracted with the neighbour.::

         __|__      __|__        __|__      __      __|__
      __|  N1 |____|  N2 | ---> | N1' |____|__|____|  N2 |
        |_____|    |_____|      |_____|            |_____|

                __|__      __|__ 
      --->   __| N1' |____| N2' |
               |_____|    |_____|

    Args:
        ttn (TreeTensorNetwork): The tree tensor network in which to perform
            this action.
        node_id (str): The identifier of the node to be split.
        neighbour_id (str): The identifier of the neigbour to which to split.
        mode: The mode to be used for the QR decomposition. For details refer to
            `tensor_util.tensor_qr_decomposition`.
    """
    node = ttn.nodes[node_id]
    q_legs, r_legs = _build_qr_leg_specs(node, neighbour_id)
    r_tensor_id = str(uuid1()) # Avoid identifier duplication
    ttn.split_node_qr(node_id, q_legs, r_legs,
                        q_identifier=node_id,
                        r_identifier=r_tensor_id,
                        mode=mode)
    ttn.contract_nodes(neighbour_id, r_tensor_id,
                        new_identifier=neighbour_id)

def split_svd_contract_sv_to_neighbour(ttn: TreeTensorNetwork,
                                     node_id: str,
                                     neighbour_id: str,
                                     SVDParameters,
                                     contr_mode):

    node = ttn.nodes[node_id]
    u_legs, v_legs = _build_qr_leg_specs(node, neighbour_id)
    r_tensor_id = str(uuid1()) # Avoid identifier duplication
    ttn.split_node_svd(node_id , u_legs, v_legs,
                       svd_params = SVDParameters,
                       u_identifier = node_id, v_identifier = r_tensor_id,
                       contr_mode = contr_mode)
    ttn.contract_nodes(neighbour_id, r_tensor_id,
                        new_identifier=neighbour_id) 

def _build_qr_leg_specs(node: Node,
                        min_neighbour_id: str) -> Tuple[LegSpecification,LegSpecification]:
    """
    Construct the leg specifications required for the qr decompositions during
     canonicalisation.

    Args:
        node (Node): The node which is to be split.
        min_neighbour_id (str): The identifier of the neighbour of the node
         which is closest to the orthogonality center.

    Returns:
        Tuple[LegSpecification,LegSpecification]: 
            The leg specifications for the legs of the Q-tensor, i.e. what
            remains as the node, and the R-tensor, i.e. what will be absorbed
            into the node defined by `min_neighbour_id`.
    """
    q_legs = LegSpecification(None, copy(node.children), node.open_legs)
    if node.is_child_of(min_neighbour_id):
        r_legs = LegSpecification(min_neighbour_id, [], [])
    else:
        q_legs.parent_leg = node.parent
        q_legs.child_legs.remove(min_neighbour_id)
        r_legs = LegSpecification(None, [min_neighbour_id], [])
    if node.is_root():
        q_legs.is_root = True
    return q_legs, r_legs
