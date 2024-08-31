"""
This module provides functions to contract a TTNS with a TTNO
"""

from __future__ import annotations

import numpy as np

from ..core.node import Node
from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import (contract_all_but_one_neighbour_block_to_ket,
                               contract_all_neighbour_blocks_to_ket,
                               get_equivalent_legs,
                               contract_all_neighbour_blocks_to_ket_Lindblad)
from copy import deepcopy

__all__ = ['expectation_value']

def expectation_value(state: TreeTensorNetworkState,
                      operator: TTNO) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Operator.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = state.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             state, operator,
                             dictionary)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    return complex(contract_node_with_environment(state.root_id,
                                                  state, operator,
                                                  dictionary))

#def get_equivalent_legs_Lindblad(node1,
#                                 node2,
#                                 ignore_legs):
    """
    Get the equivalent legs of two nodes. This is useful when contracting
     two nodes with equal neighbour identifiers, that may potentially be in
     different orders. Some neighbours may also be ignored.
    
    Args:
        node1 (Node): The first node.
        node2 (Node): The second node.
        ignore_legs (Union[None,List[str],str]): The legs to ignore.
    
    Returns:
        Tuple[List[int],List[int]]: The equivalent legs of the two nodes. This
            means the indeces of legs to the same neighbour are at the same
            position in each list.
    """
    if ignore_legs is None:
        ignore_legs = []
    elif isinstance(ignore_legs, str):
        ignore_legs = [ignore_legs]
    legs1 = []
    legs2 = []
    
    if node1.is_root():
        neighbouring_nodes = node1.neighbouring_nodes()[1:]
        for neighbour_id1 in neighbouring_nodes:
            i, j = neighbour_id1.replace('Site(', '').replace(')', '').split(',')
            neighbour_id2 = f"Node({i},{j})"

            if neighbour_id1 in ignore_legs:
                continue
            legs1.append(node1.neighbour_index(neighbour_id1))
            legs2.append(node2.neighbour_index(neighbour_id2))
    else:
        for neighbour_id1 in node1.neighbouring_nodes():
            i, j = neighbour_id1.replace('Site(', '').replace(')', '').split(',')
            neighbour_id2 = f"Node({i},{j})"
            if neighbour_id1 in ignore_legs:
                continue
            legs1.append(node1.neighbour_index(neighbour_id1))
            legs2.append(node2.neighbour_index(neighbour_id2))
    return legs1, legs2

# def check_bra_ket_compatibility(state: TreeTensorNetworkState):
    for ket_id in list(state.nodes.keys())[0:len(state.nodes)//2]:
        ket_node = state.nodes[ket_id]
        i, j = ket_id.replace('Site(', '').replace(')', '').split(',')
        bra_id = f"Node({i},{j})"
        bra_node = state.nodes[bra_id]
        legs1, legs2 = get_equivalent_legs_Lindblad(ket_node, bra_node, [])
        assert legs1 == legs2,  f"Legs mismatch: {legs1} != {legs2} for ket_id {ket_id} and bra_id {bra_id}"

#def adjust_bra_ket(state: TreeTensorNetworkState):
    for ket_id in list(state.nodes.keys())[0:len(state.nodes)//2]:
        i, j = ket_id.replace('Site(', '').replace(')', '').split(',')
        bra_id = f"Node({i},{j})"
        if state.nodes[ket_id].is_root():
            perm = list(range(state.tensors[ket_id].ndim ))
            n = state.nodes[ket_id].neighbour_index(bra_id)
            perm.pop(n)
            perm.insert(0,n)
            perm = tuple(perm)
            T = np.transpose(state.tensors[ket_id], perm)

            state.tensors[ket_id] = T
            state.nodes[ket_id].link_tensor(T)
            
            perm = list(range(state.tensors[ket_id].ndim - 1))
            n = state.nodes[ket_id].neighbour_index(bra_id)
            perm.pop(n)
            perm.insert(0,n)
            bra_children = np.array(state.nodes[ket_id].children)
            
            state.nodes[ket_id].children = bra_children[perm].tolist()


        ket_node = state.nodes[ket_id]
        bra_node = state.nodes[bra_id]
        
        if state.nodes[ket_id].is_root():
            legs1, legs2 = get_equivalent_legs_Lindblad(ket_node, bra_node, [])
            perm = (0,) + tuple([x for x in legs2]) + tuple((len(legs1)+1,))
        else:
            legs1, legs2 = get_equivalent_legs_Lindblad(ket_node, bra_node, [])            
            perm = tuple(legs2) + tuple((len(legs2),))
        
        T = np.transpose(state.tensors[bra_id], perm)
        state.tensors[bra_id] = T
        state.nodes[bra_id].link_tensor(T)  

        children = []
        for child_id in ket_node.children:
            i, j = child_id.replace('Site(', '').replace(')', '').split(',')
            children.append(f"Node({i},{j})")

        if state.nodes[ket_id].is_root():
            state.nodes[bra_id].children = children[1:]
        else:
            state.nodes[bra_id].children = children      



def transpose_node_with_neighbouring_nodes(state, ket_id, neighbours):
    perm = []
    for neighbour in neighbours: 
        n = state.nodes[ket_id].neighbour_index(neighbour)
        perm.append(n)    
    if state.nodes[ket_id].nopen_legs() == 1:    
       perm = tuple(perm) + (len(perm),)
    elif state.nodes[ket_id].nopen_legs() == 2:
        perm = tuple(perm) + (len(perm), len(perm) + 1)   
    T = np.transpose(state.tensors[ket_id], perm)
    state.tensors[ket_id] = T
    state.nodes[ket_id].link_tensor(T)
    if state.nodes[ket_id].is_root():
        state.nodes[ket_id].children = neighbours
    else:
        state.nodes[ket_id].children = neighbours[1:]    

def adjust_operator_to_ket(operator,state):
    for ket_id in list(state.nodes.keys())[0:len(state.nodes)//2]:
        if operator.nodes[ket_id].is_root():
            i, j = ket_id.replace('Site(', '').replace(')', '').split(',')
            bra_id = f"Node({i},{j})"

            perm = list(range(state.tensors[ket_id].ndim - 1))
            n = state.nodes[ket_id].neighbour_index(bra_id)
            perm.pop(n)
            perm.insert(0,n)
            neighbours = np.array(state.nodes[ket_id].children)
            neighbours = np.array(state.nodes[ket_id].neighbouring_nodes())
            neighbours = neighbours[perm].tolist()
            transpose_node_with_neighbouring_nodes(state, ket_id, neighbours)
            transpose_node_with_neighbouring_nodes(operator, ket_id, neighbours)
        else:    
            neighbours = state.nodes[ket_id].neighbouring_nodes()
            transpose_node_with_neighbouring_nodes(operator, ket_id, neighbours)    
    
def convert_sites_and_nodes(input_list):
    converted_list = []
    
    for item in input_list:
        if item.startswith("Site"):
            converted_item = item.replace("Site", "Node")
        elif item.startswith("Node"):
            converted_item = item.replace("Node", "Site")
        else:
            converted_item = item
        
        converted_list.append(converted_item)
    
    return converted_list

def adjust_bra_to_ket(state):
    for ket_id in list(state.nodes.keys())[0:len(state.nodes)//2]:
        if state.nodes[ket_id].is_root():
            i, j = ket_id.replace('Site(', '').replace(')', '').split(',')
            bra_id = f"Node({i},{j})"

            perm = list(range(state.tensors[ket_id].ndim - 1))
            n = state.nodes[ket_id].neighbour_index(bra_id)
            perm.pop(n)
            perm.insert(0,n)
            neighbours = np.array(state.nodes[ket_id].neighbouring_nodes())
            neighbours = neighbours[perm].tolist()
            transpose_node_with_neighbouring_nodes(state, ket_id, neighbours)
            neighbours = convert_sites_and_nodes(neighbours)
            transpose_node_with_neighbouring_nodes(state, bra_id, neighbours)
        else:
            i, j = ket_id.replace('Site(', '').replace(')', '').split(',')
            bra_id = f"Node({i},{j})"                
            neighbours = state.nodes[ket_id].neighbouring_nodes()
            neighbours = convert_sites_and_nodes(neighbours)
            transpose_node_with_neighbouring_nodes(state, bra_id, neighbours) 

def expectation_value_Lindblad(ttn: TreeTensorNetworkState,
                               ttno: TTNO) -> complex:
    state = deepcopy(ttn)
    operator = deepcopy(ttno)
    adjust_operator_to_ket(operator,state)
    adjust_bra_to_ket(state)
    dict = contract_all_except_root(state, operator)
    #adjust_operator_to_ket(operator,state)
    #adjust_bra_to_ket(state)
    ket_node, ket_tensor = state[state.root_id]
    bra_id = state.nodes[state.root_id].neighbouring_nodes()[0]
    _ , bra_tensor = state[bra_id]
    ket_neigh_block = contract_all_neighbour_blocks_to_ket_Lindblad(ket_tensor,
                                                            ket_node,
                                                            dict)
    perm = list(range(ket_neigh_block.ndim)) 
    perm.append(perm.pop(1))
    ket_neigh_block = ket_neigh_block.transpose(perm)

    op_node , op_tensor = operator[state.root_id]
    op_neighbours = operator.nodes[state.root_id].neighbouring_nodes()
    ket_node, ket_tensor = state[state.root_id]
    ket_neighbours = state.nodes[state.root_id].neighbouring_nodes()

    element_map = {elem: i for i, elem in enumerate(op_neighbours)}
    permutation = tuple(element_map[elem] for elem in ket_neighbours)
    nneighbours = state.nodes[state.root_id].nneighbours()
    permutation = permutation + (len(permutation), len(permutation)+1)
    op_tensor = np.transpose(op_tensor, permutation)

    shape = list(op_tensor.shape)
    shape.pop(0)
    shape = tuple(shape)
    op_tensor = np.reshape(op_tensor, shape)
    
    ham_legs = tuple(range(0,nneighbours-1 )) + (_node_operator_input_leg(op_node)-1,)
    #block_legs = tuple(2*i+1 for i in range(nneighbours-1)) + (ket_neigh_block.ndim - 1,)
    block_legs = (1,3,5)
    kethamblock = np.tensordot(ket_neigh_block, op_tensor,
                            axes=(block_legs, ham_legs))
    state_legs = tuple(range(0, nneighbours+1))

    return np.tensordot(bra_tensor, kethamblock,
                        axes=(state_legs,state_legs))    



def contract_all_except_root(state: ptn.TreeTensorNetworkState,
                             operator: ptn.TTNO) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Operator.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == state.root_id, errstr
    # assert computation_order[-1] == operator.root_id, errstr
    computation_order = computation_order[:-1]
    for i, node_id1 in enumerate(computation_order[len(state.nodes) // 2:]):  # The last one is the root node
        node_id2 = computation_order[i]
        node = state.nodes[node_id1]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any_Lindblad(node_id1, node_id2, parent_id,
                              state, operator,
                              dictionary)
        dictionary.add_entry(node_id1, parent_id, block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id, node_id1)
    return dictionary

def contract_node_with_environment(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TTNO,
                                   dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

                            ______
                 _____     |      |      _____
                |     |____|  A*  |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    ket_node, ket_tensor = state[node_id]
    ket_neigh_block = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    op_node, op_tensor = operator[node_id]
    state_legs, ham_legs = get_equivalent_legs(ket_node, op_node)
    ham_legs.append(_node_operator_input_leg(op_node))
    block_legs = list(range(1,2*ket_node.nneighbours(),2))
    block_legs.append(0)
    kethamblock = np.tensordot(ket_neigh_block, op_tensor,
                               axes=(block_legs, ham_legs))
    bra_tensor = ket_tensor.conj()
    state_legs.append(len(state_legs))
    return np.tensordot(bra_tensor, kethamblock,
                        axes=(state_legs,state_legs))

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TTNO,
                 dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts any node. 
    
    Rather the entire subtree starting from the node is contracted. The
    subtrees below the node already have to be contracted, except for the
    specified neighbour.
    This function combines the two options of contracting a leaf node or
    a general node using the dictionary in one function.
    
    Args:
        node_id (str): Identifier of the node.
        next_node_id (str): Identifier of the node towards which the open
            legs will point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        
    Returns:
        np.ndarray: The contracted tensor.
    """
    node = state.nodes[node_id]
    if node.is_leaf():
        return contract_leaf(node_id, state, operator)
    return contract_subtrees_using_dictionary(node_id,
                                              next_node_id,
                                              state,
                                              operator,
                                              dictionary)

def contract_any_Lindblad(node_id1: str,
                         node_id2: str, 
                         next_node_id: str,
                         state: TreeTensorNetworkState,
                         operator: TTNO,
                         dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts any node. 
    
    Rather the entire subtree starting from the node is contracted. The
    subtrees below the node already have to be contracted, except for the
    specified neighbour.
    This function combines the two options of contracting a leaf node or
    a general node using the dictionary in one function.
    
    Args:
        node_id (str): Identifier of the node.
        next_node_id (str): Identifier of the node towards which the open
            legs will point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        
    Returns:
        np.ndarray: The contracted tensor.
    """
    node = state.nodes[node_id1]
    if node.is_leaf():
        return contract_leaf_Lindblad(node_id1,node_id2, state, operator)
    return contract_subtrees_using_dictionary_Lindblad(node_id1,
                                                       node_id2,
                                                       next_node_id,
                                                       state,
                                                       operator,
                                                       dictionary)

def contract_leaf(node_id: str,
                  state: TreeTensorNetworkState,
                  operator: TTNO) -> np.ndarray:
    """
    Contracts for a leaf node the state, operator and conjugate state tensors.

    If the current subtree starts at a leaf, only the three tensors
    corresponding to that site must be contracted. Furthermore, the retained
    legs must point towards the leaf's parent.

    Args:
        node_id (str): Identifier of the leaf node
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.

    Returns:
        np.ndarray: The contracted partial tree::
    
                     _____
           2    ____|     |
                    |  A* |
                    |_____|
                       |
                       |1
                     __|__
           1    ____|     |
                  0 |  H  |
                    |_____|
                       |2
                       |
                     __|__
           0    ____|     |
                    |  A  |
                    |_____|
        
    """
    ket_node, ket_tensor = state[node_id]
    bra_tensor = ket_tensor.conj()
    ham_node, ham_tensor = operator[node_id]
    bra_ham = np.tensordot(ham_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(ham_node),
                                 _node_state_phys_leg(ket_node)))
    bra_ham_ket = np.tensordot(ket_tensor, bra_ham,
                               axes=(_node_state_phys_leg(ket_node),
                                     _node_operator_input_leg(ham_node)-1))
    return bra_ham_ket


def contract_leaf_Lindblad(node_id1: str,
                           node_id2: str,
                           state: TreeTensorNetworkState,
                           operator: TTNO) -> np.ndarray:
    """
    Contracts for a leaf node the state, operator and conjugate state tensors.

    If the current subtree starts at a leaf, only the three tensors
    corresponding to that site must be contracted. Furthermore, the retained
    legs must point towards the leaf's parent.

    Args:
        node_id (str): Identifier of the leaf node
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.

    Returns:
        np.ndarray: The contracted partial tree::
    
                     _____
           2    ____|     |
                    |  A* |
                    |_____|
                       |
                       |1
                     __|__
           1    ____|     |
                  0 |  H  |
                    |_____|
                       |2
                       |
                     __|__
           0    ____|     |
                    |  A  |
                    |_____|
        
    """
    ket_node, ket_tensor = state[node_id1]
    bra_tensor = state.tensors[node_id2]
    ham_node, ham_tensor = operator[node_id1]
    bra_ham = np.tensordot(ham_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(ham_node),
                                 _node_state_phys_leg(ket_node)))
    bra_ham_ket = np.tensordot(ket_tensor, bra_ham,
                               axes=(_node_state_phys_leg(ket_node),
                                     _node_operator_input_leg(ham_node)-1))
    return bra_ham_ket


def contract_subtrees_using_dictionary(node_id: str, ignored_node_id: str,
                                       state: TreeTensorNetworkState,
                                       operator: TTNO,
                                       dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards.

    Args:
        node_id (str): Identifier of the node.
        ignored_node_id (str): Identifier of the node to which the remaining
            legs should point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the operator.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The contracted tensor::

                     _____      ______
              2 ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    
    """
    ket_node, ket_tensor = state[node_id]
    tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                         ket_node,
                                                         ignored_node_id,
                                                         dictionary)
    op_node, op_tensor = operator[node_id]
    tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                       ket_node,
                                                       op_tensor,
                                                       op_node,
                                                       ignored_node_id)
    bra_tensor = ket_tensor.conj()
    return contract_bra_tensor_ignore_one_leg(bra_tensor,
                                              tensor,
                                              ket_node,
                                              ignored_node_id)

def contract_subtrees_using_dictionary_Lindblad(node_id1: str,
                                       node_id2: str,
                                       ignored_node_id: str,
                                       state: TreeTensorNetworkState,
                                       operator: TTNO,
                                       dictionary: PartialTreeCachDict) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards.

    Args:
        node_id (str): Identifier of the node.
        ignored_node_id (str): Identifier of the node to which the remaining
            legs should point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the operator.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.

    Returns:
        np.ndarray: The contracted tensor::

                     _____      ______
              2 ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    
    """
    ket_node, ket_tensor = state[node_id1]
    tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                         ket_node,
                                                         ignored_node_id,
                                                         dictionary)
    op_node, op_tensor = operator[node_id1]
    tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                       ket_node,
                                                       op_tensor,
                                                       op_node,
                                                       ignored_node_id)
    bra_tensor = state.tensors[node_id2]
    return contract_bra_tensor_ignore_one_leg(bra_tensor,
                                              tensor,
                                              ket_node,
                                              ignored_node_id)

def contract_operator_tensor_ignoring_one_leg(current_tensor: np.ndarray,
                                              ket_node: Node,
                                              op_tensor: np.ndarray,
                                              op_node: Node,
                                              ignoring_node_id: str) -> np.ndarray:
    """
    Contracts the operator tensor with the current tensor.

    The current tensor is the ket tensor of this node to which all but
    one neighbour blocks are already contracted. The blocks are the already
    contracted subtrees starting from this node. The subtree that is not
    contracted is the one that the remaining legs point towards.
    
    Args:
        current_tensor (np.ndarray): The current tensor.
        ket_node (Node): The ket node.
        op_tensor (np.ndarray): The operator tensor.
        op_node (Node): The operator node.
        ignoring_node_id (str): The identifier of the node to which the
            virtual leg should not point.

    Returns:
        np.ndarray: The contracted tensor::
    
                                    ______
                                   |      |
                            _______|      |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|
    
    """
    _, op_legs = get_equivalent_legs(ket_node, op_node, [ignoring_node_id])
    # Due to the legs to the bra tensor, the legs of the current tensor are a
    # bit more complicated
    tensor_legs = list(range(2,2*ket_node.nneighbours(),2))
    # Adding the physical legs
    tensor_legs.append(1)
    op_legs.append(_node_operator_input_leg(op_node))
    return np.tensordot(current_tensor, op_tensor,
                        axes=(tensor_legs, op_legs))

def contract_bra_tensor_ignore_one_leg(bra_tensor: np.ndarray,
                                       ketopblock_tensor: np.ndarray,
                                       state_node: Node,
                                       ignoring_node_id: str) -> np.ndarray:
    """
    Contracts the bra tensor with the contracted tensor.

    The current tensor has the ket tensor and the operator tensor of this
    node already contracted with each other and with all but one neighbour
    block. The remaining neighbour block is the one that the remaining legs
    point towards. The neighbour blocks are the results of already contracted
    subtrees starting from this node.

    Args:
        bra_tensor (np.ndarray): The bra tensor.
        ketopblock_tensor (np.ndarray): The contracted tensor. (ACH in the
            diagram)
        state_node (Node): The node of the state. We assume the bra state
            is the adjoint of the ket state.
        ignoring_node_id (str): The identifier of the node to which the
            virtual leg should not point.

    Returns:
        np.ndarray: The contracted tensor::
                                    
                     _____      ______
                ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
    
    """
    num_neighbours = state_node.nneighbours()
    ignored_node_index = state_node.neighbour_index(ignoring_node_id)
    legs_tensor = list(range(1,num_neighbours))
    legs_tensor.append(num_neighbours+1)
    legs_bra_tensor = list(range(ignored_node_index))
    legs_bra_tensor.extend(range(ignored_node_index+1,num_neighbours+1))
    return np.tensordot(ketopblock_tensor, bra_tensor,
                        axes=(legs_tensor, legs_bra_tensor))

def _node_state_phys_leg(node: Node) -> int:
    """
    Finds the physical leg of a node of a state.

    Returns:
        int: The index of the physical leg.
    """
    return node.nneighbours()

def _node_operator_input_leg(node: Node) -> int:
    """
    Finds the leg of a node of the hamiltonian corresponding to the input.

    Returns:
        int: The index of the leg corresponding to input.
    """
    # Corr ket leg
    return node.nneighbours() + 1

def _node_operator_output_leg(node: Node) -> int:
    """
    Finds the leg of a node of the hamiltonian corresponding to the output.
    
    Returns:
        int: The index of the leg corresponding to output.
    """
    # Corr bra leg
    return node.nneighbours()


