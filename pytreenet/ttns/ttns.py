from __future__ import annotations
from typing import Union
from copy import deepcopy

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from ..contractions.state_state_contraction import contract_two_ttns
from ..contractions.state_operator_contraction import expectation_value , expectation_value_Lindblad , adjust_bra_to_ket
from ..util.tensor_splitting import SplitMode
from ..util import copy_object

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks
     representing a state.
    """

    def normalize_ttn(self,to_copy: bool=False) -> TreeTensorNetwork:
        return normalize_ttn(self, to_copy=to_copy)    

    def scalar_product(self) -> complex:
        """
        Computes the scalar product of this TTNS

        Returns:
            complex: The resulting scalar product <TTNS|TTNS>
        """
        if self.orthogonality_center_id is not None:
            tensor = self.tensors[self.orthogonality_center_id]
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor, tensor_conj, axes=(legs,legs)))
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        return contract_two_ttns(ttn, ttn.conjugate())

    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: np.ndarray) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on
         the node specified.
        Assumes the node has only one open leg.

        Args:
            node_id (str): The identifier of the node, the operator is applied to.
            operator (np.ndarray): The operator of which we determine the expectation value.
             Note that the state will be contracted with axis/leg 0 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >
        """
        if self.orthogonality_center_id == node_id:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,0))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

    def operator_expectation_value(self, operator: Union[TensorProduct,TTNO]) -> complex:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (Union[TensorProduct,TTNO]): A TensorProduct representing
            the operator as many single site operators. Otherwise a a TTNO
            with the same structure as the TTNS.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        if isinstance(operator, TensorProduct):
            if len(operator) == 0:
                return self.scalar_product()
            if len(operator) == 1:
                node_id = list(operator.keys())[0]
                if self.orthogonality_center_id == node_id:
                    op = operator[node_id]
                    return self.single_site_operator_expectation_value(node_id, op)
            # Very inefficient, fix later without copy
            ttn = deepcopy(self)
            conj_ttn = ttn.conjugate()
            for node_id, single_site_operator in operator.items():
                ttn.absorb_into_open_legs(node_id, single_site_operator)
            return contract_two_ttns(ttn, conj_ttn)
        # Operator is a TTNO
        return expectation_value(self, operator)

    def operator_expectation_value_Lindblad(self, operator: Union[TensorProduct,TTNO]) -> complex:
        return expectation_value_Lindblad(self, operator)


    def is_in_canonical_form(self, node_id: Union[None,str] = None) -> bool:
        """
        Returns whether the TTNS is in canonical form.
        
        If a node_id is specified, it will check as if that node is the
        orthogonalisation center. If no node is given, the current
        orthogonalisation center will be used.

        Args:
            node_id (Union[None,str], optional): The node to check. If None,
                the current orthogonalisation center will be used. Defaults
                to None.
        
        Returns:
            bool: Whether the TTNS is in canonical form.
        """
        if node_id is None:
            node_id = self.orthogonality_center_id
        if node_id is None:
            return False
        total_contraction = self.scalar_product()
        local_tensor = self.tensors[node_id]
        legs = range(local_tensor.ndim)
        local_contraction = complex(np.tensordot(local_tensor, local_tensor.conj(),
                                                 axes=(legs,legs)))
        # If the TTNS is in canonical form, the contraction of the
        # orthogonality center should be equal to the norm of the state.
        return np.allclose(total_contraction, local_contraction)

def normalize_ttn(ttn: TreeTensorNetworkState , to_copy = False):
   """
    Normalize a tree tensor network.
    Args:
        ttn : TreeTensorNetwork
        The tree tensor network to normalize.
        to_copy : bool, optional
                  If True, the input tree tensor network is not modified and a new tree tensor network is returned.
                  If False, the input tree tensor network is modified and returned.
                  Default is False.
    Returns : 
        The normalized tree tensor network.
    """
   ttn_normalized = copy_object(ttn, deep=to_copy)
   if len(ttn_normalized.nodes) == 1:
       node_id = list(ttn_normalized.nodes.keys())[0]
       tensor = ttn_normalized.tensors[node_id]
       indices  = tuple(ttn_normalized.nodes[node_id].open_legs)
       norm = np.sqrt(np.tensordot(tensor,tensor.conj(), axes = (indices , indices) ))
       ttn_normalized.tensors[node_id] = ttn_normalized.tensors[node_id] / norm
   else :    
      norm = contract_two_ttns(ttn_normalized,ttn_normalized.conjugate())
      for node_id in list(ttn_normalized.nodes.keys()):
          norm = contract_two_ttns(ttn_normalized,ttn_normalized.conjugate())
          ttn_normalized.tensors[node_id] /= np.sqrt(norm)
   return ttn_normalized 



def normalize_ttn_Lindblad_3_conj(ttn , orth_center_id_1 , orth_center_id_2 ): 
    
    ttn_normalized = copy_object(ttn, deep = True)
    adjust_bra_to_ket(ttn_normalized)
    ttn_normalized.canonical_form_twosite(orth_center_id_1, orth_center_id_2 , mode = SplitMode.REDUCED)    
    I = TTNO.Identity(ttn_normalized)
    norm = ttn_normalized.operator_expectation_value_Lindblad(I)
    norm = np.sqrt(norm)

    T = ttn_normalized.tensors[orth_center_id_1].astype(complex)
    T /= norm
    ttn_normalized.tensors[orth_center_id_1] = T
    ttn_normalized.nodes[orth_center_id_1].link_tensor(T)

    T = ttn_normalized.tensors[orth_center_id_2].astype(complex)
    T /= norm.conj()
    ttn_normalized.tensors[orth_center_id_2] = T
    ttn_normalized.nodes[orth_center_id_2].link_tensor(T)
    return ttn_normalized

def normalize_ttn_Lindblad_3(ttn , orth_center_id_1 , orth_center_id_2 ): 
    
    ttn_normalized = copy_object(ttn, deep = True)
    adjust_bra_to_ket(ttn_normalized)
    ttn_normalized.canonical_form_twosite(orth_center_id_1, orth_center_id_2 , mode = SplitMode.REDUCED)    
    I = TTNO.Identity(ttn_normalized)
    norm = ttn_normalized.operator_expectation_value_Lindblad(I)
    norm = np.sqrt(norm)

    T = ttn_normalized.tensors[orth_center_id_1].astype(complex)
    T /= norm
    ttn_normalized.tensors[orth_center_id_1] = T
    ttn_normalized.nodes[orth_center_id_1].link_tensor(T)

    T = ttn_normalized.tensors[orth_center_id_2].astype(complex)
    T /= norm
    ttn_normalized.tensors[orth_center_id_2] = T
    ttn_normalized.nodes[orth_center_id_2].link_tensor(T)
    return ttn_normalized

def normalize_ttn_Lindblad_1_conj(ttn) : 
    ttn_normalized = copy_object(ttn, deep = True)
    I = TTNO.Identity(ttn_normalized)
    norm = ttn_normalized.operator_expectation_value_Lindblad(I)
    n = len(ttn.nodes) // 2
    norm = np.sqrt(norm ** (1/n))
    for ket_id in [node.identifier for node in ttn.nodes.values() if str(node.identifier).startswith("S")]:
        bra_id = ket_id.replace('Site', 'Node')
        T = ttn_normalized.tensors[ket_id].astype(complex)
        T /= norm
        ttn_normalized.tensors[ket_id] = T
        ttn_normalized.nodes[ket_id].link_tensor(T)

        T = ttn_normalized.tensors[bra_id].astype(complex)
        T /= norm.conj()
        ttn_normalized.tensors[bra_id] = T
        ttn_normalized.nodes[bra_id].link_tensor(T)

    return ttn_normalized

def normalize_ttn_Lindblad_1(ttn) : 
    ttn_normalized = copy_object(ttn, deep = True)
    I = TTNO.Identity(ttn_normalized)
    norm = ttn_normalized.operator_expectation_value_Lindblad(I)
    n = len(ttn.nodes) // 2
    norm = np.sqrt(norm ** (1/n))
    for ket_id in [node.identifier for node in ttn.nodes.values() if str(node.identifier).startswith("S")]:
        bra_id = ket_id.replace('Site', 'Node')
        T = ttn_normalized.tensors[ket_id].astype(complex)
        T /= norm
        ttn_normalized.tensors[ket_id] = T
        ttn_normalized.nodes[ket_id].link_tensor(T)

        T = ttn_normalized.tensors[bra_id].astype(complex)
        T /= norm
        ttn_normalized.tensors[bra_id] = T
        ttn_normalized.nodes[bra_id].link_tensor(T)

    return ttn_normalized


def normalize_ttn_Lindblad_4(ttn , orth_center_id): 
    ttn_normalized = copy_object(ttn, deep = True)
    adjust_bra_to_ket(ttn_normalized)
    ttn_normalized.canonical_form(orth_center_id, mode = SplitMode.REDUCED) 
 
    I = TTNO.Identity(ttn_normalized)
    norm = ttn_normalized.operator_expectation_value_Lindblad(I)

    T = ttn_normalized.tensors[orth_center_id].astype(complex)
    T /= norm
    ttn_normalized.tensors[orth_center_id] = T
    ttn_normalized.nodes[orth_center_id].link_tensor(T)
    return ttn_normalized


def normalize_ttn_Lindblad_5_conj(ttn) : 
    ttn_normalized = copy_object(ttn, deep = True)
    I = TTNO.Identity(ttn_normalized)

    for ket_id in [node.identifier for node in ttn.nodes.values() if str(node.identifier).startswith("S")]:
        bra_id = ket_id.replace('Site', 'Node')
        norm = np.sqrt(ttn_normalized.operator_expectation_value_Lindblad(I))

        T = ttn_normalized.tensors[ket_id].astype(complex)
        T /= norm
        ttn_normalized.tensors[ket_id] = T
        ttn_normalized.nodes[ket_id].link_tensor(T)

        T = ttn_normalized.tensors[bra_id].astype(complex)
        T /= norm.conj()
        ttn_normalized.tensors[bra_id] = T
        ttn_normalized.nodes[bra_id].link_tensor(T)        

    return ttn_normalized

def normalize_ttn_Lindblad_5(ttn) : 
    ttn_normalized = copy_object(ttn, deep = True)
    I = TTNO.Identity(ttn_normalized)
    for node_id in list(ttn.nodes.keys()):
        norm = ttn_normalized.operator_expectation_value_Lindblad(I)
        T = ttn_normalized.tensors[node_id].astype(complex)
        T /= norm
        ttn_normalized.tensors[node_id] = T
        ttn_normalized.nodes[node_id].link_tensor(T)

    return ttn_normalized

import pytreenet as ptn
def normalize_ttn_Lindblad_XX(vectorized_pho , orth_center_id_1 , orth_center_id_2): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized.canonical_form_twosite( orth_center_id_1, orth_center_id_2,mode = SplitMode.REDUCED)
    pho_normalized = ptn.adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    pho_normalized_conj = pho_normalized.conjugate()
    norm = contract_two_ttns(pho_normalized_conj , pho_normalized)
    norm = np.sqrt(norm)
    T = pho_normalized.tensors[orth_center_id_1].astype(complex)
    T /= np.sqrt(norm)
    pho_normalized.tensors[orth_center_id_1] = T
    pho_normalized.nodes[orth_center_id_1].link_tensor(T)

    T = pho_normalized.tensors[orth_center_id_2].astype(complex)
    T /= np.sqrt(norm).conj()
    pho_normalized.tensors[orth_center_id_2] = T
    pho_normalized.nodes[orth_center_id_2].link_tensor(T)
    return pho_normalized

TTNS = TreeTensorNetworkState

