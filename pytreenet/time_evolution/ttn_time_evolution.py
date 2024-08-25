from __future__ import annotations
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass

from numpy import ndarray
from copy import deepcopy
from .time_evolution import TimeEvolution
from ..ttns import TreeTensorNetworkState
from ..ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from pytreenet.time_evolution.Subspace_expansion import contract_ttno_with_ttn
from pytreenet.contractions.state_state_contraction import contract_ttn_Lindblad 
from pytreenet.time_evolution.Subspace_expansion import original_form
@dataclass
class TTNTimeEvolutionConfig:
    """
    Configuration for the TTN time evolution.

    In this configuration class additional parameters for the time evolution
    of a tree tensor network can be specified and entered. This allows for the
    same extendability as `**kwargs` but with the added benefit of type hints
    and better documentation.
    """
    record_bond_dim: bool = False
    Lindblad: bool = False

class TTNTimeEvolution(TimeEvolution):
    """
    A time evolution for tree tensor networks.
    
    Provides functionality to compute expectation values of operators during
    the time evolution and record bond dimensions of the current state.

    Attributes:
        bond_dims (Union[None,Dict[str,int]]): If a recording of the bond
            dimension is intended, they are recorded here.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 time_step_size: float, final_time: float,
                 operators: Union[List[Union[TensorProduct, TTNO]], TensorProduct, TTNO],
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        A time evolution for a tree tensor network state.

        Args:
            initial_state (TreeTensorNetwork): The initial state of the time
                evolution.
            time_step_site (float): The time difference progressed by one time
                step.
            final_time (float): The final time until which the time evolution
                runs.
            operators (Union[List[Union[TensorProduct, TTNO]], TensorProduct, TTNO]):
                Operators for which the expectation value should be recorded
                during the time evolution.
            config (Union[TTNTimeEvolutionConfig,None]): The configuration of
                time evolution. Defaults to None.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)
        self.initial_state: TreeTensorNetworkState
        self.state: TreeTensorNetworkState

        if config is not None and config.record_bond_dim:
            self.bond_dims = {}
        else:
            self.bond_dims = None

        if config.Lindblad:
            self.Lindblad = True    
        else:
            self.Lindblad = False    

    @property
    def records_bond_dim(self) -> bool:
        """
        Are the bond dimensions recorded or not.
        """
        return self.bond_dims is not None
    
    @property 
    def Lindblad_mode(self) -> bool:
        """
        Is the Lindblad mode activated or not.
        """
        return self.Lindblad


    def obtain_bond_dims(self) -> Dict[Tuple[str,str], int]:
        """
        Obtains a dictionary of all bond dimensions in the current state.
        """
        return self.state.bond_dims()

    def record_bond_dimensions(self):
        """
        Records the bond dimensions of the current state, if desired to do so.
        """
        if self.records_bond_dim:
            if len(self.bond_dims) == 0:
                self.bond_dims = {key: [value] for key, value in self.obtain_bond_dims().items()}
            else:
                for key, value in self.obtain_bond_dims().items():
                    self.bond_dims[key].append(value)

    def operator_result(self,
                        operator_id: str | int,
                        realise: bool = False) -> ndarray:
        """
        Includes the possibility to obtain the bond dimension from the results.

        Args:
            operator_id (Union[str,int]): The identifier or position of the
                operator, whose expectation value results should be returned.
            realise (bool, optional): Whether the results should be
                transformed into real numbers.

        Returns:
            ndarray: The expectation value results over time.
        """
        if isinstance(operator_id, str) and operator_id == "bond_dim":
            if self.records_bond_dim is not None:
                return self.bond_dims
            errstr = "Bond dimensions are not being recorded."
            raise ValueError(errstr)
        return super().operator_result(operator_id, realise)

    def evaluate_operators(self) -> ndarray:
        """
        Evaluates the operator including the recording of bond dimensions.
        """
        current_results = super().evaluate_operators()
        self.record_bond_dimensions()
        return current_results

    def evaluate_operator(self, operator: Union[TensorProduct,TTNO]) -> complex:
        """
        Evaluate the expectation value of a single operator.

        Args:
            operator (TensorProduct): The operator for which to compute the
                expectation value.
        
        Returns:
            np.ndarray: The expectation value of the operator with respect to
                the current state.
        """
        if self.Lindblad_mode:
        #    ttn = deepcopy(self.state)
        #    ttno = deepcopy(operator)
        #    ttn = original_form(ttn , self.two_neighbour_form_dict)
        #    op_state = contract_ttno_with_ttn( ttno, ttn)
        #    return contract_ttn_Lindblad(op_state)  
        #   
            ttn = deepcopy(self.state)
            ttno = deepcopy(operator)
            ttn = original_form(ttn , self.two_neighbour_form_dict)
            return ttn.operator_expectation_value_Lindblad(ttno)     
        #ttn = deepcopy(self.state)
        #ttno = deepcopy(operator)
        #ttn = original_form(ttn , self.two_neighbour_form_dict)
        #return ttn.operator_expectation_value(ttno)       
