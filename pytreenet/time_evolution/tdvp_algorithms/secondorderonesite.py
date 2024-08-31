from __future__ import annotations
from typing import List, Union
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from ...operators.tensorproduct import TensorProduct
from ...ttno.ttno import TTNO
from ...ttns import TreeTensorNetworkState , normalize_ttn_Lindblad
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ..Subspace_expansion import expand_subspace , KrylovBasisMode , max_two_neighbour_form , original_form
from ...util.tensor_splitting import SplitMode , SVDParameters
from ...core.canonical_form import adjust_ttn1_structure_to_ttn2 , adjust_ttno_structure_to_ttn
from pytreenet.contractions.state_operator_contraction import adjust_operator_to_ket , adjust_bra_to_ket
from .onesitetdvp import OneSiteTDVP
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder


class SecondOrderOneSiteTDVP(OneSiteTDVP):
    """
    The first order one site TDVP algorithm.

    This means we have second order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At/2)*exp(Bt/2)*exp(Bt/2)*exp(At/2)

    Has the same attributes as the TDVP-Algorithm clas with two additions.

    Attributes:
        backwards_update_path (List[str]): The update path that traverses
            backwards.
        backwards_orth_path (List[List[str]]): The orthogonalisation paths for
            the backwards run.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 num_vecs: int = 3, 
                 tau: float = 1e-2, 
                 SVDParameters : SVDParameters = SVDParameters(),
                 expansion_steps: int = 10,
                 initial_tol: float = 1e-5,
                 tol_step: float = 1e-1, 
                 max_bond: int = 32, 
                 norm_tol: float = np.inf,
                 KrylovBasisMode : KrylovBasisMode = KrylovBasisMode.apply_ham,                  
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        Initialize the second order one site TDVP algorithm.

        Args:
            initial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian of the system.
            time_step_size (float): The time step size.
            final_time (float): The final time of the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): The operators
                for which the expectation values are calculated.
            config (Union[TTNTimeEvolutionConfig,None], optional): The time
                evolution configuration. Defaults to None.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time, operators, num_vecs, 
                         tau,
                         SVDParameters,
                         expansion_steps,
                         initial_tol,
                         tol_step,
                         max_bond,
                         KrylovBasisMode,  
                         config)
        self.norm_tol = norm_tol
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()
        self.state , self.hamiltonian = self._init_two_neighbour_form()

    def _init_two_neighbour_form(self):
        """
        Transform the state and the Hamiltonian into the max two neighbour form.
        """
        # adjust_operator_to_ket(self.hamiltonian , self.state)
        self.hamiltonian = adjust_ttno_structure_to_ttn(self.hamiltonian , self.state)
        self.state , dict1 = max_two_neighbour_form(self.state)
        self.hamiltonian , dict2 = max_two_neighbour_form(self.hamiltonian , dict1)
        #self.hamiltonian = adjust_ttno_structure_to_ttn(tdvp_ex1.hamiltonian , tdvp_ex1.state)
        assert dict1 == dict2
        self.partial_tree_cache = PartialTreeCachDict()
        self._init_partial_tree_cache()
        self.update_path = TDVPUpdatePathFinder(self.state).find_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path) 
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path() 
        self.two_neighbour_form_dict = dict1
        return self.state , self.hamiltonian
    
    def _init_second_order_update_path(self) -> List[str]:
        """
        Find the update path that traverses backwards.
        """
        return list(reversed(self.update_path))

    def _init_second_order_orth_path(self) -> List[List[str]]:
        """
        Find the orthogonalisation paths for the backwards run.
        """
        back_orthogonalization_path = []
        for i, node_id in enumerate(self.backwards_update_path[1:-1]):
            current_path = self.state.path_from_to(node_id,
                                                   self.backwards_update_path[i+2])
            current_path = current_path[:-1]
            back_orthogonalization_path.append(current_path)
        back_orthogonalization_path.append([self.backwards_update_path[-1]])
        return back_orthogonalization_path

    def _update_forward_site_and_link(self, node_id: str,
                                      next_node_id: str):
        """
        Run the forward update with half time step.

        First the site tensor is updated and then the link tensor.

        Args:
            node_id (str): The identifier of the site to be updated.
            next_node_id (str): The other node of the link to be updated.
        """
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id,
                          time_step_factor=0.5)
        self._update_link(node_id, next_node_id,
                          time_step_factor=0.5)

    def forward_sweep(self):
        """
        Perform the forward sweep through the state.
        """
        for i, node_id in enumerate(self.update_path[:-1]):
            # Orthogonalize
            if i>0:
                self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i-1])
            # Select Next Node
            next_node_id = self.orthogonalization_path[i][0]
            # Update
            self._update_forward_site_and_link(node_id, next_node_id)

    def _final_forward_update(self):
        """
        Perform the final forward update. 
        
        To save some computation, the update is performed with a full time
        step. Since the first update backwards occurs on the same node. We
        also do not need to update any link tensors.
        """
        node_id = self.update_path[-1]
        assert node_id == self.backwards_update_path[0]
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id)

    def _update_first_backward_link(self):
        """
        Update the link between the first and second node in the backwards
        update path with a half time step.
        
        We have already updated the first site on the backwards update path
        and the link will always be next to it, so the orthogonality center
        is already at the correct position.
        """
        next_node_id = self.backwards_update_path[1]
        self._update_link(self.state.orthogonality_center_id,
                          next_node_id,
                          time_step_factor=0.5)

    def _normal_backward_update(self, node_id: str,
                                update_index: int):
        """
        The normal way to make a backwards update.
        
        First the site tensor is updated. Then the orthogonality center is
        moved, if needed. Finally the link tensor between the new
        orthogonality center and the next node is updated. 
        
        Args:
            node_id (str): The identifier of the site to be updated.
            update_index (int): The index of the update in the backwards
                update path.
        """
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id, time_step_factor=0.5)
        new_orth_center = self.backwards_orth_path[update_index-1]
        self._move_orth_and_update_cache_for_path(new_orth_center)
        next_node_id = self.backwards_update_path[update_index+1]
        self._update_link(self.state.orthogonality_center_id,
                          next_node_id,
                          time_step_factor=0.5)

    def _final_backward_update(self):
        """
        Perform the final backward update.
        
        Since this is the last node that needs updating, no link update is
        required afterwards.
        """
        node_id = self.backwards_update_path[-1]
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id, time_step_factor=0.5)

    def backward_sweep(self):
        """
        Perform the backward sweep through the state.
        """
        self._update_first_backward_link()
        for i, node_id in enumerate(self.backwards_update_path[1:-1]):
            self._normal_backward_update(node_id, i+1)
        self._final_backward_update()

    def run_one_time_step(self):
        """
        Run a single second order time step.
        
        This mean we run a full forward and a full backward sweep through the
        tree.
        """
        self.forward_sweep()
        self._final_forward_update()
        self.backward_sweep()


    def forward_sweep_ex(self):
        """
        Perform the forward sweep through the state.
        """
        for i, node_id in enumerate(self.update_path[:-1]):
            # Orthogonalize
            if i>0:
                self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i-1])
            # Select Next Node
            next_node_id = self.orthogonalization_path[i][0]
            # Update
            self._update_forward_site_and_link(node_id, next_node_id)

    def run_one_time_step_ex(self):
        """
        Run a single second order time step.
        
        This mean we run a full forward and a full backward sweep through the
        tree.
        """

        self.forward_sweep_ex()
        self._final_forward_update()
        self.backward_sweep()  
    

    def run_ex(self, evaluation_time: Union[int,"inf"] = 1, filepath: str = "",
            pgbar: bool = True,):
        """
        Runs this time evolution algorithm for the given parameters.

        The desired operator expectation values are evaluated and saved.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
            filepath (str, optional): If results are to be saved in an external file,
                the path to that file can be specified here. Defaults to "".
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
      
        self._init_results(evaluation_time)
        assert self._results is not None
        tol_step = -1
        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):
            if i != 0:  # We also measure the initial expectation_values   

                ###########
                tol_step += 1
                if i % self.expansion_steps == 0 :
                    if state_bond.max_bond_dim() < self.max_bond:
                        tol = self.initial_tol * self.tol_step ** tol_step
                        state_ex = expand_subspace(state_ex, self.hamiltonian, 
                                                    self.num_vecs, self.tau, 
                                                    self.SVDParameters, tol, self.KrylovBasisMode)
                        self.state = state_ex
                        self.state.move_orthogonalization_center(self.update_path[0],mode = SplitMode.KEEP)
                        self.partial_tree_cache = PartialTreeCachDict()
                        self._init_partial_tree_cache()
                    else:
                        assert self.state.orthogonality_center_id == self.update_path[0]   
                ##########

                self.run_one_time_step_ex() 
                state_ex = deepcopy(self.state) 
            ttn = deepcopy(self.state)
            ttn = original_form(ttn , self.two_neighbour_form_dict)
            I = TTNO.Identity(ttn)
            I_ex = ttn.operator_expectation_value_Lindblad(I)
            norm = I_ex
            if np.abs(np.abs(I_ex) - 1)  > self.norm_tol :
                self.state = normalize_ttn_Lindblad(ttn)     
            else:
                self.state = ttn                   
            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                current_results = self.evaluate_operators() / norm
                self._results[0:-1, index] = current_results
                # Save current time
                self._results[-1, index] = i*self.time_step_size  
            max_two_neighbour_form(self.state , self.two_neighbour_form_dict)

            state_bond = deepcopy(self.state)
            self.record_bond_dimensions()
            
            self._orthogonalize_init()
            self.partial_tree_cache = PartialTreeCachDict()
            self._init_partial_tree_cache()                 
        
        if evaluation_time == "inf":
            current_results = self.evaluate_operators()
            self._results[0:-1, 0] = current_results
            self._results[-1, 0] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)        #   