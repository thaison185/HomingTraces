import json
from typing import Dict, List, Tuple
import itertools
from itertools import product
import json
from collections import defaultdict, deque
from collections import defaultdict, deque
from typing import Tuple, List, Dict, FrozenSet, Any

# ============================================================================
# TFSM ABSTRACTION - TreeNode and FSMWithTimeout Classes
# ============================================================================

class TreeNode:
    """
    Node in the truncated successor tree for TFSM analysis.
    Used to build and traverse FSM abstraction hierarchically.
    """
    def __init__(self, value: Tuple[str, str], states: List[str]):
        """
        Initialize tree node

        Args:
            value: Tuple of (input_symbol, output_symbol)
            states: List of states reachable from this node
        """
        self.value = value
        self.states = states
        self.children = []

    def add_child(self, child_node: 'TreeNode') -> None:
        """Add a child node to current node"""
        self.children.append(child_node)

    def __repr__(self, level=0) -> str:
        """String representation for tree visualization"""
        ret = "\t" * level + f"Node(value={self.value}, states={self.states})\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class FSMWithTimeout:
    """
    FSM with Timeout support for protocol state identification.

    Key methods:
    - derive_fsm_abstraction(): Derive abstract FSM from FSM
    - io_homing_sequences(): Find I/O homing sequences for FSM
    """

    def __init__(self, config_file: str):
        """
        Initialize FSM model from JSON configuration

        Args:
            config_file: Path to JSON file containing TFSM specification
        """
        with open(config_file, "r") as file:
            fsm_description = json.load(file)

        self.states = fsm_description.get("states", {})
        self.inputs = fsm_description.get("inputs", {})
        self.outputs = fsm_description.get("outputs", {})
        self.transitions = fsm_description.get("transitions", {})
        self.timeouts = fsm_description.get("timeouts", {})
        self.abstraction = None
        self.tree = None

    def derive_fsm_abstraction(self):
        """
        Derive FSM abstraction from TFSM with timeouts.

        For each state with timeout T:
        - Create abstract states: state_t0, state_t1, ..., state_tT
        - Add time-tick transitions: state_ti --('1','1')--> state_ti+1
        - Add time-tick timeout transition: state_tT --('1','1')--> next_state_t0
        - For each input, add transitions at all time points
        """
        abstraction_states = []
        abstraction_transitions = {}

        # Create abstract states for timeouts
        for state in self.states:
            if state not in self.timeouts:
                abstraction_state = f"{state}_t0"
                abstraction_states.append(abstraction_state)
                abstraction_transitions.setdefault(abstraction_state, []).append(("1", "1", abstraction_state))
            else:
                timeout_info = self.timeouts.get(state)
                next_state = timeout_info.get("next_state", {}) if isinstance(timeout_info, dict) else timeout_info[0]
                timeout_duration = timeout_info.get("duration_seconds", {}) if isinstance(timeout_info, dict) else timeout_info[1]

                for t in range(timeout_duration):
                    abstraction_state = f"{state}_t{t}"
                    abstraction_states.append(abstraction_state)

                    if t < timeout_duration - 1:
                        # Time-tick transition to next time state
                        abstraction_transitions.setdefault(abstraction_state, []).append(
                            ("1", "1", f"{state}_t{t+1}")
                        )
                    else:
                        # Timeout transition to next state
                        abstraction_transitions.setdefault(abstraction_state, []).append(
                            ("1", "1", f"{next_state}_t0")
                        )

        # Add input transitions at all time points
        for state, transitions in self.transitions.items():
            for input_symbol, output_symbol, next_state in transitions:
                if state in self.timeouts:
                    timeout_info = self.timeouts.get(state)
                    timeout_duration = timeout_info.get("duration_seconds", {}) if isinstance(timeout_info, dict) else timeout_info[1]

                    for t in range(timeout_duration):
                        abstraction_state = f"{state}_t{t}"
                        abstraction_transitions.setdefault(abstraction_state, []).append(
                            (input_symbol, output_symbol, f"{next_state}_t0")
                        )
                else:
                    abstraction_state = f"{state}_t0"
                    abstraction_transitions.setdefault(abstraction_state, []).append(
                        (input_symbol, output_symbol, f"{next_state}_t0")
                    )

        self.abstraction = {
            "states": abstraction_states,
            "inputs": self.inputs + ["1"],
            "outputs": self.outputs + ["1"],
            "transitions": abstraction_transitions
        }

    def get_inputs(self) -> List[str]:
        """Return list of input symbols"""
        return self.inputs

    def get_outputs(self) -> List[str]:
        """Return list of output symbols"""
        return self.outputs

    def abstraction_successor(self, state: str, input_symbol: str, output_symbol: str) -> List[str]:
        """
        Get successor state(s) for given state, input, and output in abstraction FSM

        Returns: List of successor states, or [-1] if transition undefined
        """
        if self.abstraction is None:
            self.derive_fsm_abstraction()

        abstraction_transitions = self.abstraction["transitions"]

        if state in abstraction_transitions:
            ret = [(out, next_state) for inp, out, next_state in abstraction_transitions[state] if inp == input_symbol ]
            if not ret: 
                return [-1]
            return [next_state for inp, out, next_state in abstraction_transitions[state] if (inp == input_symbol and out == output_symbol)]
        return []

    def create_fsm_without_timeout(self) -> Dict:
        """
        Create classical FSM by removing all timeout transitions.
        """
        # Copy original FSM structure (no timeout)
        fsm_no_timeout = {
            "states": self.states.copy(),
            "inputs": self.inputs.copy(),
            "outputs": self.outputs.copy(),
            "transitions": {}
        }

        # Copy transitions from original TFSM (same as self.transitions)
        for state, transitions_list in self.transitions.items():
            fsm_no_timeout["transitions"][state] = []
            for input_symbol, output_symbol, next_state in transitions_list:
                fsm_no_timeout["transitions"][state].append(
                    (input_symbol, output_symbol, next_state)
                )

        return fsm_no_timeout

    def _check_duplicate(self, A, B):
        lens = {len(t) for t in A}
        for i in lens:
            if i <= len(B) and B[-i:] in A:
                return False
        return True

    def io_homing_sequences_from_model(self, fsm_model: Dict, max_length: int) -> List[Tuple[List[Tuple[str, str]], str]]:
        """
        Find homing sequences from given FSM model.

        Args:
            fsm_model: FSM model dict with states, inputs, outputs, transitions
            max_length: Maximum sequence length to explore

        Returns:
            List of (io_sequence, final_state) tuples
        """
        abstraction_states = fsm_model.get("states", [])
        abstraction_transitions = fsm_model.get("transitions", {})
        inputs = fsm_model.get("inputs", [])
        outputs = fsm_model.get("outputs", [])

        homing_sequences = []
        seen_sequences = set()

        # Generate I/O pairs
        io_pairs = list((i, o) for (i, o) in product(inputs, outputs) if (i == '1') == (o == '1'))

        # Search for homing sequences of increasing length
        for length in range(1, max_length + 1):
            for sequence in itertools.product(io_pairs, repeat=length):
                if sequence[-1] == ('1', '1'):
                    continue
                flag = True
                current_states = abstraction_states.copy()
                io_sequence = []

                # Execute sequence on all states
                for input_symbol, output_symbol in sequence:
                    next_states = []
                    for state in current_states:
                        if state in abstraction_transitions:
                            successors = [(out, next_state) for inp, out, next_state in abstraction_transitions[state] if inp == input_symbol ]
                            if not successors:
                                flag = False
                                break
                            successors = [next_state for inp, out, next_state in abstraction_transitions[state] if (inp == input_symbol and out == output_symbol)]
                        next_states.extend(successors)
                    io_sequence.extend([(input_symbol, output_symbol)])

                    if not flag:
                        break

                    current_states = list(set(next_states))  # Remove duplicates

                    # Early termination if all states converge before sequence end
                    if len(current_states) == 1 and io_sequence[-1] != ('1', '1'):
                        key = tuple(io_sequence)
                        if key not in seen_sequences and self._check_duplicate(seen_sequences, key):
                            seen_sequences.add(key)
                            homing_sequences.append((io_sequence, current_states[0]))
                        flag = False
                        break

                # Check if valid homing sequence
                if flag and len(set(current_states)) == 1:
                    key = tuple(io_sequence)
                    if key not in seen_sequences and self._check_duplicate(seen_sequences, key):
                        seen_sequences.add(key)
                        homing_sequences.append((io_sequence, current_states[0]))

        return homing_sequences

    def io_homing_sequences(self, max_length: int) -> List[Tuple[List[Tuple[str, str]], str]]:
        """
        Derive I/O homing sequences for the FSM WITH timeout.
        """
        if self.abstraction is None:
            self.derive_fsm_abstraction()

        return self.io_homing_sequences_from_model(self.abstraction, max_length)

    def build_truncated_tree(self, abstraction_fsm: Dict[str, any], max_length: int) -> TreeNode:
        """
        Build the truncated successor tree as a linked structure.
        return: Root node of the tree.
        """
        abstraction_states = abstraction_fsm["states"]
        abstraction_transitions = abstraction_fsm["transitions"]
        root = TreeNode(value=("root", "root"), states=abstraction_states)


        def add_children(node: TreeNode, depth: int):
            if depth >= max_length:
                return
            for input_symbol in abstraction_fsm["inputs"]:
                for output_symbol in abstraction_fsm["outputs"]:
                    if input_symbol == 1 and output_symbol != 1 or input_symbol != 1 and output_symbol == 1:
                        continue
                    next_states = []
                    for state in node.states:
                        successors = self.abstraction_successor(state, input_symbol, output_symbol)
                        if successors != [-1]:
                            next_states.extend(successors)
                    if next_states:
                        next_states = list(set(next_states))
                        child_node = TreeNode(value=(input_symbol, output_symbol), states=next_states)
                        node.add_child(child_node)
                        if (len(next_states) > 1):
                            add_children(child_node, depth + 1)

        add_children(root, 0)
        return root

# ============================================================================
# EFSM CLASS
# ============================================================================
class EFSM:
    """Extended Finite State Machine - brute-force homing on l-equivalent edges."""

    def __init__(self, json_file: str, initial_state = None):
        """
        Load EFSM from JSON file and prepare initial configurations.
        """

        with open(json_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)

        self.states = spec.get("states", [])
        self.inputs = spec.get("inputs", [])
        self.outputs = spec.get("outputs", [])
        self.initial_vars = spec.get("variables", {})
        self.transitions = spec.get("transitions", [])

        self.var_names = list(self.initial_vars.keys())
        self.initial_state = initial_state

        # Variable bounds for simulation
        self.var_bounds = {}
        for var_name, var_default in self.initial_vars.items():
            if isinstance(var_default, bool):
                self.var_bounds[var_name] = [False, True]
            elif isinstance(var_default, int):
                self.var_bounds[var_name] = list(range(0, 2))
            else:
                self.var_bounds[var_name] = [var_default]

        # Prepare initial configurations
        self._prepare_initial_configs()

    def _prepare_initial_configs(self):
        """
        Prepare initial configurations.
        """

        # Generate variable domain combinations
        var_domains = [self.var_bounds[v] for v in self.var_names]
        all_var_combos = list(itertools.product(*var_domains))

        # Determine which states to include
        if self.initial_state is not None:
            # Known initial state
            states_to_use = [self.initial_state]
        else:
            # Unknown initial state (all states)
            states_to_use = self.states

        # Build initial configurations
        configs = [(s,) + combo for s in states_to_use for combo in all_var_combos]
        self.initial_configs = frozenset(configs)

    def _eval_guard(self, state: str, guard_str: str, var_values: Dict[str, Any]) -> bool:
        """Evaluate guard condition."""
        if not guard_str or guard_str.strip() == "":
            return True
        try:
            env = var_values.copy()
            return eval(guard_str, {"__builtins__": {}}, env)
        except:
            return False

    def _apply_action(self, action_str: str, var_values: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action (state update)."""
        new_vars = var_values.copy()
        if not action_str or action_str.strip() == "":
            return new_vars

        statements = [s.strip() for s in action_str.split(';') if s.strip()]
        for stmt in statements:
            if '=' in stmt:
                var, expr = stmt.split('=', 1)
                var = var.strip()
                expr = expr.strip()
                try:
                    env = new_vars.copy()
                    val = eval(expr, {"__builtins__": {}}, env)
                    new_vars[var] = val
                except:
                    pass

        return new_vars

    def step(self, config: Tuple, inp: str) -> List[Tuple[str, str, Tuple]]:
        """
        Transition from config=(state, **vars) under input inp.
        Returns list of (input, output, next_config) tuples.
        """
        state = config[0]
        var_values = {}
        for i, var_name in enumerate(self.var_names):
            if i + 1 < len(config):
                var_values[var_name] = config[i + 1]

        succ = []
        for t in self.transitions:
            if t.get("from") == state and t.get("input") == inp:
                if self._eval_guard(state, t.get("guard", ""), var_values):
                    new_vars = self._apply_action(t.get("action", ""), var_values)
                    next_state = t.get("to")
                    output = t.get("output", "")
                    next_config = (next_state,) + tuple(new_vars[v] for v in self.var_names)
                    succ.append((inp, output, next_config))

        return succ

    def build_l_equivalent(self, l: int = 3) -> Tuple[List[FrozenSet], List[Tuple]]:
        """
        Build l-equivalent (belief/successor tree) starting from self.initial_configs.

        Args:
            l: maximum depth of l-equivalent

        Returns:
            (nodes, edges) where:
            - nodes[i] = frozenset of configurations at belief node i
            - edges = list of (src_idx, inp, out, dst_idx, depth)
        """
        node_id = {self.initial_configs: 0}
        nodes = [self.initial_configs]
        edges = []

        q = deque([(0, 0)])

        while q:
            nid, depth = q.popleft()
            if depth == l:
                continue

            belief = nodes[nid]

            for inp in self.inputs:
                out_to_succ = defaultdict(set)
                for cfg in belief:
                    for _, out, ncfg in self.step(cfg, inp):
                        out_to_succ[out].add(ncfg)

                for out, succ_set in out_to_succ.items():
                    belief_p = frozenset(succ_set)
                    if belief_p not in node_id:
                        node_id[belief_p] = len(nodes)
                        nodes.append(belief_p)
                        q.append((node_id[belief_p], depth + 1))

                    edges.append((nid, inp, out, node_id[belief_p], depth + 1))

        return nodes, edges

    def _proj_states(self, belief: FrozenSet[Tuple]) -> FrozenSet[str]:
        """Project configuration belief to state belief."""
        return frozenset(cfg[0] for cfg in belief)

    def _check_duplicate(self, A, B):
        lens = {len(t) for t in A}
        for i in lens:
            if i <= len(B) and B[-i:] in A:
                return False
        return True

    def generate_homing_traces(self, nodes: List[FrozenSet], edges: List[Tuple], l: int = 3) -> List[Dict[str, Any]]:
        """
        Generate homing traces by traversing actual edges from l-equivalent.
        """

        # Build adjacency: adj[nid][inp] = [(out, dst, depth)]
        adj = defaultdict(lambda: defaultdict(list))
        for src, inp, out, dst, d in edges:
            adj[src][inp].append((out, dst, d))

        homing = []
        visited = set()
        seen_sequences = set()

        dq = deque([(0, tuple(), 0)])  # (node, trace, depth)

        while dq:
            nid, trace, depth = dq.popleft()
            belief = nodes[nid]
            state_belief = self._proj_states(belief)

            # Check if homing (singleton state belief)
            is_singleton = len(state_belief) == 1

            if is_singleton and len(trace) > 0:
                if trace not in seen_sequences and self._check_duplicate(seen_sequences, trace):
                    seen_sequences.add(trace)
                    identified_state = next(iter(state_belief))
                    homing.append({
                        'length': len(trace),
                        'trace': ' '.join([f'{i}/{o}' for i, o in trace]),
                        'state': identified_state
                    })
                # Early convergence
                continue

            if depth >= l:
                continue

            # Expand on actual edges only
            for inp in self.inputs:
                if inp not in adj[nid]:
                    continue

                for out, dst, d in adj[nid][inp]:
                    if d == depth + 1:
                        new_trace = trace + ((inp, out),)
                        key = (dst, new_trace)

                        if key in visited:
                            continue
                        visited.add(key)

                        dq.append((dst, new_trace, depth + 1))

        # Sort by length, then state
        homing.sort(key=lambda x: (x['length'], x['state']))

        return homing

# ============================================================================
if __name__ == '__main__':
    print("This module provides TFSM and EFSM modeling, and homing traces generation.")
