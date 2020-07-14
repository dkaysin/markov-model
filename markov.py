# Multi-Layer Markov Model
#
# State is the global state that contains all available information. State_0 is current state.
# State must be of a hashable type (e.g., tuple).
#
# Each layer of the MarkovModel interprets the state in its own way by converting it
# into a MarkovMemory using the "mapper" function MarkovLayer.f_map
#
# Only one layer is used to transition to the next State (state_1).
# This layer is chosen via the "selector" function MarkovModel.f_select based on ranking of all layers.
#
# After the layer is chosen a node that represents current state (node_0) is determined and the next node (node_1)
# is selected among all node_0's children nodes based on weighted random sampling.
# Probability of each child node to be selected is determined based on its weight and
# the "weighter" function MarkovLayer.f_weight
#
# After node_1 is selected, the next state (state_1) is generated based on state_0 and node_1's memory via
# the "reducer" function MarkovLayer.f_reduce


import random
from typing import NewType

IS_DEBUG = True


MarkovState = NewType('MarkovState', tuple)
MarkovMemory = NewType('MarkovMemory', tuple)


class MarkovNode:
    def __init__(self, _memory: MarkovMemory):
        self.memory = _memory    # MarkovMemory
        self.children = {}       # key - MarkovMemory, value - pointer to MarkovNode
        self.weights = {}        # key - MarkovMemory, value - weight (float)
        self.is_head = False
        self.is_tail = False

    def __repr__(self):
        return '%s:%s' % (self.memory, self.weights)


class MarkovEdge:

    def __init__(self, _state_0: MarkovState = None, _state_1: MarkovState = None,
                 _weight=0, _is_head=False, _is_tail=False):
        self.state_0 = _state_0
        self.state_1 = _state_1
        self.weight = _weight
        self.is_head = _is_head
        self.is_tail = _is_tail

    def __repr__(self):
        return 'Edge from %s to %s with weight %s' % (self.state_0, self.state_1, self.weight)


class MarkovLayer:

    def __init__(self, _f_map, _f_weight, _f_reduce):
        self.nodes = {}
        self.f_map = _f_map
        self.f_weight = _f_weight
        self.f_reduce = _f_reduce

    def __repr__(self):
        return str([k for (k, _) in self.nodes.items()])

    def get_node(self, _state: MarkovState) -> MarkovNode:
        memory = self.f_map(_state)
        node = self.nodes.get(memory)
        return node

    def get_or_add_node(self, _state: MarkovState) -> MarkovNode:
        node = self.get_node(_state)
        if node is None:
            memory = self.f_map(_state)
            node = MarkovNode(memory)
            self.nodes[memory] = node
        return node

    def get_or_add_child(self, _node_0: MarkovNode, _state_1: MarkovState,
                         _weight, _is_head=False, _is_tail=False) -> MarkovNode:
        memory_1 = self.f_map(_state_1)
        node_1 = _node_0.children.get(memory_1)
        if node_1 is None:
            node_1 = self.get_or_add_node(_state_1)
            _node_0.children[memory_1] = node_1
            _node_0.weights[memory_1] = 0
        _node_0.weights[memory_1] += _weight
        _node_0.is_head = _is_head
        node_1.is_tail = _is_tail
        return node_1

    def get_ranking(self, _state: MarkovState) -> float:
        node = self.get_node(_state)
        if node is None:
            rank = 0
        else:
            rank = sum(node.weights.values())
        return rank

    def iterate(self, _state_0: MarkovState) -> MarkovState:
        if _state_0 is None:
            return None
        node_0 = self.get_node(_state_0)
        if node_0 is None or len(node_0.children) == 0:
            return None
        memory_1 = sample_from_dict(self.f_weight(node_0, _state_0))
        if memory_1 is None:
            return None  # empty exemplar of class is needed here
        else:
            return self.f_reduce(_state_0, memory_1)


class MarkovModel:
    def __init__(self, _f_select):
        self.layers = {}
        self.f_select = _f_select

    def create_layer(self, _name, _f_map, _f_weight, _f_reduce) -> MarkovLayer:
        new_layer = MarkovLayer(_f_map, _f_weight, _f_reduce)
        self.layers[_name] = new_layer
        return new_layer

    def add_edge(self, _edge: MarkovEdge):
        for (_, v) in self.layers.items():
            node_0 = v.get_or_add_node(_edge.state_0)
            _ = v.get_or_add_child(node_0, _edge.state_1, _edge.weight, _edge.is_head, _edge.is_tail)

    def choose_layer(self, _state_0: MarkovState) -> MarkovLayer:
        ranking_table = {k: v.get_ranking(_state_0) for (k, v) in self.layers.items()}
        chosen_key = self.f_select(ranking_table)
        if IS_DEBUG:
            print(ranking_table, chosen_key)
        chosen_layer = self.layers[chosen_key]
        return chosen_layer

    def generate_chain(self, _state_0: MarkovState, _n=-1) -> MarkovState:
        s = _state_0
        while s is not None and _n != 0:
            _n += -1
            chosen_layer = self.choose_layer(s)
            s = chosen_layer.iterate(s)
            if s is not None:
                yield s


# Random sampling from dict {k:v} where v is the probability weight
# Weights do not need to sum to 1
def sample_from_dict(d):
    _max = sum(d.values())
    if len(d) == 0:
        raise ValueError('Input dictionary has length 0')
    elif _max == 0:
        return list(d.keys())[0]
    else:
        rolled_value = random.uniform(0, _max)
    total = 0
    for k, v in d.items():
        total += v
        if rolled_value <= total:
            return k
    raise ValueError('Rolled random value larger than total weights of items in the dict', rolled_value, total)
