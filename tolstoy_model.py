import math
from copy import deepcopy

from markov import MarkovModel, MarkovEdge, sample_from_dict
from markov import MarkovNode, MarkovState, MarkovMemory  # for type-checking


punct_mid_sentence = [',', '-', ':', ';']
punct_end_sentence = ['.', '?', '!']
punct_remove = ['"', '“', '”']
nb_prefixes = ['a', 'an', 'the']


# Some voodoo magic here
def my_selector(_ranking: dict) -> str:
    if _ranking['2-word'] > 1 and _ranking['3-word'] / _ranking['2-word'] > 0.75:
        return sample_from_dict({k: math.log(v + 1) for (k, v) in _ranking.items()})
    elif _ranking['2-word'] > 1 and _ranking['3-word'] / _ranking['2-word'] > 0.5:
        return '2-word'
    if _ranking['3-word'] > 0:
        return '3-word'
    elif _ranking['2-word'] > 0 and _ranking['1-word'] > 0:
        return sample_from_dict({k: math.log(v + 1) for (k, v) in _ranking.items()})
    else:
        return '1-word'


def my_weighter(_node_0: MarkovNode, _state_0: MarkovState) -> dict:
    return {
        v.memory: _node_0.weights[k] * weight_coeff_len(len(_state_0)) if v.is_tail else _node_0.weights[k]
        for (k, v) in _node_0.children.items()
    }


def weight_coeff_len(l: int) -> float:
    n = max(0, min(10, l) - 5)
    return math.exp(n) - 0.9


def my_reducer(_state_0: MarkovState, _memory_1: MarkovMemory) -> MarkovState:
    return MarkovState(_state_0 + (_memory_1[-1],))


txt = open('corpus.txt', encoding='utf8').read()
corpus = txt.split()

model = MarkovModel(my_selector)
model.create_layer(
    '3-word',
    lambda _state_0: _state_0[-3:],
    my_weighter,
    my_reducer
)
model.create_layer(
    '2-word',
    lambda _state_0: _state_0[-2:],
    my_weighter,
    my_reducer
)
model.create_layer(
    '1-word',
    lambda _state_0: _state_0[-1:],
    my_weighter,
    my_reducer
)

state_empty = MarkovState((None,))

state_0 = deepcopy(state_empty)
prefix = []

for word in corpus:

    # text preparation
    for punct in punct_remove:
        word = word.replace(punct, '')
    if word in nb_prefixes:
        prefix += [word]
        continue
    if len(prefix) != 0:
        word = ' '.join(prefix + [word])
        prefix = []

    # prepare MarkovEdge object to be added to the model
    current_edge = MarkovEdge()
    current_edge.state_0 = deepcopy(state_0)

    state_1 = my_reducer(state_0, MarkovMemory((word,)))
    current_edge.state_1 = state_1
    current_edge.weight = 1

    if len(state_0) == 0:
        current_edge.is_head = True

    if word[-1] in punct_end_sentence:
        current_edge.is_tail = True
        state_0 = deepcopy(state_empty)
    else:
        state_0 = state_1

    model.add_edge(current_edge)


starter = 'When was the last time'  # ...
input_state = MarkovState(tuple(starter.split(' ')))

# Generate 10 continuations of the "starter" string and print them
for _ in range(10):
    result = list(model.generate_chain(input_state))
    result = [x[-1] for x in result]
    result = list(input_state) + result
    result = [x for x in result if x is not None]
    result_str = ' '.join(result)
    print(result_str)
