from models.mcts import State, MonteCarloTreeSearchNode

str_rep = [
    'S.SS...A',
    '........',
    '.S......',
    '........',
    '......SS',
    '....S...',
    '........',
    'M.......'
]
str_rep = [e.replace('.', '-') for e in str_rep]
initial_state = State(str_rep)

root = MonteCarloTreeSearchNode(state=initial_state)
selected_node = root.best_action()
