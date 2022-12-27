import copy
from sys import exit

import numpy as np
from collections import defaultdict

np.random.seed(0)


class State:
    def __init__(self, state_rep: list[str], size_x=8, size_y=8, past_mariah_pos=None):
        self.state_rep = state_rep
        self.size_x = size_x
        self.size_y = size_y
        self.board = np.zeros([size_y, size_x], dtype=np.object)
        self.statues = []
        self.maria_pos = []
        self.anna_pos = [0, size_x - 1]
        self.orig_maria_pos = [size_y - 1, 0]
        self.past_mariah_pos = past_mariah_pos or []
        # self.all_moves = ['up', 'up-left', 'up-right', 'down', 'down-left', 'down-right', 'left', 'right', 'stay']
        self.all_moves = ['up', 'up-left', 'up-right', 'left', 'right']
        self._parse()

    def _parse(self):
        for i, row in enumerate(self.state_rep):
            chars = list(row)
            for j, c in enumerate(chars):
                if c == 'A':
                    self.anna_pos = [i, j]
                elif c == 'M':
                    self.maria_pos = [i, j]
                    self.past_mariah_pos.append(self.maria_pos)
                elif c == 'S':
                    self.statues.append([i, j])
            self.board[i] = chars

    def __str__(self):
        return '\n'.join([''.join(row) for row in self.board.astype(np.str).tolist()])

    def is_terminal(self):
        return self.maria_pos == self.anna_pos or self.board[tuple(self.orig_maria_pos)] == 'S'

    def is_win(self):
        return self.maria_pos == self.anna_pos

    def simulate_move(self, direction, old_pos: list):
        new_pos = old_pos.copy()
        if direction == 'up':
            new_pos[0] -= 1
        elif direction == 'up-right':
            new_pos[0] -= 1
            new_pos[1] += 1
        elif direction == 'up-left':
            new_pos[0] -= 1
            new_pos[1] -= 1
        elif direction == 'down':
            new_pos[0] += 1
        elif direction == 'down-right':
            new_pos[0] += 1
            new_pos[1] += 1
        elif direction == 'down-left':
            new_pos[0] += 1
            new_pos[1] -= 1
        elif direction == 'left':
            new_pos[1] -= 1
        elif direction == 'right':
            new_pos[1] += 1

        return new_pos

    def move_statues(self):
        old_statues = copy.deepcopy(self.statues)
        statues = []
        for i in range(len(old_statues)):
            statue = old_statues[i]
            if statue[0] < self.size_y - 1:
                statue[0] += 1
                statues.append(statue)
        return statues

    @staticmethod
    def from_positions(statues, maria_pos, anna_pos=None, size_x=8, size_y=8, past_mariah_pos=None):
        if not anna_pos:
            anna_pos = [0, 7]
        board = np.zeros((size_y, size_x), dtype=np.object)
        for statue in statues:
            board[tuple(statue)] = 'S'
        board[tuple(anna_pos)] = 'A'
        board[tuple(maria_pos)] = 'M'
        board[board == 0] = '-'
        state_rep = [''.join(row) for row in board]
        return State(state_rep, size_x, size_y, past_mariah_pos)

    def get_legal_actions(self):
        """
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        """
        moves = self.all_moves.copy()
        m_pos = self.maria_pos
        to_remove = set()
        if m_pos[0] == 0:
            to_remove |= {'up', 'up-right', 'up-left'}
        # elif m_pos[0] == self.size_y - 1:
            # to_remove |= {'down', 'down-right', 'down-left'}
        if m_pos[1] == 0:
            # to_remove |= {'left', 'up-left', 'down-left'}
            to_remove |= {'left', 'up-left'}
        elif m_pos[1] == self.size_x - 1:
            # to_remove |= {'right', 'up-right', 'down-right'}
            to_remove |= {'right', 'up-right'}
        [moves.remove(m) for m in to_remove]
        final_moves = moves.copy()
        new_statues = self.move_statues()
        all_statues = self.statues + new_statues
        for m in moves:
            new_m_pos = self.simulate_move(m, m_pos)
            if new_m_pos in all_statues or (new_m_pos in self.past_mariah_pos):
                final_moves.remove(m)
        return final_moves

    def is_game_over(self):
        """
        Modify according to your game or
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        """
        return self.is_terminal()

    def game_result(self, path_len):
        """
        Modify according to your game or
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        """
        if np.abs(np.array(self.maria_pos) - np.array(self.anna_pos)).sum() <= 2:  #and path_len < self.size_y * 1.5:
            return 1
        return -1

    def move(self, action, _print=False):
        """
        Modify according to your game or
        needs. Changes the state of your
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board
        position is empty. If you place x in
        row 2 column 3, then it would be some
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns
        the new state after making a move.
        """
        m_pos = self.simulate_move(action, self.maria_pos)
        if _print:
            print(f'{self.maria_pos} -> {action} -> {m_pos}')
        statues = self.move_statues()
        return self.from_positions(statues, m_pos, past_mariah_pos=copy.deepcopy(self.past_mariah_pos))


class MonteCarloTreeSearchNode:
    def __init__(self, state: State, parent: 'MonteCarloTreeSearchNode' = None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._max_len = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.action_path = []
        if self.parent:
            self.action_path = self.parent.action_path + [self.parent_action]

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        actions = []
        n = self
        dist = np.abs(np.array(self.state.maria_pos) - np.array(self.state.anna_pos)).sum()
        while n.parent:
            actions.append(n.parent_action)
            n = n.parent
        good_actions = actions.count('up-right') + actions.count('up') + actions.count('right')
        return - dist

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over() or (self.is_fully_expanded() and not self.children)

    def rollout(self):
        current_rollout_state = self.state
        path_len = 0
        actions = []
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            if not possible_moves:
                break

            action = self.rollout_policy(possible_moves)
            actions.append(action)
            current_rollout_state = current_rollout_state.move(action)
            path_len += 1
        if path_len > self._max_len:
            self._max_len = path_len
        n = self
        while n.parent:
            n = n.parent
            path_len += 1
        return current_rollout_state.game_result(path_len), current_rollout_state, actions

    def backpropagate(self, result, path_len):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self._max_len < path_len:
            self._max_len = path_len
        if self.parent:
            self.parent.backpropagate(result, path_len + 1)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, draw=False):
        simulation_no = 100

        for i in range(simulation_no):
            print(f'SIMULATION {i}')
            v = self._tree_policy()
            reward, terminal_state, actions = v.rollout()
            action_path = v.action_path + actions
            if terminal_state.is_win():
                print('WIN')
                print(f'WINNING ACTIONS: {action_path}')
                if draw:
                    print(self.state)
                    s = self.state.move(action_path[0], _print=True)
                    print(s)
                    for action in action_path[1:]:
                        s = s.move(action, _print=True)
                        print(s)
                exit()
            print('LOSE')
            v.backpropagate(reward, v._max_len)
        return self.best_child(c_param=0.)


def main(draw=False):
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
    selected_node = root.best_action(draw)
    return selected_node, root.state











