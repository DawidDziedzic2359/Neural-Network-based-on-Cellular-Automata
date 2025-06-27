class CellularAutomaton:
    def __init__(self, rule):
        self.rule = [int(bit) for bit in f"{rule:08b}"[::-1]]

    def apply_rule(self, state):
        new_state = [0] * len(state)
        for i in range(len(state)):
            left = state[(i - 1) % len(state)]
            center = state[i]
            right = state[(i + 1) % len(state)]
            rule_index = (left << 2) | (center << 1) | right
            new_state[i] = self.rule[rule_index]
        return new_state[1:-1]