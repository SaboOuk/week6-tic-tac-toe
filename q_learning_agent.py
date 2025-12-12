# q_learning_agent.py
import json
import os
import random

class QLearningAgent:
    """
    Q-Learning agent for Tic-Tac-Toe.

    Q-table maps (state, action) -> Q value.
    State is a tuple of 9 ints: (0, 1, -1, ...)
    Action is a tuple: (row, col)
    """

    def __init__(self, player_id=1, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.training = True
        self.q_table = {}  # key: "state|r,c" -> float

    def _key(self, state, action):
        # state is tuple(9), action is (r,c)
        return f"{state}|{action[0]},{action[1]}"

    def get_q_value(self, state, action):
        k = self._key(state, action)
        if k not in self.q_table:
            self.q_table[k] = 0.0
        return self.q_table[k]

    def choose_action(self, state, available_actions):
        """Epsilon-greedy selection."""
        if not available_actions:
            return None

        # Explore
        if self.training and random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploit (best known)
        best_action = None
        best_value = float("-inf")
        for a in available_actions:
            q = self.get_q_value(state, a)
            if q > best_value:
                best_value = q
                best_action = a

        # If everything tied (0.0), best_action might still be first; that's fine.
        return best_action if best_action is not None else random.choice(available_actions)

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """
        Q-learning update:
        Q(s,a) = Q(s,a) + lr * (reward + gamma*max_a' Q(s',a') - Q(s,a))
        """
        current_q = self.get_q_value(state, action)

        if next_actions:
            max_next_q = max(self.get_q_value(next_state, a2) for a2 in next_actions)
        else:
            max_next_q = 0.0

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[self._key(state, action)] = new_q

    def set_training(self, training: bool):
        self.training = training
        if not training:
            self.exploration_rate = 0.0  # no random moves in play mode

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.q_table, f, indent=2)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            self.q_table = json.load(f)
        print(f"✅ Model loaded from {filepath}")
        return True

    def get_stats(self):
        # rough stats for README printouts
        states = set()
        for k in self.q_table.keys():
            # k looks like "(...)|r,c"
            state_part = k.split("|", 1)[0]
            states.add(state_part)
        avg_q = sum(self.q_table.values()) / len(self.q_table) if self.q_table else 0.0
        return {
            "states_learned": len(states),
            "total_q_values": len(self.q_table),
            "avg_q_value": avg_q
        }
