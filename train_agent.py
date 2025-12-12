# train_agent.py
import os
import random
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent

RESULTS_DIR = "results"
PROGRESS_FILE = os.path.join(RESULTS_DIR, "training_progress.txt")
MODEL_FILE = "trained_model.json"

class Trainer:
    """Train the Q-learning agent vs a random opponent."""

    def __init__(self):
        self.game = TicTacToe()
        self.agent = QLearningAgent(player_id=1)  # agent is X by default
        self.wins = {"agent": 0, "random": 0, "tie": 0}

    def _write_progress(self, text):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def play_training_game(self):
        """
        Run one training game:
        - Agent plays X (1)
        - Random opponent plays O (-1)
        We collect agent transitions, then apply reward backward at the end.
        """
        self.game.reset()
        transitions = []  # (state, action, next_state, next_actions)

        while True:
            # Agent turn (X)
            state = self.game.get_state()
            available = self.game.get_available_actions()
            if not available:
                break

            action = self.agent.choose_action(state, available)
            self.game.make_move(action, 1)

            next_state = self.game.get_state()
            next_actions = self.game.get_available_actions()

            transitions.append((state, action, next_state, next_actions))

            winner = self.game.check_winner()
            if winner is not None:
                self._process_game_end(winner, transitions)
                break

            # Random opponent turn (O)
            available = self.game.get_available_actions()
            if available:
                opp_action = random.choice(available)
                self.game.make_move(opp_action, -1)

            winner = self.game.check_winner()
            if winner is not None:
                self._process_game_end(winner, transitions)
                break

    def _process_game_end(self, winner, transitions):
        # Assign outcome reward
        if winner == 1:
            reward = 10
            self.wins["agent"] += 1
        elif winner == -1:
            reward = -10
            self.wins["random"] += 1
        else:
            reward = -5
            self.wins["tie"] += 1

        # Update Q-values backward so earlier moves get discounted reward
        transitions = list(reversed(transitions))
        for i, (state, action, next_state, next_actions) in enumerate(transitions):
            if i == 0:
                # last agent move gets full reward
                self.agent.update_q_value(state, action, reward, next_state, [])
            else:
                # earlier moves get discounted reward, and still consider next state's best Q
                discounted = reward * (self.agent.discount_factor ** i)
                self.agent.update_q_value(state, action, discounted, next_state, next_actions)

    def train(self, episodes=1000):
        # fresh progress file each run
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write("TRAINING TIC-TAC-TOE AI (Q-LEARNING)\n")
            f.write("=================================\n")

        print("🤖 TRAINING TIC-TAC-TOE AI")
        print("-" * 40)

        checkpoints = [100, 500, 1000, 5000, 10000]
        checkpoints = [c for c in checkpoints if c <= episodes] + [episodes]
        checkpoints = sorted(set(checkpoints))

        for ep in range(1, episodes + 1):
            self.play_training_game()

            if ep in checkpoints:
                win_rate = (self.wins["agent"] / ep) * 100
                stats = self.agent.get_stats()

                line = (
                    f"Episode {ep}: "
                    f"Wins={self.wins['agent']} "
                    f"Losses={self.wins['random']} "
                    f"Ties={self.wins['tie']} "
                    f"WinRate={win_rate:.2f}% "
                    f"StatesLearned={stats['states_learned']} "
                    f"AvgQ={stats['avg_q_value']:.3f}"
                )

                print(line)
                self._write_progress(line)

                # Reduce exploration over time (more “serious” later)
                if ep == 500:
                    self.agent.exploration_rate = 0.2
                elif ep == 1000:
                    self.agent.exploration_rate = 0.1
                elif ep == 5000:
                    self.agent.exploration_rate = 0.05

        self.agent.save_model(MODEL_FILE)
        self._write_progress(f"\nSaved model to {MODEL_FILE}")

def main():
    print("\n" + "=" * 60)
    print("🎯 REINFORCEMENT LEARNING DEMONSTRATION")
    print("Training an AI to play Tic-Tac-Toe")
    print("=" * 60)

    print("\nTraining Options:")
    print("1. Quick training (1,000 games)")
    print("2. Standard training (5,000 games)")
    print("3. Intensive training (10,000 games)")

    choice = input("\nChoice (1-3): ").strip()
    episodes = {"1": 1000, "2": 5000, "3": 10000}.get(choice, 1000)

    print(f"\nTraining with {episodes} games...\nWatch the AI improve!\n")

    trainer = Trainer()
    trainer.train(episodes)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("Now run: python play_game.py")
    print("Model file: trained_model.json")
    print("Progress log: results/training_progress.txt")

if __name__ == "__main__":
    main()
