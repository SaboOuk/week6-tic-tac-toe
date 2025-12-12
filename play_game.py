# play_game.py
import random
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent

MODEL_FILE = "trained_model.json"

def play_against_ai():
    """Human (X) vs AI (O)."""
    game = TicTacToe()
    ai = QLearningAgent(player_id=-1)
    if not ai.load_model(MODEL_FILE):
        print("❌ No trained model found. Run train_agent.py first.")
        return
    ai.set_training(False)

    print("\n" + "=" * 50)
    print("🎮 PLAY AGAINST THE AI")
    print("=" * 50)
    print("You are X. AI is O.")
    print("Enter moves as: row,col  (example: 1,2)\n")

    human_turn = random.choice([True, False])
    if human_turn:
        print("You go first!\n")
    else:
        print("AI goes first!\n")

    game.reset()

    while True:
        game.display()
        winner = game.check_winner()
        if winner is not None:
            if winner == 1:
                print("🎉 YOU WIN!")
            elif winner == -1:
                print("🤖 AI WINS!")
            else:
                print("🤝 IT'S A TIE!")
            break

        if human_turn:
            available = game.get_available_actions()
            print(f"Available moves: {available}")
            while True:
                move = input("Your move (row,col): ").strip()
                try:
                    r, c = move.split(",")
                    r, c = int(r), int(c)
                    if (r, c) in available:
                        game.make_move((r, c), 1)
                        break
                    else:
                        print("Invalid move (not available). Try again.")
                except Exception:
                    print("Format error. Use row,col (example: 0,2)")
        else:
            # AI is O (-1)
            state = game.get_state()

            # IMPORTANT: AI was trained as X moves, but it can still choose from Q-table patterns.
            # We'll flip the board perspective so AI can reuse learned values:
            # Convert state to "AI-as-X" view by multiplying by -1 (swap X/O)
            flipped_state = tuple([-v for v in state])

            available = game.get_available_actions()
            action = ai.choose_action(flipped_state, available)
            if action is None:
                print("AI has no moves.")
                break
            game.make_move(action, -1)
            print(f"AI moves to: {action}")

        human_turn = not human_turn

def watch_ai_vs_random(games=20):
    """Show that trained AI generally does better than random."""
    game = TicTacToe()
    ai = QLearningAgent(player_id=1)
    if not ai.load_model(MODEL_FILE):
        print("❌ No trained model found. Run train_agent.py first.")
        return
    ai.set_training(False)

    stats = {"ai_wins": 0, "random_wins": 0, "ties": 0}

    for _ in range(games):
        game.reset()
        while True:
            # AI plays X (1)
            state = game.get_state()
            avail = game.get_available_actions()
            a = ai.choose_action(state, avail)
            game.make_move(a, 1)

            winner = game.check_winner()
            if winner is not None:
                if winner == 1:
                    stats["ai_wins"] += 1
                elif winner == -1:
                    stats["random_wins"] += 1
                else:
                    stats["ties"] += 1
                break

            # Random plays O (-1)
            avail = game.get_available_actions()
            if avail:
                opp = random.choice(avail)
                game.make_move(opp, -1)

            winner = game.check_winner()
            if winner is not None:
                if winner == 1:
                    stats["ai_wins"] += 1
                elif winner == -1:
                    stats["random_wins"] += 1
                else:
                    stats["ties"] += 1
                break

    print("\n" + "=" * 50)
    print("🤖 AI vs 🎲 Random Results")
    print("=" * 50)
    print(f"Games: {games}")
    print(f"AI wins: {stats['ai_wins']}")
    print(f"Random wins: {stats['random_wins']}")
    print(f"Ties: {stats['ties']}")
    print("=" * 50)

def about_q_learning():
    print("\n" + "=" * 50)
    print("📘 ABOUT Q-LEARNING")
    print("=" * 50)
    print("Q-Learning is reinforcement learning where an agent learns by trial and error.")
    print("Q = Quality of taking an action in a state.")
    print("The agent updates Q-values based on rewards:")
    print("  Win = +10, Loss = -10, Tie = -5")
    print("Over many games, good moves get higher Q-values.")
    print("=" * 50)

def main():
    while True:
        print("\n" + "-" * 60)
        print("✅ TIC-TAC-TOE WITH Q-LEARNING AI")
        print("-" * 60)
        print("1. Play against trained AI")
        print("2. Watch AI vs random (quick proof)")
        print("3. Learn about Q-Learning")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()
        if choice == "1":
            play_against_ai()
        elif choice == "2":
            watch_ai_vs_random(games=20)
        elif choice == "3":
            about_q_learning()
        elif choice == "4":
            print("\nThanks for playing! 👋")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
