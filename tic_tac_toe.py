# tic_tac_toe.py
import numpy as np

class TicTacToe:
    """
    Tic-Tac-Toe game environment.
    Board values:
      0 = empty
      1 = X
     -1 = O
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board and set current player to X (1)."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """Return a hashable state representation."""
        # tuple is easier/cleaner than string, but we can stringify later for JSON keys
        return tuple(self.board.flatten().tolist())

    def get_available_actions(self):
        """Return list of empty positions (row, col)."""
        actions = []
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    actions.append((r, c))
        return actions

    def make_move(self, action, player):
        """Place player's mark at (row, col) if empty."""
        r, c = action
        if self.board[r, c] != 0:
            return False
        self.board[r, c] = player
        return True

    def check_winner(self):
        """
        Returns:
          1 if X wins
         -1 if O wins
          0 if tie
          None if game continues
        """
        # rows
        for r in range(3):
            s = self.board[r, :].sum()
            if s == 3:
                return 1
            if s == -3:
                return -1

        # cols
        for c in range(3):
            s = self.board[:, c].sum()
            if s == 3:
                return 1
            if s == -3:
                return -1

        # diagonals
        d1 = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        d2 = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        if d1 == 3 or d2 == 3:
            return 1
        if d1 == -3 or d2 == -3:
            return -1

        # tie or continue
        if len(self.get_available_actions()) == 0:
            return 0
        return None

    def display(self):
        """Print the board nicely."""
        sym = {0: " ", 1: "X", -1: "O"}
        print("\n   0   1   2")
        print("  -----------")
        for r in range(3):
            row = f"{r}  " + " | ".join(sym[self.board[r, c]] for c in range(3))
            print(row)
            print("  -----------")
