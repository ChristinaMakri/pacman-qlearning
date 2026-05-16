import random
import pickle
import os
from collections import defaultdict
from constants import UP, DOWN, LEFT, RIGHT, STOP, FREIGHT, SPAWN

ACTIONS = [UP, DOWN, LEFT, RIGHT]


def _default_q():
    return {a: 0.0 for a in ACTIONS}


class QLearningAgent:
    """
    Tabular Q-learning agent for Pac-Man.

    State: tuple of 14 booleans/ints derived from the current node graph.
    Actions: UP, DOWN, LEFT, RIGHT (constants from constants.py).
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # exploration probability
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(_default_q)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def get_state(self, pacman, ghosts, pellets):
        """
        Build a discrete state tuple from the live game objects.

        Features (14 values):
          can_up, can_down, can_left, can_right  -- wall/access flags
          ghost_danger_up/down/left/right        -- dangerous ghost in path
          ghost_freight_up/down/left/right       -- eatable (FREIGHT) ghost in path
          pellet_dx, pellet_dy                   -- direction to nearest pellet (-1/0/1)
        """
        node = pacman.node

        valid = tuple(pacman.validDirection(d) for d in ACTIONS)

        ghost_danger = []
        ghost_freight = []
        for d in ACTIONS:
            danger, freight = self._path_check_ghosts(node, d, ghosts, depth=5)
            ghost_danger.append(danger)
            ghost_freight.append(freight)

        pdx, pdy = self._nearest_pellet_dir(pacman.position, pellets)

        return valid + tuple(ghost_danger) + tuple(ghost_freight) + (pdx, pdy)

    def _path_check_ghosts(self, start_node, direction, ghosts, depth=5):
        """
        Walk up to `depth` nodes in `direction` and check for ghosts.
        Returns (danger: bool, freight: bool).
        """
        path_ids = set()
        node = start_node
        for _ in range(depth):
            nxt = node.neighbors.get(direction)
            if nxt is None:
                break
            path_ids.add(id(nxt))
            node = nxt

        danger = False
        freight = False
        for ghost in ghosts:
            if id(ghost.node) in path_ids:
                mode = ghost.mode.current
                if mode == FREIGHT:
                    freight = True
                elif mode != SPAWN:
                    danger = True
        return danger, freight

    def _nearest_pellet_dir(self, position, pellets):
        """Quantized direction (-1/0/1 each axis) toward the nearest pellet."""
        if not pellets.pelletList:
            return 0, 0
        best_dist = float('inf')
        best_diff = None
        for pellet in pellets.pelletList:
            diff = pellet.position - position
            dist = diff.magnitudeSquared()
            if dist < best_dist:
                best_dist = dist
                best_diff = diff
        if best_diff is None:
            return 0, 0
        dx = 0 if abs(best_diff.x) < 8 else (1 if best_diff.x > 0 else -1)
        dy = 0 if abs(best_diff.y) < 8 else (1 if best_diff.y > 0 else -1)
        return dx, dy

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state, valid_dirs):
        """Epsilon-greedy selection restricted to valid_dirs."""
        if not valid_dirs:
            return STOP
        if random.random() < self.epsilon:
            return random.choice(valid_dirs)
        q = self.q_table[state]
        return max(valid_dirs, key=lambda d: q.get(d, 0.0))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, next_valid_dirs):
        """Standard Q-learning update (Bellman equation)."""
        if next_state is not None and next_valid_dirs:
            max_next_q = max(
                self.q_table[next_state].get(d, 0.0) for d in next_valid_dirs
            )
        else:
            max_next_q = 0.0

        current_q = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path="qtable.pkl"):
        payload = {
            "q_table": {s: dict(a) for s, a in self.q_table.items()},
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[Agent] Saved {len(self.q_table)} states → {path}")

    def load(self, path="qtable.pkl"):
        if not os.path.exists(path):
            print(f"[Agent] No Q-table at {path}, starting fresh.")
            return False
        with open(path, "rb") as f:
            payload = pickle.load(f)
        raw = payload.get("q_table", payload)
        self.q_table = defaultdict(_default_q, {s: dict(a) for s, a in raw.items()})
        self.epsilon = payload.get("epsilon", self.epsilon)
        print(f"[Agent] Loaded {len(self.q_table)} states from {path} (ε={self.epsilon:.4f})")
        return True
