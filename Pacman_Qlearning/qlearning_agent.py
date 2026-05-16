import random
import pickle
import os
from collections import defaultdict
from constants import UP, DOWN, LEFT, RIGHT, STOP, FREIGHT, SPAWN, TILEWIDTH

ACTIONS = [UP, DOWN, LEFT, RIGHT]


def _default_q():
    return {a: 0.0 for a in ACTIONS}


class QLearningAgent:
    """
    Q(λ) agent for Pac-Man with eligibility traces and richer state features.

    State (21 values total):
        can_move × 4          — wall / access flags per direction
        ghost_danger × 4      — 0=none, 1=far (4-6 nodes), 2=close (1-3 nodes)
        ghost_freight × 4     — same scale for eatable freight ghosts
        power_pellet × 4      — power pellet present in that direction path
        any_freight           — global bool: any ghost currently eatable
        pellet_dx, pellet_dy  — direction to nearest pellet (−1 / 0 / +1)
        pellet_dist           — 0=close (<4 tiles), 1=medium, 2=far

    Algorithm: Q(λ) with replacing eligibility traces.
        δ  = r + γ·max_a′Q(s′,a′) − Q(s,a)
        e(s,a) ← 1   (replacing trace for taken action)
        e(s,a) ← γλ·e(s,a)  (decay all traces each step)
        Q(s,a) += α·δ·e(s,a)  for all (s,a) with e>0
    """

    def __init__(self, alpha=0.1, gamma=0.9, lam=0.8,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.alpha        = alpha
        self.gamma        = gamma
        self.lam          = lam          # trace decay parameter λ
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(_default_q)
        self.traces  = {}                # eligibility traces: {state: {action: float}}

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def get_state(self, pacman, ghosts, pellets):
        node = pacman.node

        valid = tuple(pacman.validDirection(d) for d in ACTIONS)

        danger_lvl  = []
        freight_lvl = []
        power_dir   = []
        for d in ACTIONS:
            dl, fl = self._ghost_proximity(node, d, ghosts, depth=6)
            danger_lvl.append(dl)
            freight_lvl.append(fl)
            power_dir.append(self._power_pellet_in_dir(node, d, pellets, depth=8))

        any_freight = any(g.mode.current == FREIGHT for g in ghosts)
        pdx, pdy, pdist = self._pellet_info(pacman.position, pellets)

        return (
            valid
            + tuple(danger_lvl)
            + tuple(freight_lvl)
            + tuple(power_dir)
            + (any_freight, pdx, pdy, pdist)
        )

    def _ghost_proximity(self, start_node, direction, ghosts, depth=6):
        """
        Walk path in `direction` up to `depth` nodes.
        Returns (danger_level, freight_level): 0=none, 1=far, 2=close.
        """
        path = {}           # node_id → hop distance
        node = start_node
        for step in range(1, depth + 1):
            nxt = node.neighbors.get(direction)
            if nxt is None:
                break
            path[id(nxt)] = step
            node = nxt

        danger = 0
        freight = 0
        for ghost in ghosts:
            nid = id(ghost.node)
            if nid in path:
                dist   = path[nid]
                level  = 2 if dist <= 3 else 1
                mode   = ghost.mode.current
                if mode == FREIGHT:
                    freight = max(freight, level)
                elif mode != SPAWN:
                    danger  = max(danger, level)
        return danger, freight

    def _power_pellet_in_dir(self, start_node, direction, pellets, depth=8):
        """True if a power pellet lies in `direction` path within `depth` nodes."""
        pp_positions = {
            (round(p.position.x), round(p.position.y))
            for p in pellets.powerpellets
        }
        node = start_node
        for _ in range(depth):
            nxt = node.neighbors.get(direction)
            if nxt is None:
                break
            if (round(nxt.position.x), round(nxt.position.y)) in pp_positions:
                return True
            node = nxt
        return False

    def _pellet_info(self, position, pellets):
        """
        Returns (dx, dy, dist_bucket) toward the nearest pellet.
        dx / dy : −1 / 0 / +1   (quantised direction)
        dist_bucket : 0=close (<4 tiles), 1=medium (<8 tiles), 2=far
        """
        if not pellets.pelletList:
            return 0, 0, 2
        best_dist = float('inf')
        best_diff = None
        for pellet in pellets.pelletList:
            diff = pellet.position - position
            d    = diff.magnitudeSquared()
            if d < best_dist:
                best_dist = d
                best_diff = diff
        if best_diff is None:
            return 0, 0, 2

        dx = 0 if abs(best_diff.x) < 8 else (1 if best_diff.x > 0 else -1)
        dy = 0 if abs(best_diff.y) < 8 else (1 if best_diff.y > 0 else -1)

        if best_dist < (4 * TILEWIDTH) ** 2:
            bucket = 0
        elif best_dist < (8 * TILEWIDTH) ** 2:
            bucket = 1
        else:
            bucket = 2
        return dx, dy, bucket

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state, valid_dirs):
        """Epsilon-greedy, restricted to valid directions."""
        if not valid_dirs:
            return STOP
        if random.random() < self.epsilon:
            return random.choice(valid_dirs)
        q = self.q_table[state]
        return max(valid_dirs, key=lambda d: q.get(d, 0.0))

    # ------------------------------------------------------------------
    # Q(λ) update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, next_valid_dirs):
        """
        Q(λ) with replacing traces.

        1. Compute TD error δ.
        2. Set e(state, action) = 1  (replacing trace).
        3. Decay all existing traces by γλ.
        4. Update every Q(s,a) with a non-zero trace.
        """
        # ── TD error ──────────────────────────────────────────────────
        if next_state is not None and next_valid_dirs:
            max_next = max(self.q_table[next_state].get(d, 0.0) for d in next_valid_dirs)
        else:
            max_next = 0.0

        delta = reward + self.gamma * max_next - self.q_table[state].get(action, 0.0)

        # ── Replacing trace for current (s, a) ───────────────────────
        if state not in self.traces:
            self.traces[state] = {a: 0.0 for a in ACTIONS}
        self.traces[state][action] = 1.0

        # ── Decay traces + Q-update for every tracked (s, a) ─────────
        decay     = self.gamma * self.lam
        to_remove = []
        for s, action_map in self.traces.items():
            for a in ACTIONS:
                e = action_map.get(a, 0.0)
                if abs(e) < 1e-5:
                    continue
                self.q_table[s][a] = self.q_table[s].get(a, 0.0) + self.alpha * delta * e
                action_map[a]      = e * decay

            if all(abs(action_map.get(a, 0.0)) < 1e-5 for a in ACTIONS):
                to_remove.append(s)

        for s in to_remove:
            del self.traces[s]

    def reset_traces(self):
        """Call at the end of every episode (death or level clear)."""
        self.traces.clear()

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
        raw          = payload.get("q_table", payload)
        self.q_table = defaultdict(_default_q, {s: dict(a) for s, a in raw.items()})
        self.epsilon = payload.get("epsilon", self.epsilon)
        print(f"[Agent] Loaded {len(self.q_table)} states (ε={self.epsilon:.4f})")
        return True
