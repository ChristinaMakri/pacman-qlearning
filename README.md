# Pac-Man Q-Learning Agent

A reinforcement learning controller for Pac-Man using **Q(λ)** (Q-learning with eligibility traces).  
The agent learns to play one level of the game autonomously, without any hard-coded rules.

---

## How it works

### Algorithm: Q(λ) with Replacing Eligibility Traces

Standard Q-learning only updates the Q-value of the **last** (state, action) pair after each step.  
Q(λ) propagates credit **backwards** through the episode using eligibility traces, updating all recently visited pairs with a decaying factor λ.

```
δ  = r + γ · max_a′ Q(s′,a′) − Q(s,a)      ← TD error
e(s,a) ← 1                                   ← replacing trace for taken action
e(s,a) ← γλ · e(s,a)                         ← decay all other traces
Q(s,a) += α_eff · δ · e(s,a)                 ← update every traced (s,a)
```

**Adaptive learning rate** per (state, action):
```
α_eff(s,a) = α / √(n(s,a) + 1)
```
This guarantees convergence: α_t → 0, Σα_t = ∞, Σα_t² < ∞.

### State Representation (22 features)

| Feature | Values | Description |
|---------|--------|-------------|
| `can_move` × 4 | bool | Whether Pac-Man can move UP/DOWN/LEFT/RIGHT |
| `ghost_danger` × 4 | 0/1/2 | Dangerous ghost in path: 0=none, 1=far (4-6 nodes), 2=close (1-3 nodes) |
| `ghost_freight` × 4 | 0/1/2 | Eatable (FREIGHT) ghost in path, same scale |
| `power_pellet` × 4 | bool | Power pellet present in that direction |
| `any_freight` | bool | Any ghost currently in FREIGHT mode (globally) |
| `pellet_dx`, `pellet_dy` | −1/0/+1 | Direction to nearest pellet |
| `pellet_dist` | 0/1/2 | Distance bucket: close / medium / far |
| `current_dir` | int | Pac-Man's current movement direction |

### Reward Signal

| Event | Reward |
|-------|--------|
| Eat regular pellet | +10 |
| Eat power pellet | +50 |
| Eat FREIGHT ghost | +200 / +400 / +800 / +1600 (chains) |
| Die from ghost | −500 |
| Node transition (step penalty) | −1 |
| Survival bonus (per node) | +0.5 |
| Danger penalty (ghost nearby) | −1 / −3 / −8 by proximity |
| FREIGHT ghost nearby | +2 |

---

## Project Structure

```
Pacman_Qlearning/
├── run.py               # Main entry point — GameController + TrainingController
├── qlearning_agent.py   # Q(λ) agent: state extraction, learning, save/load
├── pacman.py            # Modified Pac-Man: ai_direction hook + just_reached_node flag
├── constants.py         # Game constants (directions, modes, tile sizes)
├── entity.py            # Base entity movement (node graph traversal)
├── ghosts.py            # Ghost AI (Blinky, Pinky, Inky, Clyde)
├── nodes.py             # Maze node graph
├── pellets.py           # Pellet system
├── ...                  # Other original game files
└── qtable.pkl           # Saved Q-table (created after training)
```

The original game template is in `Pacman_Basic/` (kept as reference).

---

## Usage

### Install dependencies
```bash
pip install pygame numpy
```

### Train the agent
```bash
# Visual training (see the game while training)
python run.py --mode train --episodes 500

# Headless training — no display, 10-50× faster
python run.py --mode train --episodes 2000 --headless

# Save checkpoint every 200 episodes
python run.py --mode train --episodes 2000 --headless --save-every 200
```

Training progress is printed to the terminal:
```
Episode    1 | Score   1180 | Reward  -1402.0 | ε=1.0000 | States=93
Episode    2 | Score    870 | Reward  -1950.0 | ε=0.9950 | States=156
...
```

### Watch the trained agent play
```bash
python run.py --mode play
```
Loads `qtable.pkl` and runs with ε=0 (no exploration).

### Play manually
```bash
python run.py --mode human
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Initial learning rate (decays per visit) |
| `gamma` | 0.9 | Discount factor |
| `lambda` | 0.8 | Eligibility trace decay |
| `epsilon` | 1.0 → 0.05 | Exploration rate (decays by 0.995 per episode) |

---

## Files modified from original template

| File | Change |
|------|--------|
| `pacman.py` | Added `ai_direction` (agent sets direction) and `just_reached_node` flag |
| `run.py` | Added `TrainingController` subclass with reward tracking, headless mode, episode management |
| `qlearning_agent.py` | New file — full Q(λ) agent implementation |
