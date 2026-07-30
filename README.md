# traffic-flow-nonlocal-model

Numerical implementation of **nonlocal traffic flow models** on road networks.
The code computes **steady-state** solutions based on Riccati-type density profiles, buffer mass balance at junctions/crossroads, and (optionally) travel times and plots of density, nonlocal impact, and velocity.

---

## Requirements

- Python 3.10+ (recommended)
- `numpy`
- `scipy`
- `matplotlib`
- `networkx` (needed for `network_classes.py` plotting)

Install with:

```bash
pip install numpy scipy matplotlib networkx
```

---

## Repository structure

| File | Role |
|---|---|
| `utils.py` | Shared helpers and default parameters for the modular network code |
| `network_classes.py` | Object-oriented network framework (`Crossroad`, `Street`, `Network`, …) |
| `1-1.py` | Standalone example: one incoming road → one outgoing road |
| `n-m.py` | Standalone example: several incoming roads → several outgoing roads |
| `linear_network.py` | Standalone example: linear chain of roads and junctions |

The three scripts `1-1.py`, `n-m.py`, and `linear_network.py` are **self-contained demos** (they define their own `Road` / `Junction` classes).
`network_classes.py` + `utils.py` is the **reusable framework** for building larger networks (grids, Braess, custom topologies).

---

## `utils.py`

Contains:

- `riccati(rho_l, rho_r, lambda_, eta, x)` — closed-form density profile $q(x)$ on a street
- `default_params` — default street parameters $(\lambda, \eta, L, q_{\max})$
- `params_for_short_street` — same, but with small $L$ (useful for short links)
- `rho_l_in`, `rho_r_out` — prescribed boundary densities at origins / destinations

Used by `network_classes.py`.

---

## `network_classes.py`

Modular implementation of a full traffic network.

### Main classes

- **`Crossroad`** — network node (origin / interior / destination)
- **`Street`** — directed edge with parameters; computes $q(x)$, $W(x)$, $v(x)$, travel time
- **`Commodity`** — path through crossroads (routing; currently optional / simple)
- **`Network`** — collects the graph and solves the steady-state system $F(b)=0$ for buffer loads $b$

### Typical workflow

1. Choose street parameters (or use `utils.default_params`)
2. Create crossroads and connect them with street templates
3. Build a `Network`
4. Call `network.solve()` to obtain $b^\star$
5. Plot / evaluate densities, velocities, nonlocal impact, travel times

Minimal example:

```python
import utils
from network_classes import Crossroad, Street, Network, clear_crossroad_list_and_street_list

s0 = Street(utils.default_params)
cr0 = Crossroad()
cr1 = cr0.create_and_add_successor(s0)
cr2 = cr1.create_and_add_successor(s0)

network = Network(Crossroad.crossroad_list, Street.street_list)
b_star = network.solve()
print(b_star)
```

### Built-in examples (in `__main__`)

In `network_classes.py` you can run selected demo functions, for example:

- `showcase_network()` — solve a $3\times 3$ grid and save plots to `images/`
- `solve_braexes_network()` — Braess network with / without the extra street
- `solve_line_like_braess()` — short middle street in a line

Run:

```bash
python network_classes.py
```

(which demo runs depends on what is uncommented in `if __name__ == "__main__"`).

### Plot helpers

Functions such as:

- `draw_network_grid` / `draw_network_grid_colored`
- `plot_network_density`
- `plot_network_nonlocal_impact`
- `plot_network_velocity`
- `plot_street_travel_times`

can either show figures or save them via `show=False` and `save_path=...`.

---

## Standalone junction demos

These scripts solve a single junction (or a short chain of junctions), then write plots into a folder next to the script.

### `1-1.py` — one-to-one junction

Topology:

```text
R1 ---> [Junction] ---> R2
```

- Prescribes $\rho_l$ on the incoming road and $\rho_r$ on the outgoing road
- Solves for the buffer / unknown boundary values
- Saves plots under `1-1/`

Run:

```bash
python 1-1.py
```

### `n-m.py` — many-to-many junction

Topology (default setup in `main`):

```text
R1 \                 / R4
R2 ----> [Junction] ----> R5
R3 /                 \ R6
```

- Several incoming and outgoing roads
- Routing / commodities determine how mass is split
- Saves plots under a result folder created by the script

Run:

```bash
python n-m.py
```

### `linear_network.py` — linear chain

Topology (default setup):

```text
R1 ---> [J1] ---> R2 ---> [J2] ---> R3
```

- Two junctions in a row
- Useful to study successive couplings and short middle roads
- Saves plots into a result folder

Run:

```bash
python linear_network.py
```

### Output of the standalone demos

Each demo typically creates:

- per-road plots: `density.png`, `nonlocal_density.png`, `velocity.png`
- summary plots over all roads: `density_all_roads.png`, `nonlocal_density_all_roads.png`, `velocity_all_roads.png`

---

## Model quantities (short)

| Symbol | Meaning |
|---|---|
| $q(x)$ | density along a road |
| $W(x)$ | nonlocal impact / perceived congestion ahead |
| $v(x)=e^{-\lambda W(x)}$ | velocity |
| $b$ | buffer load(s) at the beginning of outgoing roads |
| $\rho_l$, $\rho_r$ | left / right boundary densities of a road |
| travel time $\daleth$ | $\int_0^L \frac{1}{v(x)}\,dx$ (in `network_classes.py`) |

---

## Notes

- Origins use a prescribed inflow density (`rho_l_in` in `utils.py` for the modular code).
- Destinations use a prescribed outflow density (`rho_r_out`).
- Steady states are found with `scipy.optimize.root`.
- Random seeds are fixed in the standalone scripts for reproducibility.
