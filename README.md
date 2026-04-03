# Drone racing in Isaac Lab with PPO

**Aria X. Shi** · University of Pennsylvania (MSE, PICS) · with **Jiacheng Zhu** (MSE CIS)

End-to-end reinforcement learning stack for **autonomous quadcopter racing** in **NVIDIA Isaac Lab / Isaac Sim**: parallel **PPO** training (actor–critic, GAE, clipped surrogate), custom **reward shaping** to match real racing behavior, and **geometry-based gate validation** to stop reward hacking. This repo is the code behind the technical report in [`Report_Overleaf/`](Report_Overleaf/) (LaTeX source).

**License:** [BSD-3-Clause](LICENSE)

---

## Implementation

- **PPO implementation** in a vendored **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** tree (`src/third_parties/rsl_rl_local/`): on-policy rollouts, **Generalized Advantage Estimation (GAE)**, clipped policy objective, value loss, and entropy regularization. Hyperparameters tuned for stability vs. sample efficiency (e.g. learning rate **1e-3**, clip **ε = 0.2**, discount **γ = 0.998**).
- **Reward engineering** that moves from a naive distance-to-goal baseline to a **racing-aligned objective**: potential-based distance progress, **velocity progress** along the goal direction, upright/smoothness/centering terms, **speed** and **time** incentives, and **exit alignment** toward the next gate for smoother cornering.
- **Diagnosis and fixes for reward hacking**: identified **reverse gate crossing** (wrong direction then re-enter for credit) caused by threshold-only gate logic; implemented **direction-aware, geometry-based gate traversal** (correct half-space crossing + aperture bounds) with **reverse** and **wrong-side** penalties.
- **Second-order shaping** after penalties created a **detour / fly-around** local minimum: introduced **ghost waypoints** above/ahead of the gate to force an apex-style path and **constraint masking** so acrobatic segments are not fighting flat-flight centering rewards.
- **Training and evaluation pipeline** for `Isaac-Quadcopter-Race-v0`: thousands of parallel envs for throughput, scripted **playback** with optional **video** export for demos and analysis.

**Stack:** Python · PyTorch · Isaac Lab / Isaac Sim · Gymnasium · RSL-RL-style PPO · parallel vec envs

---

## Overview

| Piece | Detail |
|--------|--------|
| Task | `Isaac-Quadcopter-Race-v0` — sequential gates, racing-style flight |
| Algorithm | PPO (actor–critic, GAE, clipped surrogate + value + entropy) |
| Simulator | Isaac Lab extension under `src/isaac_quad_sim2real/` |
| RL library | Local RSL-RL fork (see `train_race.py` / `play_race.py` path injection) |

---

## Prerequisites

- **NVIDIA GPU** suitable for Isaac Sim ([requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html)).
- **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** working with Isaac Sim.
- Python environment as recommended by your Isaac Lab install.

---

## Repository layout

Clone this repository **next to** your Isaac Lab root so a typical extension workflow stays consistent (same parent directory as `IsaacLab`).

```text
~/workspace/
├── IsaacLab/
└── ese651_project/    # this repository
```

Register or wire the extension per [Isaac Lab: extensions](https://isaac-sim.github.io/IsaacLab/main/source/extensions/create_new_extension.html) if needed; many setups only require the Isaac Lab env and running scripts from this repo’s root.

---

## Training

From the **repository root**, with the Isaac Lab environment active:

```bash
python scripts/rsl_rl/train_race.py \
  --task Isaac-Quadcopter-Race-v0 \
  --num_envs 8192 \
  --max_iterations 5000 \
  --headless
```

Tune `--num_envs` and `--max_iterations` to your GPU. Training uses the local RSL-RL under `src/third_parties/rsl_rl_local/`.

---

## Evaluation / playback

```bash
python scripts/rsl_rl/play_race.py \
  --task Isaac-Quadcopter-Race-v0 \
  --num_envs 1 \
  --load_run YYYY-MM-DD_HH-MM-SS \
  --checkpoint best_model.pt \
  --headless \
  --video \
  --video_length 800
```

Use the timestamped folder under `logs/rsl_rl/quadcopter_direct/` (or your log root). Drop `--headless` for the Isaac Sim UI.

---

## Project structure

| Path | Role |
|------|------|
| `config/extension.toml` | Isaac Lab extension metadata |
| `src/isaac_quad_sim2real/` | Race task, env config, reward / gate logic |
| `src/third_parties/rsl_rl_local/` | RSL-RL + PPO, runners, storage |
| `scripts/rsl_rl/` | `train_race.py`, `play_race.py`, CLI helpers |
| `Report_Overleaf/` | LaTeX report (PPO, reward design, results) |

---

## Running scripts

Run commands from the **repository root**. Scripts import `src.isaac_quad_sim2real`; keep cwd at the project root or adjust `PYTHONPATH` if your layout differs. Packaging metadata: `pyproject.toml`, `config/extension.toml`.

---

## Report

The write-up walks through **PPO**, the **reward redesign** (baseline → progress terms → direction constraints → ghost points / masking), and **before/after** behavior (gate-pass curves, trajectory figures). Source: [`Report_Overleaf/main.tex`](Report_Overleaf/main.tex).

---

## Acknowledgments

Built on **Isaac Lab**, **Isaac Sim**, and **RSL-RL**. The **race task**, **environment**, and **extension** layout are based on [isaac_quad_sim2real](https://github.com/Jirl-upenn/isaac_quad_sim2real) and ship with this repo once cloned;
