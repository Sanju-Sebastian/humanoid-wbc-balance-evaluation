# Humanoid WBC Balance Evaluation

**Evaluation of Whole-Body Control Architectures for Humanoid Balance  
in Simulation with Learning-Based Residual Compensation**

Master's Thesis — MSc Robotics and Automation  
University West, Sweden · 2026  
**Author: Sanju N Sebastian**

> This repository contains my independent implementation and all experimental 
> scripts from my Master's thesis. A parallel implementation by my thesis partner 
> Tharindu Dananjaya (Pinocchio-based HQP on Ubuntu/ROS2) is available at 
> [thul0002/Master_Thesis_EvaluationOf_WBC_Architectures_for_Humanoid_Robot](https://github.com/thul0002/Master_Thesis_EvaluationOf_WBC_Architectures_for_Humanoid_Robot).
> The literature review was developed jointly. All controller implementations, 
> simulation experiments, analysis, and results in this repository are my own 
> independent work.

---

## Overview

This project presents a simulation-based comparative evaluation of four whole-body 
control (WBC) architectures for humanoid robot balance, implemented for the 
**Unitree H1** in **MuJoCo** and evaluated under a stochastic multi-push disturbance 
protocol. A supervised residual MLP was designed, ablated across three training 
formulations, and shown to produce statistically significant improvements in balance 
stability across all four base controllers.

**Environment:** Windows · MuJoCo only · No Pinocchio or ROS required

---

## Key Results

| Controller | Stability Rate | Mean Survival (s) |
|---|---|---|
| Weighted QP | 51.5% | 7.90 |
| Hierarchical QP | 51.5% | 7.89 |
| Passivity-Based | 55.0% | 7.99 |
| LQR-on-LIPM | 57.5% | 8.13 |
| **Passivity + MLP v3** | **62.5%** | **8.46** |

- **WQP = HQP** across all 200 independent seeds (mean CoM difference: 0.50 mm),  
  while HQP incurs 1.89× the computational cost — no architectural benefit in  
  static balance push-recovery
- **Passivity-based controller** produces 11% lower CoM displacement on stable  
  trials due to energy-dissipating damping injection layer
- **Residual MLP v3** improves passivity baseline by +7.5 pp (p < 0.001,  
  McNemar's exact test) — zero previously stable trials destabilised
- **LIPM-based supervised loss fails** due to unstable discrete-time eigenvalue  
  (1.003) — theoretically identified and numerically verified

---

## Repository Structure

```text

├── controllers/
│   ├── wbc_weighted_qp_experiment_final_v2.py   # Weighted QP controller
│   ├── wbc_hierarchial_qp_experiment_final.py   # Hierarchical QP controller
│   ├── wbc_passivity_experiment.py              # Passivity-based controller
│   └── lqr_lipm_controller.py                  # LQR on LIPM (reduced-order)
│
├── disturbance/
│   └── disturbance.py          # Seeded stochastic push injection
│                               # 10 body sites · 20–250 N · log-uniform sampling
│
├── residual_mlp/
│   ├── residual_mlp.py         # MLP v1 — LIPM prediction loss (null result)
│   ├── residual_mlp_v2.py      # MLP v3 — passivity base controller
│   ├── residual_mlp_lqr.py     # MLP v3 — LQR base controller
│   ├── residual_mlp_wqp.py     # MLP v3 — WQP base controller
│   ├── residual_mlp_hqp.py     # MLP v3 — HQP base controller
│   └── cross_controller_distillation.py   # WQP trained to replicate passivity torques
│
├── analysis/
│   ├── analysis.py             # Part 1a — deterministic grid analysis
│   └── analysis_part1b.py      # Part 1b — stochastic batch analysis + all figures
│
└── tests/
    └── hqp_divergence_test.py  # Verifies HQP priority structure is active and correct
```

---

## Experimental Protocol

**Part 1a — Deterministic Step-Push Grid**  
6 force levels × 2 body sites × 3 directions × 5 trials = 180 trials per controller.  
Used for qualitative baseline characterisation only.

**Part 1b — Stochastic Multi-Push Evaluation**  
200 independent seeded trials per controller (800 total).  
Each seed determines: push count (Poisson, mean 2.5), force magnitude (log-uniform  
20–250 N), duration (0.05–0.20 s), direction (uniform azimuth), and body site  
(weighted across 10 sites). Genuine statistical independence — same seed always  
produces the same trial.

**Outcome Classification**  
Three categories: **Stable** (CoM returns within 20 mm), **Fell** (pelvis below  
0.65 m), **Neither** (survived but did not recover). The Neither category captures  
lingering quasi-equilibria absent from binary frameworks in prior benchmarking work.

---

## Residual MLP Architecture

- **Input:** 260-dimensional (5 frames × 52 dims — joint states, CoM error,  
  CoM velocity, base controller torques)
- **Architecture:** 3 hidden layers × 128 units, ReLU activations, tanh output  
  bounded to ±20 Nm
- **Applied to:** 6 balance-critical joints only (ankle pitch, hip pitch, hip roll)
- **Training target (v3):** Pitch-rate behaviour cloning derived from ankle strategy

---

## Setup

```bash
# Clone this repository
git clone https://github.com/Sanju-Sebastian/humanoid-wbc-balance-evaluation.git
cd humanoid-wbc-balance-evaluation

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install mujoco numpy torch scipy pandas cvxpy osqp
```

---

## Running the Controllers

Each controller script runs independently:

```bash
python controllers/wbc_passivity_experiment.py
python controllers/lqr_lipm_controller.py
python controllers/wbc_weighted_qp_experiment_final_v2.py
```

To run the MLP-augmented controller, ensure the trained model weights are present  
in `residual_mlp/` and run:

```bash
python residual_mlp/residual_mlp_v2.py
```

---

## Technical Stack

Python · MuJoCo · PyTorch · CVXPY · OSQP · NumPy · SciPy · Pandas

---

## Citation

If you use this work, please cite:
Sanju N Sebastian, "Evaluation of Whole-Body Control Architectures for Humanoid
Balance in Simulation with Learning-Based Residual Compensation,"
Master's Thesis, University West, Sweden, 2026.

---

## Contact

**Sanju N Sebastian**  
[linkedin.com/in/sanju-n-sebastian](https://linkedin.com/in/sanju-n-sebastian)  
sanju.n.sebastian@gmail.com
