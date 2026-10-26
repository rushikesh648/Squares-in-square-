# Squares in a Square (Kaggle Optimization Solver)

A modular Python framework to solve the classic geometric packing problem: packing $N$ congruent unit squares into a bounding square of minimal side length $L$ without overlaps, allowing continuous 2D rotations and translations.

---

## Overview

The problem is non-convex and NP-hard with multiple local minima. This repository implements a hybrid optimization pipeline:

1. **Heuristic Grid Seeding:** Generates initial spatial layouts on regular grids with random rotation orientations ($\theta \in [0, \pi/2]$).
2. **Global Simulated Annealing (SA):** Explores configuration space by perturbing position coordinates $(x, y)$ and rotation angles $\theta$, penalizing overlap area heavily.
3. **Local Polish (Powell / L-BFGS-B):** Smooths boundaries and eliminates micro-overlaps.
4. **Validation & Normalization:** Aligns coordinates to origin $(0, 0)$, validates zero overlap tolerance via exact polygon intersection (Shapely), and generates submission CSVs and layout visualizations.

---

## Directory Structure

```text
squares_in_square/
├── requirements.txt         # Project dependencies
├── config.py                # Hyperparameters and experiment configurations
├── geometry.py              # Corner transforms, Shapely polygons, overlap checks
├── optimizer.py             # Simulated annealing & local continuous minimization
├── visualizer.py            # High-resolution packing diagram generator
├── evaluate_and_submit.py   # Coordinate alignment, scoring, and CSV exporter
├── run.py                   # Main execution pipeline
└── outputs/                 # Generated solutions and plots
