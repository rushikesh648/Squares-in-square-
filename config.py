"""Configuration parameters for solver and experiment settings."""
from dataclasses import dataclass

@dataclass
class SolverConfig:
    n_squares: int = 11            # Default N squares to pack
    square_size: float = 1.0       # Side length of each inner square
    max_iterations: int = 3000     # Simulated annealing / SLSQP iterations
    init_temp: float = 1.0         # SA start temperature
    cooling_rate: float = 0.995    # SA cooling multiplier
    penalty_overlap: float = 1e4   # Overlap violation penalty weight
    penalty_boundary: float = 1e4  # Container overflow penalty weight
    random_seed: int = 42
    output_dir: str = "outputs"
