from config import SolverConfig
from run import main

config = SolverConfig(
    n_squares=11,             # Number of unit squares to pack
    square_size=1.0,          # Unit square edge length
    max_iterations=5000,      # Annealing steps
    init_temp=1.5,            # Starting temperature
    cooling_rate=0.997,       # Decay factor per iteration
    penalty_overlap=1e5,      # Overlap penalty multiplier
    random_seed=42
)
