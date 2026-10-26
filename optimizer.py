"""Solvers: Grid Initialization, Simulated Annealing, and Gradient Refinement."""
import numpy as np
from scipy.optimize import minimize
from geometry import get_square_vertices, create_square_polygon, compute_total_overlap, compute_bounding_square_size
from config import SolverConfig

def initialize_grid(n: int, side: float = 1.0) -> np.ndarray:
    """
    Heuristic initialization: arranges n squares in a coarse regular grid.
    Returns 1D state array: [x0, y0, t0, x1, y1, t1, ...]
    """
    cols = int(np.ceil(np.sqrt(n)))
    state = []
    spacing = side * 1.15
    for i in range(n):
        r = i // cols
        c = i % cols
        x = c * spacing + side
        y = r * spacing + side
        theta = np.random.uniform(0, np.pi / 2) # symmetry reduced to [0, pi/2]
        state.extend([x, y, theta])
    return np.array(state, dtype=np.float64)

def objective_fn(params: np.ndarray, n: int, side: float, penalty: float = 1e4) -> float:
    """
    Cost function: Bounding square length + heavily penalized overlap area.
    """
    polygons = []
    for i in range(n):
        x, y, theta = params[3 * i : 3 * i + 3]
        polygons.append(create_square_polygon(x, y, theta, side))
        
    side_len = compute_bounding_square_size(polygons)
    overlap = compute_total_overlap(polygons)
    
    return side_len + (penalty * overlap)

def solve_simulated_annealing(config: SolverConfig) -> tuple[np.ndarray, float]:
    """Simulated Annealing exploration for global layout search."""
    np.random.seed(config.random_seed)
    n = config.n_squares
    current_state = initialize_grid(n, config.square_size)
    current_energy = objective_fn(current_state, n, config.square_size, config.penalty_overlap)
    
    best_state = current_state.copy()
    best_energy = current_energy
    temp = config.init_temp
    
    for _ in range(config.max_iterations):
        step_scale = max(0.01, temp)
        candidate = current_state.copy()
        
        # Perturb one random square's pose
        idx = np.random.randint(0, n)
        candidate[3 * idx] += np.random.normal(0, 0.1 * step_scale)
        candidate[3 * idx + 1] += np.random.normal(0, 0.1 * step_scale)
        candidate[3 * idx + 2] = (candidate[3 * idx + 2] + np.random.normal(0, 0.2 * step_scale)) % (np.pi / 2)
        
        cand_energy = objective_fn(candidate, n, config.square_size, config.penalty_overlap)
        delta = cand_energy - current_energy
        
        if delta < 0 or np.exp(-delta / max(temp, 1e-8)) > np.random.rand():
            current_state = candidate
            current_energy = cand_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = current_state.copy()
                
        temp *= config.cooling_rate
        
    return best_state, best_energy

def refine_solution(best_state: np.ndarray, config: SolverConfig) -> np.ndarray:
    """Local polish via L-BFGS-B / Powell algorithm."""
    res = minimize(
        fun=objective_fn,
        x0=best_state,
        args=(config.n_squares, config.square_size, config.penalty_overlap * 2),
        method="Powell",
        options={"maxiter": 500, "disp": False}
    )
    return res.x
