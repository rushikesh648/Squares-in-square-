"""Main entry point to execute search, refinement, and artifact generation."""
import os
from config import SolverConfig
from optimizer import solve_simulated_annealing, refine_solution
from visualizer import plot_packing
from evaluate_and_submit import export_submission

def main():
    config = SolverConfig(
        n_squares=11,          # Set your target N
        max_iterations=4000,
        random_seed=42,
        output_dir="outputs"
    )
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[*] Starting Global Annealing for N={config.n_squares}...")
    sa_state, sa_energy = solve_simulated_annealing(config)
    print(f"[*] Annealing completed. Energy: {sa_energy:.4f}")

    print("[*] Running Local Polishing Optimizer...")
    refined_state = refine_solution(sa_state, config)

    # Export outputs
    sub_csv = os.path.join(config.output_dir, "submission.csv")
    plot_img = os.path.join(config.output_dir, "packing_solution.png")
    
    export_submission(refined_state, config.n_squares, config.square_size, sub_csv)
    plot_packing(refined_state, config.n_squares, config.square_size, plot_img)

if __name__ == "__main__":
    main()
