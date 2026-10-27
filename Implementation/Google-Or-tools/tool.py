import math
from typing import List, Tuple
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def solve_square_packing_cpsat(
    square_sizes: List[float], 
    scale_factor: int = 100, 
    time_limit_sec: float = 30.0
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Finds the minimum container square size L that fits all given squares using CP-SAT.
    
    :param square_sizes: List of side lengths of each square.
    :param scale_factor: Integer scaling factor for continuous precision (100 = 2 decimals).
    :param time_limit_sec: Max solver runtime in seconds.
    :return: (optimal_L, [(x, y, size), ...])
    """
    model = cp_model.CpModel()
    n = len(square_sizes)
    
    # Scale sizes to integers
    int_sizes = [int(round(s * scale_factor)) for s in square_sizes]
    
    # Mathematical lower & upper bounds for container side L
    total_area = sum(s ** 2 for s in int_sizes)
    min_L = int(math.ceil(math.sqrt(total_area)))
    max_L = sum(int_sizes)

    # Variable: Bounding square side length L
    L = model.NewIntVar(min_L, max_L, "container_L")

    x_intervals = []
    y_intervals = []
    x_coords = []
    y_coords = []

    for i in range(n):
        s = int_sizes[i]
        
        # Position variables: 0 <= x_i <= max_L - s
        x = model.NewIntVar(0, max_L - s, f"x_{i}")
        y = model.NewIntVar(0, max_L - s, f"y_{i}")
        x_coords.append(x)
        y_coords.append(y)
        
        # Interval variables for dimension (start, size, end)
        x_end = model.NewIntVar(s, max_L, f"x_end_{i}")
        y_end = model.NewIntVar(s, max_L, f"y_end_{i}")
        
        model.Add(x_end == x + s)
        model.Add(y_end == y + s)
        
        x_interval = model.NewIntervalVar(x, s, x_end, f"x_interval_{i}")
        y_interval = model.NewIntervalVar(y, s, y_end, f"y_interval_{i}")
        
        x_intervals.append(x_interval)
        y_intervals.append(y_interval)

        # Container boundary enforcement: x_i + s <= L and y_i + s <= L
        model.Add(x_end <= L)
        model.Add(y_end <= L)

    # Global constraint: rectangles do not overlap in 2D space
    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Symmetry breaking: place the largest square in the bottom-left quadrant
    largest_idx = int_sizes.index(max(int_sizes))
    model.Add(x_coords[largest_idx] <= (L - int_sizes[largest_idx]) // 2)
    model.Add(y_coords[largest_idx] <= (L - int_sizes[largest_idx]) // 2)

    # Objective: Minimize container side length L
    model.Minimize(L)

    # Solver setup
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_workers = 8  # Parallel branch-and-bound search
    
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        optimal_L = solver.Value(L) / scale_factor
        placements = []
        for i in range(n):
            placed_x = solver.Value(x_coords[i]) / scale_factor
            placed_y = solver.Value(y_coords[i]) / scale_factor
            placements.append((placed_x, placed_y, square_sizes[i]))
        
        status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE (Time Limit Reached)"
        print(f"Solver Status: {status_name}")
        return optimal_L, placements
    else:
        raise RuntimeError("No feasible placement found within limits.")


def plot_packing(container_l: float, placements: List[Tuple[float, float, float]]):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, container_l * 1.05)
    ax.set_ylim(0, container_l * 1.05)
    ax.set_aspect("equal")

    # Bounding square
    container_patch = patches.Rectangle(
        (0, 0), container_l, container_l,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"
    )
    ax.add_patch(container_patch)

    # Internal squares
    cmap = plt.cm.get_cmap("tab20", len(placements))
    for i, (x, y, s) in enumerate(placements):
        rect = patches.Rectangle(
            (x, y), s, s,
            facecolor=cmap(i), edgecolor="black", alpha=0.8, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(
            x + s / 2, y + s / 2, f"{s:.1f}",
            color="black", weight="bold", fontsize=8, ha="center", va="center"
        )

    plt.title(f"CP-SAT Exact Solution: Container L = {container_l:.2f}")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Test instance with 10 squares
    sizes = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0]

    opt_l, coords = solve_square_packing_cpsat(sizes, scale_factor=10, time_limit_sec=15.0)
    print(f"Minimum Bounding Square Side: {opt_l}")
    print(f"Packing Density: {(sum(s**2 for s in sizes) / (opt_l**2)) * 100:.2f}%")

    # plot_packing(opt_l, coords)
