import math
from typing import List, Tuple
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def solve_rectangles_packing_cpsat(
    rectangles: List[Tuple[float, float]],
    scale_factor: int = 10,
    time_limit_sec: float = 30.0
) -> Tuple[float, List[Tuple[float, float, float, float, bool]]]:
    """
    Finds the minimum square container side length L that fits all given rectangles,
    allowing 90-degree rotations.
    
    :param rectangles: List of (width, height) tuples.
    :param scale_factor: Integer scaling factor for decimal precision.
    :param time_limit_sec: Max runtime in seconds.
    :return: (optimal_L, [(x, y, placed_w, placed_h, rotated), ...])
    """
    model = cp_model.CpModel()
    n = len(rectangles)

    # Scale to integer dimensions
    scaled_rects = [
        (int(round(w * scale_factor)), int(round(h * scale_factor)))
        for w, h in rectangles
    ]

    total_area = sum(w * h for w, h in scaled_rects)
    min_L = int(math.ceil(math.sqrt(total_area)))
    max_L = sum(max(w, h) for w, h in scaled_rects)

    # Container size variable
    L = model.NewIntVar(min_L, max_L, "container_L")

    x_intervals = []
    y_intervals = []
    x_coords = []
    y_coords = []
    dx_vars = []
    dy_vars = []
    rotated_vars = []

    for i, (w, h) in enumerate(scaled_rects):
        # Rotation flag: 1 = rotated 90 degrees, 0 = original orientation
        r = model.NewBoolVar(f"rot_{i}")
        rotated_vars.append(r)

        # If width == height, freeze rotation to prune redundant states
        if w == h:
            model.Add(r == 0)

        # Dynamic width and height variables
        dx = model.NewIntVar(min(w, h), max(w, h), f"dx_{i}")
        dy = model.NewIntVar(min(w, h), max(w, h), f"dy_{i}")
        
        # Link dx and dy with rotation:
        # dx == w + r * (h - w)
        # dy == h + r * (w - h)
        model.Add(dx == w + r * (h - w))
        model.Add(dy == h + r * (w - h))

        dx_vars.append(dx)
        dy_vars.append(dy)

        # Coordinate positions
        x = model.NewIntVar(0, max_L, f"x_{i}")
        y = model.NewIntVar(0, max_L, f"y_{i}")
        x_coords.append(x)
        y_coords.append(y)

        x_end = model.NewIntVar(0, max_L, f"x_end_{i}")
        y_end = model.NewIntVar(0, max_L, f"y_end_{i}")

        model.Add(x_end == x + dx)
        model.Add(y_end == y + dy)

        # Bounding container bounds
        model.Add(x_end <= L)
        model.Add(y_end <= L)

        # Interval variables with dynamic size
        x_interval = model.NewIntervalVar(x, dx, x_end, f"x_int_{i}")
        y_interval = model.NewIntervalVar(y, dy, y_end, f"y_int_{i}")

        x_intervals.append(x_interval)
        y_intervals.append(y_interval)

    # 2D non-overlapping placement constraint
    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Symmetry breaking: place rectangle with largest area in the bottom-left quadrant
    areas = [w * h for w, h in scaled_rects]
    largest_idx = areas.index(max(areas))
    model.Add(x_coords[largest_idx] <= (L - dx_vars[largest_idx]) // 2)
    model.Add(y_coords[largest_idx] <= (L - dy_vars[largest_idx]) // 2)

    # Objective: Minimize container side length
    model.Minimize(L)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_workers = 8

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        optimal_L = solver.Value(L) / scale_factor
        placements = []
        for i in range(n):
            px = solver.Value(x_coords[i]) / scale_factor
            py = solver.Value(y_coords[i]) / scale_factor
            p_w = solver.Value(dx_vars[i]) / scale_factor
            p_h = solver.Value(dy_vars[i]) / scale_factor
            is_rot = bool(solver.Value(rotated_vars[i]))
            placements.append((px, py, p_w, p_h, is_rot))

        status_text = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        print(f"Solver Status: {status_text}")
        return optimal_L, placements
    else:
        raise RuntimeError("No feasible layout found.")


def visualize_rectangles(container_l: float, placements: List[Tuple[float, float, float, float, bool]]):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, container_l * 1.05)
    ax.set_ylim(0, container_l * 1.05)
    ax.set_aspect("equal")

    # Bounding square
    container_patch = patches.Rectangle(
        (0, 0), container_l, container_l,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"
    )
    ax.add_patch(container_patch)

    cmap = plt.cm.get_cmap("tab20", len(placements))
    for i, (x, y, w, h, rotated) in enumerate(placements):
        rect = patches.Rectangle(
            (x, y), w, h,
            facecolor=cmap(i), edgecolor="black", alpha=0.8, linewidth=1.2
        )
        ax.add_patch(rect)
        rot_indicator = " (R)" if rotated else ""
        ax.text(
            x + w / 2, y + h / 2, f"{w:.1f}×{h:.1f}{rot_indicator}",
            color="black", weight="bold", fontsize=7.5, ha="center", va="center"
        )

    plt.title(f"Optimized Rectangles with Rotation: L = {container_l:.2f}")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.show()


if __name__ == "__main__":
    # Test instances of rectangles (width, height)
    input_rectangles = [
        (12.0, 4.0),
        (8.0, 5.0),
        (6.0, 6.0),
        (10.0, 3.0),
        (7.0, 4.0),
        (9.0, 2.0),
        (5.0, 3.0),
        (4.0, 2.0),
    ]

    opt_l, coords = solve_rectangles_packing_cpsat(input_rectangles, scale_factor=10, time_limit_sec=15.0)
    print(f"Minimum Bounding Square Side: {opt_l}")
    
    total_rect_area = sum(w * h for w, h in input_rectangles)
    print(f"Packing Density: {(total_rect_area / (opt_l ** 2)) * 100:.2f}%")

    # visualize_rectangles(opt_l, coords)
